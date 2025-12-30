"""
多模态 RQ-VAE
一个模型包含文本和图像两个模态的 RQ
支持 LoRA 共享 A 矩阵 + InfoNCE 对齐
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from info_nce import InfoNCE

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class MMRQVAE(nn.Module):
    """多模态 RQ-VAE：文本 + 图像"""
    
    def __init__(self,
                 text_dim=4096,
                 image_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0,
                 use_lora=False,
                 lora_rank=8,
                 lora_alpha=1.0,
                 lora_layers=None,
                 align_weight=0.01,
                 lora_reg_weight=1e-4,
                 lora_ortho_weight=1e-5,
                 use_recursive_lora=False
        ):
        super(MMRQVAE, self).__init__()

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha  # Fixed to 1.0, use regularization instead
        self.align_weight = align_weight
        self.lora_reg_weight = lora_reg_weight
        self.lora_ortho_weight = lora_ortho_weight
        self.use_recursive_lora = use_recursive_lora

        # ========== 文本模态 ==========
        self.text_encode_layer_dims = [self.text_dim] + self.layers + [self.e_dim]
        self.text_encoder = MLPLayers(
            layers=self.text_encode_layer_dims,
            dropout=self.dropout_prob, bn=self.bn
        )
        self.text_rq = ResidualVectorQuantizer(
            num_emb_list, e_dim,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
            use_linear=use_linear,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_layers=lora_layers,
            use_recursive_lora=use_recursive_lora
        )
        self.text_decode_layer_dims = self.text_encode_layer_dims[::-1]
        self.text_decoder = MLPLayers(
            layers=self.text_decode_layer_dims,
            dropout=self.dropout_prob, bn=self.bn
        )

        # ========== 图像模态 ==========
        self.image_encode_layer_dims = [self.image_dim] + self.layers + [self.e_dim]
        self.image_encoder = MLPLayers(
            layers=self.image_encode_layer_dims,
            dropout=self.dropout_prob, bn=self.bn
        )
        self.image_rq = ResidualVectorQuantizer(
            num_emb_list, e_dim,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
            use_linear=use_linear,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_layers=lora_layers,
            use_recursive_lora=use_recursive_lora
        )
        self.image_decode_layer_dims = self.image_encode_layer_dims[::-1]
        self.image_decoder = MLPLayers(
            layers=self.image_decode_layer_dims,
            dropout=self.dropout_prob, bn=self.bn
        )
        if use_lora:
            self.text_rq.share_lora_A_with(self.image_rq)
            print("✓ LoRA A: Shared common part + Modality-specific residuals")

        # ========== InfoNCE ==========
        self.infonce = InfoNCE()

    def forward(self, text_x, image_x, use_sk=True):

        # 文本模态
        text_e = self.text_encoder(text_x)
        z_q_text, text_rq_loss, text_indices, _ = self.text_rq(text_e, use_sk=use_sk)
        text_out = self.text_decoder(z_q_text)

        # 图像模态
        image_e = self.image_encoder(image_x)
        z_q_image, image_rq_loss, image_indices, _ = self.image_rq(image_e, use_sk=use_sk)
        image_out = self.image_decoder(z_q_image)

        return (text_out, image_out, 
                text_rq_loss, image_rq_loss,
                text_indices, image_indices,
                z_q_text, z_q_image)

    @torch.no_grad()
    def get_indices(self, text_x=None, image_x=None, use_sk=False):
        """获取码本索引"""
        text_indices, image_indices = None, None
        
        if text_x is not None:
            text_e = self.text_encoder(text_x)
            _, _, text_indices, _ = self.text_rq(text_e, use_sk=use_sk)
        
        if image_x is not None:
            image_e = self.image_encoder(image_x)
            _, _, image_indices, _ = self.image_rq(image_e, use_sk=use_sk)
        
        return text_indices, image_indices

    def compute_loss(self, text_out, image_out, 
                     text_rq_loss, image_rq_loss,
                     z_q_text, z_q_image,
                     text_x, image_x):
        """
        计算损失
        
        total_loss = recon_loss + quant_loss + align_weight * align_loss
        
        其中 align_loss = InfoNCE(z_q_text, z_q_image) + InfoNCE(z_q_image, z_q_text)
        """
        # 重建损失
        if self.loss_type == 'mse':
            loss_recon_text = F.mse_loss(text_out, text_x, reduction='mean')
            loss_recon_image = F.mse_loss(image_out, image_x, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon_text = F.l1_loss(text_out, text_x, reduction='mean')
            loss_recon_image = F.l1_loss(image_out, image_x, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_recon = loss_recon_text + loss_recon_image
        
        # 量化损失
        loss_quant = text_rq_loss + image_rq_loss

        # 对齐损失（双向 InfoNCE）- 可通过 align_weight=0 关闭
        if self.align_weight > 0:
            align_loss = self.infonce(z_q_text, z_q_image) + self.infonce(z_q_image, z_q_text)
        else:
            # 返回零 tensor 以保持类型一致性
            align_loss = torch.tensor(0.0, device=text_out.device, dtype=text_out.dtype)

        # LoRA 正则化损失
        lora_reg_loss = 0.0
        lora_ortho_loss = 0.0
        
        if self.use_lora:
            # L2 正则化：约束 A 和 B 的范数
            # 注意：A 是共享的（通过 share_lora_A_with 实现）
            # text_rq.shared_lora_A_list[i] 和 image_rq.shared_lora_A_list[i] 指向同一个 Parameter
            # 因此只需要计算一次，避免重复计算
            if hasattr(self.text_rq, 'shared_lora_A_list') and hasattr(self.text_rq, 'lora_layers'):
                for i, A in enumerate(self.text_rq.shared_lora_A_list):
                    if A is not None and i in self.text_rq.lora_layers:
                        lora_reg_loss = lora_reg_loss + (A ** 2).sum()
            
            # B 的正则化
            # 在 Recursive LoRA 模式下，B 是动态计算的，需要正则化 B_init 和 evolution_network
            if self.use_recursive_lora:
                # 正则化 B_init
                if hasattr(self.text_rq, 'B_init'):
                    lora_reg_loss = lora_reg_loss + (self.text_rq.B_init ** 2).sum()
                # 正则化 Evolution Network 的参数
                if hasattr(self.text_rq, 'evolution_network'):
                    for param in self.text_rq.evolution_network.parameters():
                        lora_reg_loss = lora_reg_loss + (param ** 2).sum()
            else:
                # 标准 LoRA：B 是各模态独立的，需要分别计算 text 和 image 的 B
                # 只为使用LoRA的层计算
                for i, vq_layer in enumerate(self.text_rq.vq_layers):
                    if i in self.text_rq.lora_layers:
                        lora_B = vq_layer.get_lora_B()
                        if lora_B is not None:
                            lora_reg_loss = lora_reg_loss + (lora_B ** 2).sum()
                
                for i, vq_layer in enumerate(self.image_rq.vq_layers):
                    if i in self.image_rq.lora_layers:
                        lora_B = vq_layer.get_lora_B()
                        if lora_B is not None:
                            lora_reg_loss = lora_reg_loss + (lora_B ** 2).sum()
            
            # 相对约束：约束LoRA bias相对于Base码本的大小
            # 这样可以防止LoRA过度改变码本，从而保护Base的对齐关系
            # 与lora_reg_loss不冲突：reg_loss约束绝对大小，这里约束相对大小
            if hasattr(self.text_rq, 'shared_lora_A_list') and hasattr(self.text_rq, 'lora_layers'):
                for i, (text_vq, image_vq) in enumerate(zip(self.text_rq.vq_layers, self.image_rq.vq_layers)):
                    # 只为使用LoRA的层计算
                    if i in self.text_rq.lora_layers and text_vq.use_lora and hasattr(text_vq, 'lora_A') and text_vq.lora_A is not None:
                        # 获取当前的 B 矩阵（支持 Recursive LoRA）
                        text_B = text_vq.get_lora_B()
                        image_B = image_vq.get_lora_B()
                        if text_B is not None and image_B is not None:
                            # 计算LoRA bias
                            text_lora_bias = torch.matmul(text_vq.lora_A, text_B)  # [n_e, e_dim]
                            image_lora_bias = torch.matmul(image_vq.lora_A, image_B)  # [n_e, e_dim]
                        
                            # Base码本的范数（作为参考）
                            text_base_norm = text_vq.embedding.weight.norm(p=2, dim=1, keepdim=True)  # [n_e, 1]
                            image_base_norm = image_vq.embedding.weight.norm(p=2, dim=1, keepdim=True)  # [n_e, 1]
                            
                            # LoRA bias的范数
                            text_bias_norm = text_lora_bias.norm(p=2, dim=1, keepdim=True)  # [n_e, 1]
                            image_bias_norm = image_lora_bias.norm(p=2, dim=1, keepdim=True)  # [n_e, 1]
                            
                            # 相对比例（bias占base的比例）
                            # 如果这个比例太大，说明LoRA可能过度改变了码本，可能破坏对齐
                            text_ratio = text_bias_norm / (text_base_norm + 1e-8)  # [n_e, 1]
                            image_ratio = image_bias_norm / (image_base_norm + 1e-8)  # [n_e, 1]
                            
                            # 允许一定比例的变化（比如5-10%），但超过阈值的部分要惩罚
                            # 这是一个软约束，不会完全禁止LoRA的变化
                            threshold = 0.1  # 10%的阈值
                            text_excess = F.relu(text_ratio - threshold).mean()
                            image_excess = F.relu(image_ratio - threshold).mean()
                            
                            # 这个约束与lora_reg_loss不冲突：
                            # - lora_reg_loss: 约束A和B的绝对大小
                            # - 这个约束: 约束最终bias相对于Base的相对大小（更直接保护码本）
                            lora_reg_loss = lora_reg_loss + (text_excess + image_excess)
            
            # 正交约束：让 A 的列向量更独立（防止 rank 塌缩）
            # 注意：A 是共享的，text_rq 和 image_rq 使用同一个 A
            # 因此只需要计算一次，避免重复计算
            # 只为使用LoRA的层计算
            if hasattr(self.text_rq, 'shared_lora_A_list') and hasattr(self.text_rq, 'lora_layers'):
                for i, A in enumerate(self.text_rq.shared_lora_A_list):
                    if A is not None and i in self.text_rq.lora_layers:
                        # A: [n_e, rank]
                        # A^T @ A: [rank, rank]
                        ATA = torch.matmul(A.t(), A)  # [rank, rank]
                        I = torch.eye(A.size(1), device=ATA.device, dtype=ATA.dtype)
                        lora_ortho_loss = lora_ortho_loss + ((ATA - I) ** 2).sum()
            
            # B 的正交约束：让 B 的行向量更独立（不同 rank 方向应该独立）
            # B: [rank, e_dim]，约束 B @ B^T = I，即行向量正交
            # 在 Recursive LoRA 模式下，只约束 B_init
            if self.use_recursive_lora:
                if hasattr(self.text_rq, 'B_init'):
                    B_init = self.text_rq.B_init  # [rank, e_dim]
                    BBT = torch.matmul(B_init, B_init.t())  # [rank, rank]
                    I_B = torch.eye(B_init.size(0), device=BBT.device, dtype=BBT.dtype)
                    lora_ortho_loss = lora_ortho_loss + ((BBT - I_B) ** 2).sum()
            else:
                # 标准 LoRA：只为使用LoRA的层计算
                for i, vq_layer in enumerate(self.text_rq.vq_layers):
                    if i in self.text_rq.lora_layers:
                        B = vq_layer.get_lora_B()
                        if B is not None:
                            BBT = torch.matmul(B, B.t())  # [rank, rank]
                            I_B = torch.eye(B.size(0), device=BBT.device, dtype=BBT.dtype)
                            lora_ortho_loss = lora_ortho_loss + ((BBT - I_B) ** 2).sum()
                
                for i, vq_layer in enumerate(self.image_rq.vq_layers):
                    if i in self.image_rq.lora_layers:
                        B = vq_layer.get_lora_B()
                        if B is not None:
                            BBT = torch.matmul(B, B.t())  # [rank, rank]
                            I_B = torch.eye(B.size(0), device=BBT.device, dtype=BBT.dtype)
                            lora_ortho_loss = lora_ortho_loss + ((BBT - I_B) ** 2).sum()

        # 总损失
        loss_total = (loss_recon + 
                     self.quant_loss_weight * loss_quant + 
                     self.align_weight * align_loss +
                     self.lora_reg_weight * lora_reg_loss +
                     self.lora_ortho_weight * lora_ortho_loss)

        return loss_total, loss_recon, loss_quant, align_loss
