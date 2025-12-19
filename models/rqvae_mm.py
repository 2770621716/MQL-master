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
                 align_weight=0.01,
                 lora_reg_weight=1e-4,
                 lora_ortho_weight=1e-5
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
            lora_alpha=lora_alpha
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
            lora_alpha=lora_alpha
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
            # A 是共享的，只需要计算一次
            if hasattr(self.text_rq, 'shared_lora_A_list'):
                for A in self.text_rq.shared_lora_A_list:
                    lora_reg_loss = lora_reg_loss + (A ** 2).sum()
            
            # B 是各模态独立的
            for vq_layer in self.text_rq.vq_layers:
                if hasattr(vq_layer, 'lora_B') and vq_layer.lora_B is not None:
                    lora_reg_loss = lora_reg_loss + (vq_layer.lora_B ** 2).sum()
            
            for vq_layer in self.image_rq.vq_layers:
                if hasattr(vq_layer, 'lora_B') and vq_layer.lora_B is not None:
                    lora_reg_loss = lora_reg_loss + (vq_layer.lora_B ** 2).sum()
            
            # 正交约束：让 A 的列向量更独立（防止 rank 塌缩）
            if hasattr(self.text_rq, 'shared_lora_A_list'):
                for A in self.text_rq.shared_lora_A_list:
                    # A: [n_e, rank]
                    # A^T @ A: [rank, rank]
                    ATA = torch.matmul(A.t(), A)  # [rank, rank]
                    I = torch.eye(A.size(1), device=ATA.device, dtype=ATA.dtype)
                    lora_ortho_loss = lora_ortho_loss + ((ATA - I) ** 2).sum()

        # 总损失
        loss_total = (loss_recon + 
                     self.quant_loss_weight * loss_quant + 
                     self.align_weight * align_loss +
                     self.lora_reg_weight * lora_reg_loss +
                     self.lora_ortho_weight * lora_ortho_loss)

        return loss_total, loss_recon, loss_quant, align_loss
