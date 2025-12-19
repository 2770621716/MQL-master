import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from info_nce import InfoNCE

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0,
                 use_lora=False,
                 lora_rank=8
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.use_lora = use_lora
        self.lora_rank = lora_rank

        # InfoNCE for cross-modal alignment
        self.infonce = InfoNCE()

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          use_linear=use_linear,
                                          use_lora=use_lora,
                                          lora_rank=lora_rank)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices, distances = self.rq(x,use_sk=use_sk)
        # print(indices.shape)
        out = self.decoder(x_q)

        return out, rq_loss, indices, x_q  # 返回量化后的表示用于对齐

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices, distances = self.rq(x_e, use_sk=use_sk)
        return indices, distances
    
    def share_lora_A_with(self, other_model):
        """Share LoRA A matrices with another RQVAE model"""
        if self.use_lora and other_model.use_lora:
            self.rq.share_lora_A_with(other_model.rq)
        else:
            print("Warning: LoRA is not enabled in one or both models")

    def compute_loss(self, out, quant_loss, xs=None, x_q=None, other_x_q=None, align_weight=0.01):
        """
        计算损失
        
        Args:
            out: 解码后的输出
            quant_loss: 量化损失
            xs: 原始输入
            x_q: 当前模态的量化表示 [B, e_dim]
            other_x_q: 另一个模态的量化表示 [B, e_dim]
            align_weight: 对齐损失权重
        """
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        
        # 计算 InfoNCE 对齐损失（双向对齐）
        if x_q is not None and other_x_q is not None:
            align_loss = self.infonce(x_q, other_x_q) + self.infonce(other_x_q, x_q)
            loss_total = loss_total + align_weight * align_loss

        return loss_total, loss_recon