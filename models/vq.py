import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.01, sk_iters=100, use_linear=0,
                 use_lora=False, lora_rank=8, lora_alpha=1.0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_linear = use_linear
        
        # LoRA parameters
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha  

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()
        
        if self.use_lora:
            self.lora_A = None  
            self.lora_B = nn.Parameter(torch.zeros(lora_rank, e_dim))  

        if use_linear == 1:
            self.codebook_projection = torch.nn.Linear(self.e_dim, self.e_dim)
            torch.nn.init.normal_(self.codebook_projection.weight, std=self.e_dim ** -0.5)

    def set_shared_lora_A(self, lora_A):
        """Set the shared LoRA A matrix"""
        self.lora_A = lora_A
    
    def get_codebook(self):
        """Get base codebook (without LoRA)"""
        return self.embedding.weight
    
    def get_codebook_with_lora(self):
        """Get effective codebook with LoRA bias (alpha=1.0, regularization controls magnitude)"""
        base_codebook = self.embedding.weight  # [n_e, e_dim]
        
        if self.use_lora and self.lora_A is not None:
            # A: [n_e, rank], B: [rank, e_dim]
            # A @ B: [n_e, e_dim]
            lora_bias = torch.matmul(self.lora_A, self.lora_B)  # [n_e, e_dim]
            # alpha = 1.0 (use regularization to control magnitude instead)
            codebook_with_bias = base_codebook + lora_bias
            return codebook_with_bias
        else:
            return base_codebook

    def get_codebook_entry(self, indices, shape=None):
        if self.use_lora and self.lora_A is not None:
            codebook = self.get_codebook_with_lora()
            z_q = F.embedding(indices, codebook)
        else:
            z_q = self.embedding(indices)
            
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        """
        前向传播（硬量化）
        
        Args:
            x: 输入向量 [B, e_dim]
            use_sk: 是否使用 Sinkhorn
        """
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)
        
        # Get effective codebook (with LoRA if enabled)
        if self.use_lora and self.lora_A is not None:
            embeddings_weight = self.get_codebook_with_lora()
        else:
            embeddings_weight = self.embedding.weight
            
        if self.use_linear == 1:
            embeddings_weight = self.codebook_projection(embeddings_weight)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(embeddings_weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, embeddings_weight.t())
        
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_centered = self.center_distance_for_constraint(d)
            d_centered = d_centered.double()
            Q = sinkhorn_algorithm(d_centered, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)


        if self.use_linear == 1:
            x_q = F.embedding(indices, embeddings_weight).view(x.shape)
        else:
            x_q = self.get_codebook_entry(indices).view(x.shape)

        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices, d


