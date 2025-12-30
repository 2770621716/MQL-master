import torch
import torch.nn as nn

from .vq import VectorQuantizer
from .layers import MLPLayers


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100, use_linear=0,
                 use_lora=False, lora_rank=8, lora_alpha=1.0, lora_layers=None,
                 use_recursive_lora=False):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.use_recursive_lora = use_recursive_lora and use_lora  # 只有在启用 LoRA 时才能使用 Recursive LoRA
        
        # 指定哪些层使用LoRA，默认只在前两层 [0, 1]
        if lora_layers is None:
            lora_layers = [0, 1] if use_lora else []
        # 如果 lora_layers 是字符串，需要解析为列表
        elif isinstance(lora_layers, str):
            try:
                lora_layers = [int(x.strip()) for x in lora_layers.split(',')]
            except ValueError:
                print(f"Warning: Invalid lora_layers format '{lora_layers}', using default [0, 1]")
                lora_layers = [0, 1] if use_lora else []
        self.lora_layers = set(lora_layers)  # 转换为set便于快速查找
        if use_lora and len(self.lora_layers) == 0:
            print("Warning: use_lora=True but lora_layers is empty, LoRA will not be used")
        
        # 为每层创建VQ layer，但只有指定的层使用LoRA
        self.vq_layers = nn.ModuleList()
        for i, (n_e, sk_epsilon) in enumerate(zip(n_e_list, sk_epsilons)):
            layer_use_lora = use_lora and (i in self.lora_layers)
            vq_layer = VectorQuantizer(n_e, e_dim,
                                       kmeans_init=self.kmeans_init,
                                       kmeans_iters=self.kmeans_iters,
                                       sk_epsilon=sk_epsilon,
                                       sk_iters=sk_iters,
                                       use_linear=use_linear,
                                       use_lora=layer_use_lora,
                                       lora_rank=lora_rank,
                                       lora_alpha=lora_alpha)
            self.vq_layers.append(vq_layer)
        

        if use_lora and len(self.lora_layers) > 0:
            # 共享LoRA A (只为指定的层创建)
            self.shared_lora_A_list = nn.ParameterList()
            for i, n_e in enumerate(n_e_list):
                if i in self.lora_layers:
                    shared_A = torch.randn(n_e, lora_rank) * 0.02
                    self.shared_lora_A_list.append(nn.Parameter(shared_A))
                    self.vq_layers[i].set_shared_lora_A(self.shared_lora_A_list[-1])
                else:
                    # 不使用LoRA的层，占位为None
                    self.shared_lora_A_list.append(None)
            
            # ========== Recursive LoRA 初始化 ==========
            if self.use_recursive_lora:
                # B_init: 初始 B 矩阵 [rank, e_dim]
                self.B_init = nn.Parameter(torch.zeros(lora_rank, e_dim))
                
                # Evolution Network: 一个小的 MLP，用于演化 B
                # 输入: B [rank, e_dim] -> 展平为 [rank * e_dim]
                # 输出: delta_B [rank, e_dim]
                evolution_hidden_dim = max(lora_rank * e_dim // 2, 64)  # 隐藏层维度
                self.evolution_network = MLPLayers(
                    layers=[lora_rank * e_dim, evolution_hidden_dim, lora_rank * e_dim],
                    dropout=0.0,
                    activation="relu",
                    bn=False
                )
                
                # 初始化 Evolution Network 的输出层权重较小，确保演化是渐进式的
                if hasattr(self.evolution_network, 'mlp_layers'):
                    for module in self.evolution_network.mlp_layers:
                        if isinstance(module, nn.Linear) and module.out_features == lora_rank * e_dim:
                            # 将输出层权重初始化为很小的值，确保演化是渐进式的
                            nn.init.normal_(module.weight, mean=0.0, std=0.01)
                            if module.bias is not None:
                                nn.init.zeros_(module.bias)
                
                print(f"✓ Recursive LoRA enabled: B evolves across layers")
                print(f"  - B_init shape: [{lora_rank}, {e_dim}]")
                print(f"  - Evolution Network: {lora_rank * e_dim} -> {evolution_hidden_dim} -> {lora_rank * e_dim}")
            else:
                print(f"✓ LoRA enabled for layers: {sorted(self.lora_layers)} (total {len(self.lora_layers)}/{self.num_quantizers} layers)")

    def share_lora_A_with(self, other_rq):
        if not self.use_lora or not other_rq.use_lora:
            print("Warning: LoRA is not enabled in one or both models")
            return
        
        assert self.num_quantizers == other_rq.num_quantizers
        assert self.lora_rank == other_rq.lora_rank
        assert self.lora_layers == other_rq.lora_layers, "LoRA layers must match for sharing"
        
        # 只为使用LoRA的层共享A
        for i in self.lora_layers:
            shared_A = self.shared_lora_A_list[i]
            other_rq.shared_lora_A_list[i] = shared_A
            other_rq.vq_layers[i].set_shared_lora_A(shared_A)
        
        print(f"✓ Shared LoRA A for layers {sorted(self.lora_layers)} (B remains independent)")

    
    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []
        all_distances = []

        x_q = 0
        residual = x
        
        # ========== Recursive LoRA: 层级演化 B 矩阵 ==========
        if self.use_recursive_lora and len(self.lora_layers) > 0:
            # 初始化 B_0 = B_init
            current_B = self.B_init  # [rank, e_dim]
            
            for layer_idx, quantizer in enumerate(self.vq_layers):
                # 如果这一层使用 LoRA，则动态设置 B
                if layer_idx in self.lora_layers:
                    # 设置当前层的 B
                    quantizer.set_lora_B(current_B)
                
                # 前向传播
                x_res, loss, indices, distance = quantizer(residual, use_sk=use_sk)
                residual = residual - x_res
                x_q = x_q + x_res

                all_losses.append(loss)
                all_indices.append(indices)
                all_distances.append(distance)
                
                # 在每一层之后演化 B（即使当前层不使用 LoRA，也要继续演化，供后续层使用）
                # 计算下一层的 B: B_{i+1} = B_i + Evolve(B_i)
                # 展平 B 以便输入 Evolution Network
                B_flat = current_B.view(-1)  # [rank * e_dim]
                # 通过 Evolution Network 得到 delta_B
                delta_B_flat = self.evolution_network(B_flat)  # [rank * e_dim]
                delta_B = delta_B_flat.view(self.lora_rank, self.e_dim)  # [rank, e_dim]
                # 更新 B（为下一层准备）
                current_B = current_B + delta_B
        else:
            # 标准 LoRA：每层使用自己的 B
            for quantizer in self.vq_layers:
                x_res, loss, indices, distance = quantizer(residual, use_sk=use_sk)
                residual = residual - x_res
                x_q = x_q + x_res

                all_losses.append(loss)
                all_indices.append(indices)
                all_distances.append(distance)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_distances = torch.stack(all_distances, dim=1)

        return x_q, mean_losses, all_indices, all_distances