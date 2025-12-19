import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100, use_linear=0,
                 use_lora=False, lora_rank=8, lora_alpha=1.0):
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
        
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters,
                                                        use_linear=use_linear,
                                                        use_lora=use_lora,
                                                        lora_rank=lora_rank,
                                                        lora_alpha=lora_alpha)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])
        

        if use_lora:
            # 共享LoRA A (4层，每层一个)
            self.shared_lora_A_list = nn.ParameterList()
            for n_e in n_e_list:
                shared_A = torch.randn(n_e, lora_rank) * 0.02
                self.shared_lora_A_list.append(nn.Parameter(shared_A))
            
            for i, vq_layer in enumerate(self.vq_layers):
                vq_layer.set_shared_lora_A(self.shared_lora_A_list[i])

    def share_lora_A_with(self, other_rq):
        if not self.use_lora or not other_rq.use_lora:
            print("Warning: LoRA is not enabled in one or both models")
            return
        
        assert self.num_quantizers == other_rq.num_quantizers
        assert self.lora_rank == other_rq.lora_rank
        
        for i in range(self.num_quantizers):
            shared_A = self.shared_lora_A_list[i]
            other_rq.shared_lora_A_list[i] = shared_A
            other_rq.vq_layers[i].set_shared_lora_A(shared_A)
        
        print(f"✓ Shared LoRA A (B remains independent)")

    
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