import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.getcwd()))

from models.rqvae_mm import MMRQVAE
from datasets import DualEmbDataset

ckpt_dir = "scripts/log/Instruments/mm_rqvae_256_rank8_alpha1.00_align0.000"
ckpt_path_text = os.path.join(ckpt_dir, "best_text_model.pth")

print("="*80)
print("加载 checkpoint (best_text_model.pth)...")
ckpt = torch.load(ckpt_path_text, map_location="cpu")
args = ckpt["args"]

# 数据路径
data_root = args.data_root
dataset_name = args.datasets
text_path = os.path.join(data_root, dataset_name, f"{dataset_name}{args.text_embedding_file}")
img_path = os.path.join(data_root, dataset_name, f"{dataset_name}{args.image_embedding_file}")

text_emb = np.load(text_path, mmap_mode='r')
img_emb = np.load(img_path, mmap_mode='r')

print(f"Text data: {text_emb.shape}, Image data: {img_emb.shape}")

# 构建模型
model = MMRQVAE(
    text_dim=text_emb.shape[1],
    image_dim=img_emb.shape[1],
    num_emb_list=args.num_emb_list,
    e_dim=args.e_dim,
    layers=args.layers,
    dropout_prob=args.dropout_prob,
    bn=args.bn,
    loss_type=args.loss_type,
    quant_loss_weight=args.quant_loss_weight,
    kmeans_init=args.kmeans_init,
    kmeans_iters=args.kmeans_iters,
    sk_epsilons=args.sk_epsilons,
    sk_iters=args.sk_iters,
    use_lora=args.use_lora,
    lora_rank=args.lora_rank,
    lora_alpha=args.lora_alpha,
    align_weight=args.align_weight,
    lora_reg_weight=getattr(args, 'lora_reg_weight', 1e-4),
    lora_ortho_weight=getattr(args, 'lora_ortho_weight', 1e-5),
)

model.load_state_dict(ckpt["state_dict"], strict=True)

device = torch.device(args.device if hasattr(args, "device") else "cuda:0")
model.to(device)
model.eval()

print("模型已加载到设备:", device)

# DataLoader
paired_data = DualEmbDataset(text_path, img_path)
loader = DataLoader(paired_data, batch_size=1024, shuffle=False, num_workers=4)

num_layers = len(args.num_emb_list)
num_codes = args.num_emb_list[0]

text_counts = [np.zeros(num_codes, dtype=np.int64) for _ in range(num_layers)]
image_counts = [np.zeros(num_codes, dtype=np.int64) for _ in range(num_layers)]

# 用于统计 top-2 距离 margin（d2 - d1）
text_margin_sum = np.zeros(num_layers, dtype=np.float64)
text_margin_sq_sum = np.zeros(num_layers, dtype=np.float64)
text_margin_min = np.full(num_layers, np.inf, dtype=np.float64)
text_margin_max = np.full(num_layers, -np.inf, dtype=np.float64)

image_margin_sum = np.zeros(num_layers, dtype=np.float64)
image_margin_sq_sum = np.zeros(num_layers, dtype=np.float64)
image_margin_min = np.full(num_layers, np.inf, dtype=np.float64)
image_margin_max = np.full(num_layers, -np.inf, dtype=np.float64)

print(f"\n开始统计码本使用次数和 top-2 距离 margin（{len(paired_data)} 个样本）...")
print(f"每层码本数量: {num_codes}, 总层数: {num_layers}\n")

total_samples = 0
with torch.no_grad():
    for batch_text, batch_image, _ in loader:
        batch_text = batch_text.to(device)
        batch_image = batch_image.to(device)

        # 文本模态：通过 RQ 获取 indices 和 distances
        text_e = model.text_encoder(batch_text)
        _, _, text_indices, text_distances = model.text_rq(text_e, use_sk=False)  # text_distances: [B, L, K]

        # 图像模态
        image_e = model.image_encoder(batch_image)
        _, _, image_indices, image_distances = model.image_rq(image_e, use_sk=False)  # image_distances: [B, L, K]

        text_indices_np = text_indices.cpu().numpy()   # [B, L]
        image_indices_np = image_indices.cpu().numpy() # [B, L]

        batch_size = text_indices_np.shape[0]
        total_samples += batch_size

        # 统计码本使用次数
        for layer in range(num_layers):
            t_idx = text_indices_np[:, layer].ravel()
            i_idx = image_indices_np[:, layer].ravel()
            text_counts[layer] += np.bincount(t_idx, minlength=num_codes)
            image_counts[layer] += np.bincount(i_idx, minlength=num_codes)

        # 统计 top-2 距离 margin（d2 - d1）
        text_d = text_distances.cpu().numpy()   # [B, L, K]
        image_d = image_distances.cpu().numpy() # [B, L, K]

        for layer in range(num_layers):
            # 文本
            d_layer_text = text_d[:, layer, :]  # [B, K]
            top2_text = np.partition(d_layer_text, 1, axis=1)[:, :2]  # [B, 2]，未排序
            top2_text.sort(axis=1)
            margins_text = top2_text[:, 1] - top2_text[:, 0]  # [B]

            text_margin_sum[layer] += margins_text.sum()
            text_margin_sq_sum[layer] += (margins_text ** 2).sum()
            text_margin_min[layer] = min(text_margin_min[layer], margins_text.min())
            text_margin_max[layer] = max(text_margin_max[layer], margins_text.max())

            # 图像
            d_layer_image = image_d[:, layer, :]  # [B, K]
            top2_image = np.partition(d_layer_image, 1, axis=1)[:, :2]
            top2_image.sort(axis=1)
            margins_image = top2_image[:, 1] - top2_image[:, 0]

            image_margin_sum[layer] += margins_image.sum()
            image_margin_sq_sum[layer] += (margins_image ** 2).sum()
            image_margin_min[layer] = min(image_margin_min[layer], margins_image.min())
            image_margin_max[layer] = max(image_margin_max[layer], margins_image.max())

print(f"统计完成，共处理 {total_samples} 个样本\n")

print("="*80)
print("文本模态码本使用情况")
print("="*80)
for layer in range(num_layers):
    counts = text_counts[layer]
    used = (counts > 0).sum()
    max_use = counts.max()
    min_use = counts[counts > 0].min() if used > 0 else 0
    p = counts / counts.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(num_codes)

    mean_margin = text_margin_sum[layer] / total_samples
    var_margin = text_margin_sq_sum[layer] / total_samples - mean_margin ** 2
    std_margin = np.sqrt(max(var_margin, 0.0))
    
    print(f"\nLayer {layer}:")
    print(f"  使用率: {used}/{num_codes} ({used/num_codes*100:.1f}%)")
    print(f"  使用次数: max={max_use}, min={min_use}, mean={counts.sum()/num_codes:.1f}")
    print(f"  熵值: {entropy:.4f} / {max_entropy:.4f} (归一化: {entropy/max_entropy:.4f})")
    print(f"  top-2 距离 margin: mean={mean_margin:.4f}, std={std_margin:.4f}, "
          f"min={text_margin_min[layer]:.4f}, max={text_margin_max[layer]:.4f}")

print("\n" + "="*80)
print("图像模态码本使用情况")
print("="*80)
for layer in range(num_layers):
    counts = image_counts[layer]
    used = (counts > 0).sum()
    max_use = counts.max()
    min_use = counts[counts > 0].min() if used > 0 else 0
    p = counts / counts.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(num_codes)

    mean_margin = image_margin_sum[layer] / total_samples
    var_margin = image_margin_sq_sum[layer] / total_samples - mean_margin ** 2
    std_margin = np.sqrt(max(var_margin, 0.0))
    
    print(f"\nLayer {layer}:")
    print(f"  使用率: {used}/{num_codes} ({used/num_codes*100:.1f}%)")
    print(f"  使用次数: max={max_use}, min={min_use}, mean={counts.sum()/num_codes:.1f}")
    print(f"  熵值: {entropy:.4f} / {max_entropy:.4f} (归一化: {entropy/max_entropy:.4f})")
    print(f"  top-2 距离 margin: mean={mean_margin:.4f}, std={std_margin:.4f}, "
          f"min={image_margin_min[layer]:.4f}, max={image_margin_max[layer]:.4f}")

print("\n" + "="*80)

