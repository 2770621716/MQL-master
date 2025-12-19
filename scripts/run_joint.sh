#!/bin/bash
# MM-RQVAE with Shared+Specific LoRA A + Independent LoRA B
# 核心创新:
#   1. LoRA A分解: A_shared + ΔA_modality_specific
#      - A_shared: 跨模态共同语义 (text和image共享)
#      - ΔA_text: 文本特有语义残差
#      - ΔA_image: 图像特有语义残差
#   2. LoRA B: 每层独立，模态独立
#   3. 硬量化 + InfoNCE跨模态对齐
#
# 架构优势:
#   ✓ 既有共享(A_shared)又有特异性(ΔA)
#   ✓ 参数更少: 4个共享A + 8个ΔA vs 8个独立A
#   ✓ 更强的跨模态语义对齐
#
# lr 1e-3 3e-4 1e-4 3e-5 1e-5
# l2 3e-4 1e-4 3e-5 1e-5 3e-6

#
# MM-RQVAE Joint Training (Text + Image with Shared LoRA A)
#
# 使用方法：
#   bash run_joint.sh                          # 使用默认参数
#   bash run_joint.sh 0.1 0.001               # 指定 lora_alpha=0.1, align_weight=0.001
#   bash run_joint.sh 0.5 0.01 1e-3 2e-4      # 全参数: alpha, align, lr, l2
#
# 参数调优建议：
#   lora_alpha:   0.1, 0.5, 1.0, 2.0
#   align_weight: 0.001, 0.01, 0.1
#   lr:           1e-3, 3e-4, 1e-4
#   l2:           3e-4, 1e-4, 3e-5

# 接受命令行参数（带默认值）
LORA_ALPHA=${1:-1.00}        # 默认 0.5
ALIGN_WEIGHT=${2:-0.000}    # 默认 0.001
LR=${3:-1e-3}               # 默认 1e-3
L2=${4:-2e-4}               # 默认 2e-4
LORA_REG_WEIGHT=${5:-1e-4}   # 默认 1e-4
LORA_ORTHO_WEIGHT=${6:-1e-5} # 默认 1e-5
Code_num=256
LORA_RANK=8
NUM_LAYERS=4
Datasets='Instruments'

OUTPUT_DIR=./log/$Datasets/mm_rqvae_${Code_num}_rank${LORA_RANK}_alpha${LORA_ALPHA}_align${ALIGN_WEIGHT}
mkdir -p $OUTPUT_DIR

python -u ../main_mm.py \
  --use_lora True \
  --lora_rank $LORA_RANK \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /hy-tmp/hy-tmp/weiyonglin/MQL4GRec-master/data/ \
  --text_embedding_file .emb-llama-td.npy \
  --image_embedding_file .emb-ViT-L-14.npy \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --lr $LR \
  --weight_decay $L2 \
  --eval_step 10 \
  --batch_size 2048 \
  --epochs 1000 \
  --lora_alpha $LORA_ALPHA \
  --lora_reg_weight $LORA_REG_WEIGHT \
  --lora_ortho_weight $LORA_ORTHO_WEIGHT \
  --align_weight $ALIGN_WEIGHT

echo "Training Completed!"
