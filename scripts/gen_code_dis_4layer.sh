#!/bin/bash

# 使用 4层 LoRA 模型生成索引

Dataset=Instruments
OUTPUT_DIR=../../data/$Dataset

# 4层 LoRA 模型路径
CKPT_PATH=log/$Dataset/mm_rqvae_256_rank8_alpha1.00_align0.000/best_text_model.pth

# 生成文本索引
python -u ../generate_indices_distance.py \
  --dataset $Dataset \
  --device cuda:0 \
  --ckpt_path $CKPT_PATH \
  --output_dir $OUTPUT_DIR \
  --output_file ${Dataset}.index_lemb_4layer.json \
  --model_type mmrqvae

# 生成图像索引
python -u ../generate_indices_distance.py \
  --dataset $Dataset \
  --device cuda:0 \
  --ckpt_path $CKPT_PATH \
  --output_dir $OUTPUT_DIR \
  --output_file ${Dataset}.index_vitemb_4layer.json \
  --content image \
  --model_type mmrqvae

echo "Done! (4-layer LoRA)"
