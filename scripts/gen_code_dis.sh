Dataset=Instruments

OUTPUT_DIR=../../data/$Dataset

# 使用 MMRQVAE 模型（共享 LoRA A）
CKPT_PATH=log/$Dataset/mm_rqvae_256_rank8/best_text_model.pth

# 生成文本索引
python -u ../generate_indices_distance.py \
  --dataset $Dataset \
  --device cuda:0 \
  --ckpt_path $CKPT_PATH \
  --output_dir $OUTPUT_DIR \
  --output_file ${Dataset}.index_lemb.json \
  --model_type mmrqvae

# 生成图像索引
python -u ../generate_indices_distance.py \
  --dataset $Dataset \
  --device cuda:0 \
  --ckpt_path $CKPT_PATH \
  --output_dir $OUTPUT_DIR \
  --output_file ${Dataset}.index_vitemb.json \
  --content image \
  --model_type mmrqvae

echo "Done!"

