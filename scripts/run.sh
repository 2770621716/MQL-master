

Model=llama
Code_num=256

# LoRA settings
USE_LORA=True
LORA_RANK=8

Datasets='Instruments'

OUTPUT_DIR=./log/$Datasets/${Model}_${Code_num}_lora${LORA_RANK}
mkdir -p $OUTPUT_DIR

python -u ../main_mul.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /hy-tmp/hy-tmp/weiyonglin/MQL4GRec-master/data/ \
  --embedding_file .emb-llama-td.npy \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 2 \
  --batch_size 2048 \
  --epochs 2000 \
  --use_lora $USE_LORA \
  --lora_rank $LORA_RANK > $OUTPUT_DIR/train.log

Model=ViT-L-14
Code_num=256

# LoRA settings
USE_LORA=True
LORA_RANK=8

Datasets='Instruments'

OUTPUT_DIR=./log/$Datasets/${Model}_${Code_num}_lora${LORA_RANK}
mkdir -p $OUTPUT_DIR

nohup python -u ../main_mul.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /hy-tmp/hy-tmp/weiyonglin/MQL4GRec-master/data/ \
  --embedding_file .emb-ViT-L-14.npy \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 2 \
  --batch_size 2048 \
  --epochs 2000 \
  --use_lora $USE_LORA \
  --lora_rank $LORA_RANK > $OUTPUT_DIR/train.log

