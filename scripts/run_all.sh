#!/bin/bash

# CoLLM Baseline 运行脚本
# 支持Amazon 2018数据集，19/99负样本，Llama-3B/OPT-6.7B

set -e

# ============ 配置 ============
DATA_DIR="./data/amazon2018"
RAW_DATA="./data/raw/amazon_books_2018.csv"  # 需要自行下载
OUTPUT_DIR="./checkpoints"

# 选择负样本数：19 or 99
N_NEG=19

# 选择LLM模型
LLM_NAME="meta-llama/Llama-3.2-3B-Instruct"
# LLM_NAME="facebook/opt-6.7b"

# 选择CF模型：mf or sasrec
CF_MODEL="sasrec"

# 选择评估类型：ranking (推荐) or classification
EVAL_TYPE="ranking"

# ============ Step 1: 数据预处理 ============
echo "=========================================="
echo "Step 1: Data Preprocessing"
echo "=========================================="

python data/preprocess_amazon.py \
    --input ${RAW_DATA} \
    --output ${DATA_DIR} \
    --min_inter 5 \
    --neg_samples 19 99

# ============ Step 2: Stage 0 - 预训练CF模型 ============
echo ""
echo "=========================================="
echo "Step 2: Stage 0 - Pre-train CF Model"
echo "=========================================="

python train.py \
    --data_dir ${DATA_DIR} \
    --n_neg ${N_NEG} \
    --stage 0 \
    --cf_model ${CF_MODEL} \
    --cf_dim 64 \
    --cf_epochs 50 \
    --batch_size 1024 \
    --cf_lr 1e-3 \
    --weight_decay 1e-4 \
    --save_dir ${OUTPUT_DIR}

# ============ Step 3: Stage 1 - 训练投影层 ============
echo ""
echo "=========================================="
echo "Step 3: Stage 1 - Train Projection Layer"
echo "=========================================="

python train.py \
    --data_dir ${DATA_DIR} \
    --n_neg ${N_NEG} \
    --llm_name ${LLM_NAME} \
    --stage 1 \
    --cf_model ${CF_MODEL} \
    --cf_dim 64 \
    --n_tokens 1 \
    --num_epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-3 \
    --patience 10 \
    --eval_type ${EVAL_TYPE} \
    --cf_ckpt ${OUTPUT_DIR}/cf_${CF_MODEL}.pt \
    --save_dir ${OUTPUT_DIR}

# ============ Step 4: Stage 2 - 微调LLM (LoRA) ============
echo ""
echo "=========================================="
echo "Step 4: Stage 2 - Fine-tune LLM with LoRA"
echo "=========================================="

python train.py \
    --data_dir ${DATA_DIR} \
    --n_neg ${N_NEG} \
    --llm_name ${LLM_NAME} \
    --stage 2 \
    --cf_model ${CF_MODEL} \
    --cf_dim 64 \
    --n_tokens 1 \
    --num_epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 1e-3 \
    --patience 10 \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --eval_type ${EVAL_TYPE} \
    --cf_ckpt ${OUTPUT_DIR}/cf_${CF_MODEL}.pt \
    --save_dir ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Results saved in: ${OUTPUT_DIR}"
