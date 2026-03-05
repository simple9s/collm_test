# CoLLM Baseline for Amazon 2018 Dataset

## 📋 项目概述

这是CoLLM (Collaborative Large Language Model)的简化基线实现，用于Amazon 2018数据集的推荐任务。

**核心思想**: 将协同过滤学到的用户/物品ID嵌入映射到大语言模型的语义空间，利用LLM的语义理解能力进行推荐。

## 🏗️ 项目结构

```
collm_baseline/
├── data/
│   └── preprocess_amazon.py    # 数据预处理
├── models/
│   ├── mf.py                    # 矩阵分解模型
│   └── collm.py                 # CoLLM主模型
├── scripts/
│   └── run_all.sh               # 完整运行脚本
├── train.py                     # 训练脚本
├── requirements.txt             # 依赖包
└── README.md                    # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 数据准备

下载Amazon Books 2018数据集，格式为CSV：
```
user_id, item_id, rating, timestamp
```

### 3. 数据预处理

```bash
python data/preprocess_amazon.py \
    --input ./data/raw/amazon_books_2018.csv \
    --output ./data/amazon2018 \
    --min_inter 5 \
    --neg_samples 19 99
```

**参数说明**:
- `--min_inter 5`: 过滤交互次数<5的用户
- `--neg_samples 19 99`: 生成19和99负样本两个版本

### 4. 训练模型

#### 方法1: 一键运行（推荐）

```bash
bash scripts/run_all.sh
```

#### 方法2: 分阶段运行

**Stage 0: 预训练协同过滤模型**
```bash
python train.py \
    --data_dir ./data/amazon2018 \
    --n_neg 19 \
    --stage 0 \
    --cf_dim 64 \
    --cf_epochs 50 \
    --batch_size 1024
```

**Stage 1: 训练投影层（冻结CF和LLM）**
```bash
python train.py \
    --data_dir ./data/amazon2018 \
    --n_neg 19 \
    --llm_name meta-llama/Llama-3.2-3B-Instruct \
    --stage 1 \
    --cf_ckpt checkpoints/cf_model.pt \
    --batch_size 32 \
    --lr 1e-3
```

**Stage 2: 微调LLM（LoRA）**
```bash
python train.py \
    --data_dir ./data/amazon2018 \
    --n_neg 19 \
    --llm_name meta-llama/Llama-3.2-3B-Instruct \
    --stage 2 \
    --cf_ckpt checkpoints/cf_model.pt \
    --use_lora \
    --batch_size 16 \
    --lr 1e-4
```

## 📊 支持的配置

### 负样本数
- **19个负样本**: 标准配置
- **99个负样本**: 更难的配置

### LLM模型
- **Llama-3.2-3B-Instruct**: `meta-llama/Llama-3.2-3B-Instruct`
- **OPT-6.7B**: `facebook/opt-6.7b`

修改`scripts/run_all.sh`中的`LLM_NAME`变量即可切换。

## 📊 评估指标

### 排序指标 (Ranking Metrics) - 推荐使用

更符合推荐系统标准评估方式，在Leave-One-Out场景下评估：

| 指标 | K值 | 说明 |
|------|-----|------|
| **Hit@K** | 1, 5, 10, 20 | Top-K中是否包含真实物品 |
| **NDCG@K** | 1, 5, 10, 20 | 归一化折扣累积增益（考虑排序位置） |
| **MRR@K** | 1, 5, 10, 20 | 平均倒数排名（第一个相关物品的排名） |

**使用方法**:
```bash
python train.py --eval_type ranking ...
```

**输出示例**:
```
Test Metrics
============================================================
K= 1 | Hit: 0.0523 | NDCG: 0.0523 | MRR: 0.0523
K= 5 | Hit: 0.2156 | NDCG: 0.1245 | MRR: 0.1089
K=10 | Hit: 0.3845 | NDCG: 0.1678 | MRR: 0.1245
K=20 | Hit: 0.6234 | NDCG: 0.2134 | MRR: 0.1389
```

### 分类指标 (Classification Metrics) - 可选

用于对比二分类性能：

- **AUC**: ROC曲线下面积
- **ACC**: 分类准确率

**使用方法**:
```bash
python train.py --eval_type classification ...
```

## 📈 评估指标

- **AUC**: ROC曲线下面积
- **ACC**: 分类准确率

## ⚙️ 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cf_dim` | 64 | CF嵌入维度 |
| `n_tokens` | 1 | 每个ID映射为几个token |
| `batch_size` | 32 (stage1), 16 (stage2) | 批大小 |
| `lr` | 1e-3 (stage1), 1e-4 (stage2) | 学习率 |
| `lora_r` | 8 | LoRA秩 |
| `lora_alpha` | 16 | LoRA alpha |
| `patience` | 10 | Early stopping轮数 |

## 📁 输出文件

```
checkpoints/
├── cf_model.pt                           # CF模型
├── stage1_neg19/
│   ├── best_model.pt                    # Stage1最佳模型
│   └── results.pkl                       # Stage1结果
└── stage2_neg19/
    ├── best_model.pt                    # Stage2最佳模型
    └── results.pkl                       # Stage2结果
```

## 🔧 故障排除

### CUDA内存不足
- 减小`batch_size`
- 使用`--llm_name facebook/opt-6.7b`（较小模型）
- 使用梯度累积

### 数据加载错误
- 确认数据预处理完成
- 检查数据路径是否正确

## 📝 引用

如果使用本代码，请引用原始CoLLM论文：

```bibtex
@article{collm2024,
  title={Collaborative Large Language Models for Recommender Systems},
  author={Zhang, Yang and ...},
  journal={...},
  year={2024}
}
```

## 📧 联系方式

如有问题，请提Issue或联系作者。
