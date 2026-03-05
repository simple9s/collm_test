# CoLLM 完整使用指南

## 📦 安装步骤

### 1. 创建虚拟环境（推荐）

```bash
# 使用conda
conda create -n collm python=3.9
conda activate collm

# 或使用venv
python -m venv collm_env
source collm_env/bin/activate  # Linux/Mac
# collm_env\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖包列表**:
- torch >= 2.0.0
- transformers >= 4.36.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- tqdm >= 4.66.0
- peft >= 0.7.0 (用于LoRA)
- accelerate >= 0.25.0

### 3. 验证安装

```bash
python test_metrics.py
```

应该看到评估指标的测试输出。

---

## 🗂️ 数据准备

### 1. 下载Amazon Books 2018数据集

从 [Amazon Review Data](http://jmcauley.ucsd.edu/data/amazon/) 下载Books类别的评分数据。

**文件格式**: CSV或JSON，包含以下字段：
- `reviewerID` (用户ID)
- `asin` (商品ID)  
- `overall` (评分)
- `unixReviewTime` (时间戳)

### 2. 转换为标准格式

将数据转换为CSV格式，包含以下列：
```
user_id, item_id, rating, timestamp
```

示例代码：
```python
import pandas as pd
import json

# 读取Amazon JSON格式
data = []
with open('Books.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# 转换格式
df_processed = pd.DataFrame({
    'user_id': df['reviewerID'],
    'item_id': df['asin'],
    'rating': df['overall'],
    'timestamp': df['unixReviewTime']
})

# 保存
df_processed.to_csv('amazon_books_2018.csv', index=False)
```

### 3. 数据预处理

```bash
python data/preprocess_amazon.py \
    --input ./data/raw/amazon_books_2018.csv \
    --output ./data/amazon2018 \
    --min_inter 5 \
    --neg_samples 19 99
```

**输出结构**:
```
data/amazon2018/
├── meta.pkl              # 元数据（用户数、物品数）
├── neg_19/
│   ├── train.pkl
│   ├── valid.pkl
│   └── test.pkl
└── neg_99/
    ├── train.pkl
    ├── valid.pkl
    └── test.pkl
```

---

## 🚀 训练模型

### 方法1: 一键运行（推荐新手）

```bash
# 修改配置
vim scripts/run_all.sh

# 运行
bash scripts/run_all.sh
```

### 方法2: 分步运行（推荐调试）

#### Stage 0: 预训练CF模型

```bash
python train.py \
    --data_dir ./data/amazon2018 \
    --n_neg 19 \
    --stage 0 \
    --cf_dim 64 \
    --cf_epochs 50 \
    --batch_size 1024 \
    --cf_lr 1e-3 \
    --weight_decay 1e-4 \
    --save_dir ./checkpoints
```

**输出**: `checkpoints/cf_model.pt`

#### Stage 1: 训练投影层

```bash
python train.py \
    --data_dir ./data/amazon2018 \
    --n_neg 19 \
    --llm_name meta-llama/Llama-3.2-3B-Instruct \
    --stage 1 \
    --cf_ckpt ./checkpoints/cf_model.pt \
    --cf_dim 64 \
    --n_tokens 1 \
    --num_epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-3 \
    --patience 10 \
    --eval_type ranking \
    --save_dir ./checkpoints
```

**输出**: `checkpoints/stage1_neg19/best_model.pt`

#### Stage 2: LoRA微调LLM

```bash
python train.py \
    --data_dir ./data/amazon2018 \
    --n_neg 19 \
    --llm_name meta-llama/Llama-3.2-3B-Instruct \
    --stage 2 \
    --cf_ckpt ./checkpoints/cf_model.pt \
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
    --eval_type ranking \
    --save_dir ./checkpoints
```

**输出**: `checkpoints/stage2_neg19/best_model.pt`

---

## 📊 实验配置建议

### 基础实验（必须）

| 实验 | LLM | 负样本 | 预计时间 |
|------|-----|--------|----------|
| 1 | Llama-3B | 19 | ~4小时 |
| 2 | Llama-3B | 99 | ~6小时 |
| 3 | OPT-6.7B | 19 | ~6小时 |
| 4 | OPT-6.7B | 99 | ~8小时 |

### 消融实验（可选）

**测试CF嵌入维度**:
```bash
for dim in 32 64 128 256; do
    python train.py --cf_dim $dim ...
done
```

**测试投影token数**:
```bash
for n_tok in 1 2 3; do
    python train.py --n_tokens $n_tok ...
done
```

**测试LoRA秩**:
```bash
for r in 4 8 16 32; do
    python train.py --lora_r $r --lora_alpha $((r*2)) ...
done
```

---

## 🔍 查看结果

### 训练日志

训练过程会实时输出到终端，也可以重定向到文件：

```bash
bash scripts/run_all.sh 2>&1 | tee training.log
```

### 结果文件

每个阶段会生成：
```
checkpoints/stage{1,2}_neg{19,99}/
├── best_model.pt     # 最佳模型权重
└── results.pkl       # 评估结果
```

### 读取结果

```python
import pickle

# 读取结果
with open('checkpoints/stage2_neg19/results.pkl', 'rb') as f:
    results = pickle.load(f)

print("Best Val Metric:", results['best_val_metric'])
print("Test Metrics:", results['test_metrics'])
```

---

## ⚙️ 常见问题

### Q1: CUDA Out of Memory

**解决方案**:
1. 减小batch_size
2. 使用更小的LLM（如Llama-3B而非OPT-6.7B）
3. 减小cf_dim
4. 启用梯度检查点（需修改代码）

### Q2: 训练很慢

**原因**: LLM推理开销大

**解决方案**:
1. 使用更小的batch_size（减少等待时间）
2. 使用更快的GPU
3. 开启mixed precision训练（已默认启用）

### Q3: 指标不收敛

**可能原因**:
1. 学习率太大/太小
2. CF模型质量差
3. 数据质量问题

**解决方案**:
1. 调整学习率：Stage 1 尝试1e-2, 1e-3, 1e-4
2. 增加CF预训练轮数
3. 检查数据预处理

### Q4: 不同评估模式的区别

**Ranking模式** (推荐):
- 评估Top-K推荐质量
- 指标: Hit@K, NDCG@K, MRR@K
- 更符合推荐系统实际应用

**Classification模式**:
- 评估二分类性能
- 指标: AUC, ACC
- 用于对比或调试

---

## 📈 性能优化建议

### 1. 数据加载优化

```python
# 增加num_workers
train_loader = DataLoader(..., num_workers=8)
```

### 2. 使用更高效的LLM

- Llama-3B比OPT-6.7B快约2倍
- 可尝试更小的模型（如1B）做快速实验

### 3. 分布式训练

```bash
# 使用多GPU
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py ...
```

### 4. 缓存预处理数据

第一次运行后，数据已缓存为.pkl文件，后续运行会更快。

---

## 📝 论文写作建议

### 1. 记录所有超参数

创建一个表格：
```
| 参数 | 值 |
|------|-----|
| CF嵌入维度 | 64 |
| 投影token数 | 1 |
| 学习率(Stage1) | 1e-3 |
| 学习率(Stage2) | 1e-4 |
| LoRA秩 | 8 |
| Batch size | 32/16 |
| ... | ... |
```

### 2. 报告完整指标

报告所有K值的结果：
```
Hit@1/5/10/20
NDCG@1/5/10/20  
MRR@1/5/10/20
```

### 3. 进行显著性检验

```python
from scipy import stats

# 收集多次运行结果
model_a_scores = [0.52, 0.53, 0.51, ...]
model_b_scores = [0.48, 0.49, 0.47, ...]

# t检验
t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
print(f"p-value: {p_value:.4f}")
```

### 4. 绘制学习曲线

保存每个epoch的结果并绘图：

```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, ...]
ndcg_scores = [0.12, 0.15, 0.18, ...]

plt.plot(epochs, ndcg_scores)
plt.xlabel('Epoch')
plt.ylabel('NDCG@10')
plt.title('Training Curve')
plt.savefig('learning_curve.pdf')
```

---

## 🎓 引用

如果使用本代码，请引用：

```bibtex
@article{collm2024,
  title={Collaborative Large Language Models for Recommender Systems},
  author={Zhang, Yang and Others},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

---

## 💬 获取帮助

1. 查看README.md
2. 查看CHECKLIST.md
3. 查看INNOVATION.md
4. 运行test_metrics.py验证评估指标
5. 运行test_pipeline.py验证完整流程

如有问题，请提Issue或联系作者。
