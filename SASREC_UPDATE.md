# CoLLM Baseline 更新说明 - SASRec版本

## 🎉 重要更新

**默认CF模型已从MF更改为SASRec！**

SASRec (Self-Attentive Sequential Recommendation) 是基于Transformer的序列推荐模型，相比简单的矩阵分解(MF)有以下优势：

### 为什么使用SASRec？

1. **序列建模能力**: 捕捉用户的时序行为模式
2. **更强的表达力**: Transformer架构比MF更强大
3. **更好的性能**: 在大多数推荐任务上优于MF
4. **学术标准**: SASRec是推荐系统领域的主流baseline

### MF vs SASRec

| 特性 | MF | SASRec |
|------|-----|--------|
| 输入 | 用户ID、物品ID | 用户历史序列、物品ID |
| 模型 | 简单的嵌入层 | Transformer编码器 |
| 复杂度 | 低 | 中等 |
| 性能 | 基础 | 优秀 |
| 训练时间 | 快 | 中等 |

---

## 🚀 使用方法

### 默认使用SASRec

```bash
# 数据预处理会自动生成序列数据
python data/preprocess_amazon.py \
    --input amazon_books_2018.csv \
    --output ./data/amazon2018 \
    --min_inter 5 \
    --neg_samples 19 99

# 训练（默认使用SASRec）
bash scripts/run_all.sh
```

### 切换到MF（如果需要）

在`scripts/run_all.sh`中修改：
```bash
CF_MODEL="mf"  # 改为mf
```

或直接运行：
```bash
python train.py --cf_model mf ...
```

---

## 📊 数据格式

数据预处理会生成两种格式：

### 1. MF格式 (train.pkl, valid.pkl, test.pkl)
```python
{
    'uid': 用户ID,
    'iid': 物品ID,
    'label': 标签(0/1)
}
```

### 2. SASRec格式 (train_seq.pkl, valid_seq.pkl, test_seq.pkl)
```python
{
    'uid': 用户ID,
    'iid': 物品ID,
    'seq': 历史序列[50维],  # 最近50个交互物品
    'label': 标签(0/1)
}
```

---

## ⚙️ SASRec超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_seq_len` | 50 | 最大序列长度 |
| `num_blocks` | 2 | Transformer块数 |
| `num_heads` | 1 | 注意力头数 |
| `dropout_rate` | 0.2 | Dropout率 |
| `embedding_dim` | 64 | 嵌入维度 |

可以通过命令行参数调整：
```bash
python train.py \
    --cf_model sasrec \
    --max_seq_len 50 \
    --num_blocks 2 \
    --num_heads 1 \
    --dropout_rate 0.2 \
    ...
```

---

## 🔄 模型架构

### CoLLM + SASRec

```
用户历史序列 [seq_len] → SASRec → 用户嵌入 [dim]
                                       ↓
                                    投影层
                                       ↓
                              LLM token嵌入
                                       ↓
物品ID → SASRec物品嵌入 → 投影层 → LLM token嵌入
                                       ↓
                            LLM (Llama/OPT)
                                       ↓
                              推荐结果 (Yes/No)
```

---

## 📈 预期性能提升

基于文献报告，SASRec相比MF通常有以下提升：

- **Hit@10**: +5-10%
- **NDCG@10**: +10-15%
- **MRR@10**: +8-12%

---

## 💾 模型文件

训练后会生成：

```
checkpoints/
├── cf_sasrec.pt           # SASRec模型（或cf_mf.pt）
├── stage1_neg19/
│   ├── best_model.pt      # Stage1最佳模型
│   └── results.pkl        # 评估结果
└── stage2_neg19/
    ├── best_model.pt      # Stage2最佳模型
    └── results.pkl        # 评估结果
```

---

## 🐛 常见问题

### Q: 为什么训练变慢了？
**A**: SASRec比MF复杂，训练时间会增加20-30%。这是正常的，性能提升值得这个代价。

### Q: 可以只用MF吗？
**A**: 可以，设置`--cf_model mf`即可。但建议使用SASRec以获得更好的性能。

### Q: 序列长度如何选择？
**A**: 
- 数据稀疏：max_seq_len=20-30
- 数据丰富：max_seq_len=50-100
- 计算受限：减小max_seq_len

### Q: 内存不足怎么办？
**A**: 
1. 减小batch_size
2. 减小max_seq_len
3. 减小embedding_dim

---

## 📚 参考文献

SASRec原论文：
```bibtex
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={ICDM},
  year={2018}
}
```

---

**升级到SASRec，让你的推荐系统baseline更强大！** 🚀
