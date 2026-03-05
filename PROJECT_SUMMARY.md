# CoLLM Baseline - 项目摘要

## ✅ 已完成功能

### 1. 数据处理 ✓
- [x] Amazon 2018数据集支持
- [x] 用户过滤（交互次数≥5）
- [x] 留一法数据划分
- [x] 负采样（支持19和99个负样本）
- [x] ID重映射为连续整数

### 2. 模型实现 ✓
- [x] 矩阵分解（MF）作为CF基线
- [x] 投影层（ID嵌入→LLM token）
- [x] CoLLM完整模型
- [x] LoRA集成
- [x] 支持Llama-3.2-3B-Instruct
- [x] 支持OPT-6.7B

### 3. 评估指标 ✓
- [x] **Hit@K** (K=1,5,10,20)
- [x] **NDCG@K** (K=1,5,10,20) 
- [x] **MRR@K** (K=1,5,10,20)
- [x] AUC（可选）
- [x] ACC（可选）
- [x] Leave-One-Out评估
- [x] 批量评估

### 4. 训练流程 ✓
- [x] Stage 0: CF预训练
- [x] Stage 1: 投影层训练
- [x] Stage 2: LoRA微调
- [x] Early stopping
- [x] 自动保存最佳模型
- [x] 完整日志记录

### 5. 代码质量 ✓
- [x] 模块化设计
- [x] 详细注释
- [x] 类型提示
- [x] 错误处理
- [x] 进度条显示

### 6. 文档 ✓
- [x] README.md（快速开始）
- [x] USAGE_GUIDE.md（完整指南）
- [x] CHECKLIST.md（配置清单）
- [x] INNOVATION.md（创新点说明）
- [x] 代码注释

### 7. 测试 ✓
- [x] 评估指标单元测试
- [x] Pipeline集成测试
- [x] 示例数据生成

---

## 📂 文件结构

```
collm_baseline/
├── data/
│   └── preprocess_amazon.py        # 数据预处理脚本
├── models/
│   ├── __init__.py
│   ├── mf.py                        # 矩阵分解模型
│   └── collm.py                     # CoLLM主模型
├── utils/
│   ├── __init__.py
│   └── metrics.py                   # 评估指标（Hit/NDCG/MRR）
├── scripts/
│   └── run_all.sh                   # 一键运行脚本
├── train.py                         # 训练脚本
├── test_metrics.py                  # 指标测试
├── test_pipeline.py                 # Pipeline测试
├── requirements.txt                 # 依赖包
├── README.md                        # 项目说明
├── USAGE_GUIDE.md                   # 使用指南
├── CHECKLIST.md                     # 配置清单
└── INNOVATION.md                    # 创新说明
```

---

## 🎯 核心特性

### 1. 完整的推荐系统评估指标

**Hit@K**: 命中率
```python
Hit@10 = 在Top-10中找到真实物品的用户比例
```

**NDCG@K**: 归一化折扣累积增益
```python
NDCG@10 = DCG@10 / IDCG@10
考虑了排序位置的重要性
```

**MRR@K**: 平均倒数排名
```python
MRR@10 = 1 / (第一个相关物品的排名)
```

### 2. 灵活的实验配置

```bash
# 切换LLM
--llm_name meta-llama/Llama-3.2-3B-Instruct
--llm_name facebook/opt-6.7b

# 切换负样本数
--n_neg 19
--n_neg 99

# 切换评估模式
--eval_type ranking      # Hit/NDCG/MRR
--eval_type classification  # AUC/ACC
```

### 3. 渐进式训练

```
Stage 0 (CF预训练)
    ↓ freeze CF
Stage 1 (投影层训练)
    ↓ freeze CF + freeze LLM
Stage 2 (LoRA微调)
    freeze CF + LoRA微调LLM
```

---

## 📊 预期结果

根据CoLLM原论文和类似工作，预期结果范围：

### MovieLens-1M (参考)
- Hit@10: 0.15 - 0.35
- NDCG@10: 0.10 - 0.25
- MRR@10: 0.08 - 0.20

### Amazon Books (本实现)
- Hit@10: 0.20 - 0.40（数据规模更大）
- NDCG@10: 0.12 - 0.28
- MRR@10: 0.10 - 0.22

**注**: 实际结果取决于数据质量、超参数、训练时间等因素

---

## 🔧 关键超参数

| 参数 | 推荐值 | 可选范围 |
|------|--------|----------|
| cf_dim | 64 | 32, 64, 128, 256 |
| n_tokens | 1 | 1, 2, 3 |
| lr (Stage 1) | 1e-3 | 1e-2, 1e-3, 1e-4 |
| lr (Stage 2) | 1e-4 | 1e-3, 1e-4, 1e-5 |
| lora_r | 8 | 4, 8, 16, 32 |
| lora_alpha | 16 | 2*lora_r |
| batch_size (S1) | 32 | 16, 32, 64 |
| batch_size (S2) | 16 | 8, 16, 32 |
| patience | 10 | 5, 10, 20 |

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
python data/preprocess_amazon.py \
    --input amazon_books_2018.csv \
    --output ./data/amazon2018 \
    --min_inter 5 \
    --neg_samples 19 99

# 3. 运行训练
bash scripts/run_all.sh

# 4. 查看结果
python -c "
import pickle
with open('checkpoints/stage2_neg19/results.pkl', 'rb') as f:
    print(pickle.load(f))
"
```

---

## 📈 实验建议

### 必做实验（基线）

1. **Llama-3B + 19负样本** (约4小时)
2. **Llama-3B + 99负样本** (约6小时)

### 可选实验（对比）

3. OPT-6.7B + 19负样本
4. OPT-6.7B + 99负样本

### 消融实验

- CF嵌入维度: 32 vs 64 vs 128
- 投影token数: 1 vs 2 vs 3
- LoRA秩: 4 vs 8 vs 16
- Stage 1 vs Stage 2性能对比

---

## ⚠️ 注意事项

### 1. GPU要求
- 最低: 16GB显存（Llama-3B）
- 推荐: 24GB显存（OPT-6.7B）
- 最佳: 40GB显存（多个实验并行）

### 2. 训练时间
- CF预训练: 5-10分钟
- Stage 1: 1-2小时
- Stage 2: 2-4小时
- 总计: 约4-6小时/实验

### 3. 磁盘空间
- 原始数据: ~1GB
- 预处理数据: ~500MB
- 模型检查点: ~3GB/模型
- 建议预留: 20GB

---

## 🐛 已知问题

1. **CUDA Out of Memory**: 减小batch_size
2. **训练不收敛**: 调整学习率
3. **评估指标为0**: 检查数据格式

---

## 📚 相关资源

- **原论文**: CoLLM (待发表)
- **数据集**: Amazon Review Data
- **LLM**: Hugging Face Model Hub
- **工具**: PyTorch, Transformers, PEFT

---

## ✉️ 联系方式

如有问题或建议，请：
1. 查看文档: README.md, USAGE_GUIDE.md
2. 运行测试: test_metrics.py
3. 提Issue或联系作者

---

## 📌 版本信息

- **版本**: v2.0
- **更新日期**: 2024
- **Python**: >=3.8
- **PyTorch**: >=2.0
- **状态**: 可直接运行 ✓

---

**祝实验顺利！🎉**
