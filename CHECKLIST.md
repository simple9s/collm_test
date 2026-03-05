# CoLLM Baseline 配置检查清单

## ✅ 数据配置

- [ ] Amazon 2018数据集已下载
- [ ] 数据格式正确: `user_id, item_id, rating, timestamp`
- [ ] 已运行数据预处理脚本
- [ ] 生成了neg_19和neg_99两个版本的数据
- [ ] meta.pkl包含正确的用户和物品数量

## ✅ 模型配置

### CF模型 (Stage 0)
- [ ] `cf_dim = 64` (可调整: 32, 64, 128, 256)
- [ ] `cf_epochs = 50` (根据收敛情况调整)
- [ ] `cf_lr = 1e-3` (可尝试: 1e-2, 1e-3, 1e-4)
- [ ] `batch_size = 1024` (根据内存调整)

### 投影层训练 (Stage 1)
- [ ] LLM选择: Llama-3.2-3B-Instruct 或 OPT-6.7B
- [ ] `freeze_cf = True` (必须)
- [ ] `freeze_llm = True` (必须)
- [ ] `n_tokens = 1` (可尝试: 1, 2, 3)
- [ ] `lr = 1e-3` (可尝试: 1e-2, 1e-3, 1e-4)
- [ ] `batch_size = 32` (根据GPU内存调整)

### LLM微调 (Stage 2)
- [ ] `freeze_cf = True` (必须)
- [ ] `freeze_llm = False` (必须)
- [ ] `use_lora = True` (推荐)
- [ ] `lora_r = 8` (可尝试: 4, 8, 16)
- [ ] `lora_alpha = 16` (通常为lora_r的2倍)
- [ ] `lr = 1e-4` (可尝试: 1e-4, 1e-5)
- [ ] `batch_size = 16` (根据GPU内存调整)

## ✅ 训练配置

- [ ] `num_epochs = 100` (可根据收敛情况调整)
- [ ] `patience = 10` (Early stopping)
- [ ] `weight_decay = 1e-3` (正则化)
- [ ] `seed = 42` (可重复性)
- [ ] `eval_type = 'ranking'` (推荐) 或 `'classification'` (可选)

**评估指标选择**:
- **ranking模式**: Hit@K, NDCG@K, MRR@K (K=1,5,10,20) - 推荐使用
- **classification模式**: AUC, ACC - 用于对比

## ✅ 负样本配置

选择一个:
- [ ] `n_neg = 19` (标准配置，推荐)
- [ ] `n_neg = 99` (更难，用于对比)

## ✅ 实验设置

### 必须运行的实验
1. [ ] Llama-3B + 19负样本
2. [ ] Llama-3B + 99负样本
3. [ ] OPT-6.7B + 19负样本
4. [ ] OPT-6.7B + 99负样本

### 消融实验（可选）
- [ ] 不同CF嵌入维度: 32, 64, 128
- [ ] 不同投影token数: 1, 2, 3
- [ ] 不同LoRA秩: 4, 8, 16
- [ ] Stage 1 vs Stage 2性能对比

## ✅ 输出验证

每次训练后检查:
- [ ] `checkpoints/cf_model.pt` 存在
- [ ] `checkpoints/stage1_neg{n}/best_model.pt` 存在
- [ ] `checkpoints/stage1_neg{n}/results.pkl` 存在
- [ ] `checkpoints/stage2_neg{n}/best_model.pt` 存在
- [ ] `checkpoints/stage2_neg{n}/results.pkl` 存在
- [ ] 日志中包含AUC和ACC指标

## ✅ 结果记录

创建一个表格记录所有实验结果:

### Ranking模式（推荐）

| LLM | Neg | Stage | Val NDCG@10 | Test Hit@10 | Test NDCG@10 | Test MRR@10 |
|-----|-----|-------|-------------|-------------|--------------|-------------|
| Llama-3B | 19 | 1 | ? | ? | ? | ? |
| Llama-3B | 19 | 2 | ? | ? | ? | ? |
| Llama-3B | 99 | 1 | ? | ? | ? | ? |
| Llama-3B | 99 | 2 | ? | ? | ? | ? |
| OPT-6.7B | 19 | 1 | ? | ? | ? | ? |
| OPT-6.7B | 19 | 2 | ? | ? | ? | ? |
| OPT-6.7B | 99 | 1 | ? | ? | ? | ? |
| OPT-6.7B | 99 | 2 | ? | ? | ? | ? |

### Classification模式（可选对比）

| LLM | Neg | Stage | Val AUC | Test AUC | Test ACC |
|-----|-----|-------|---------|----------|----------|
| Llama-3B | 19 | 1 | ? | ? | ? |
| Llama-3B | 19 | 2 | ? | ? | ? |

## ✅ 常见问题检查

- [ ] CUDA可用: `torch.cuda.is_available() == True`
- [ ] GPU内存充足 (建议至少24GB for Llama-3B)
- [ ] 数据路径正确
- [ ] 所有依赖包已安装
- [ ] Python版本 >= 3.8

## ✅ 论文写作检查

- [ ] 记录了所有超参数
- [ ] 记录了训练时间
- [ ] 记录了GPU型号和数量
- [ ] 保存了训练曲线（可选）
- [ ] 进行了显著性检验（可选）

## 📝 修改记录

### 相比原CoLLM的修改:

1. **数据集**: MovieLens/Amazon-old → Amazon 2018
2. **数据划分**: 随机划分 → 留一法
3. **负样本**: 固定负样本 → 支持19/99负样本
4. **用户过滤**: 无过滤 → 过滤交互<5的用户
5. **LLM**: Vicuna → Llama-3B / OPT-6.7B
6. **代码结构**: 复杂多文件 → 简化单脚本

### 保持不变的核心设计:

1. **三阶段训练**: CF预训练 → 投影层 → LLM微调
2. **投影机制**: ID嵌入 → 投影层 → LLM token
3. **提示词模板**: 协同过滤信息 + 自然语言描述
4. **LoRA微调**: 参数高效的LLM适配

## 🎯 运行建议

1. 先用小数据集测试完整流程
2. 确认每个阶段都能正常运行
3. 使用`n_neg=19`进行快速实验
4. 确认结果合理后运行`n_neg=99`
5. 保存所有实验日志和checkpoint
