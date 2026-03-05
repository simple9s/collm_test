#!/usr/bin/env python3
"""
快速测试脚本 - 验证CoLLM pipeline是否正常工作
使用小规模数据测试完整流程
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import shutil

print("="*60)
print("CoLLM Pipeline 快速测试")
print("="*60)

# 1. 生成模拟数据
print("\n[1/5] 生成模拟数据...")
tmp_dir = Path(tempfile.mkdtemp())
data_dir = tmp_dir / "test_data"
data_dir.mkdir()

# 生成小规模数据
n_users = 50
n_items = 100
n_interactions = 500

np.random.seed(42)

# 训练数据
train_data = pd.DataFrame({
    'uid': np.random.randint(0, n_users, n_interactions),
    'iid': np.random.randint(0, n_items, n_interactions),
    'label': np.random.randint(0, 2, n_interactions)
})

# 验证和测试数据
val_data = pd.DataFrame({
    'uid': np.random.randint(0, n_users, 100),
    'iid': np.random.randint(0, n_items, 100),
    'label': np.random.randint(0, 2, 100)
})

test_data = pd.DataFrame({
    'uid': np.random.randint(0, n_users, 100),
    'iid': np.random.randint(0, n_items, 100),
    'label': np.random.randint(0, 2, 100)
})

# 保存数据
neg_dir = data_dir / "neg_19"
neg_dir.mkdir()
train_data.to_pickle(neg_dir / "train.pkl")
val_data.to_pickle(neg_dir / "valid.pkl")
test_data.to_pickle(neg_dir / "test.pkl")

# 元数据
import pickle
meta = {
    'n_users': n_users,
    'n_items': n_items
}
with open(data_dir / 'meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print(f"✓ 数据已生成: {data_dir}")
print(f"  - 用户数: {n_users}")
print(f"  - 物品数: {n_items}")
print(f"  - 训练样本: {len(train_data)}")

# 2. 测试数据加载
print("\n[2/5] 测试数据加载...")
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import RecommendationDataset, collate_fn

dataset = RecommendationDataset(neg_dir / "train.pkl")
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

batch = next(iter(loader))
user_ids, item_ids, labels = batch
print(f"✓ 数据加载成功")
print(f"  - Batch shape: users={user_ids.shape}, items={item_ids.shape}, labels={labels.shape}")

# 3. 测试CF模型
print("\n[3/5] 测试CF模型...")
from models.mf import MatrixFactorization

cf_model = MatrixFactorization(n_users=n_users, n_items=n_items, embedding_dim=16)
scores = cf_model(user_ids, item_ids)
print(f"✓ CF模型创建成功")
print(f"  - 输出shape: {scores.shape}")

# 4. 测试评估指标
print("\n[4/5] 测试评估指标...")
from utils.metrics import RankingEvaluator

evaluator = RankingEvaluator(k_list=[1, 5, 10, 20])

# 模拟预测
all_scores = [np.random.rand(20) for _ in range(10)]
all_true_items = [np.random.randint(0, 20) for _ in range(10)]
all_candidates = [np.arange(20) for _ in range(10)]

metrics = evaluator.evaluate_batch_ranking(all_scores, all_true_items, all_candidates)
print(f"✓ 评估指标计算成功")
print(f"  - Hit@10: {metrics['Hit@10']:.4f}")
print(f"  - NDCG@10: {metrics['NDCG@10']:.4f}")
print(f"  - MRR@10: {metrics['MRR@10']:.4f}")

# 5. 测试完整训练流程（仅Stage 0 - CF预训练）
print("\n[5/5] 测试CF训练流程...")
cf_model = MatrixFactorization(n_users=n_users, n_items=n_items, embedding_dim=16)
optimizer = torch.optim.Adam(cf_model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()

# 训练几个epoch
cf_model.train()
for epoch in range(3):
    total_loss = 0
    for user_ids, item_ids, labels in loader:
        scores = cf_model(user_ids, item_ids)
        loss = criterion(scores, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")

print(f"✓ CF训练流程正常")

# 清理
print(f"\n清理临时文件: {tmp_dir}")
shutil.rmtree(tmp_dir)

print("\n" + "="*60)
print("✓ 所有测试通过！CoLLM pipeline工作正常")
print("="*60)
print("\n现在可以运行完整训练:")
print("  bash scripts/run_all.sh")
