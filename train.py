"""
CoLLM训练脚本
支持两阶段训练：
Stage 1: 训练投影层（freeze CF + freeze LLM）
Stage 2: 训练LLM（freeze CF + LoRA微调LLM）
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score
import logging

from models.mf import MatrixFactorization, SASRec
from models.collm import CoLLM
from utils.metrics import RankingEvaluator, format_metrics


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommendationDataset(Dataset):
    """推荐数据集（MF格式）"""
    
    def __init__(self, data_path):
        data = pd.read_pickle(data_path)
        # FIX: 预加载为tensor，避免每次 __getitem__ 都做 iloc 行解析
        self.user_ids = torch.tensor(data['uid'].values, dtype=torch.long)
        self.item_ids = torch.tensor(data['iid'].values, dtype=torch.long)
        self.labels = torch.tensor(data['label'].values, dtype=torch.float)
        logger.info(f"Loaded {len(self.user_ids)} samples from {data_path}")

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'label': self.labels[idx]
        }


class SequenceDataset(Dataset):
    """序列推荐数据集（SASRec格式）"""

    def __init__(self, data_path):
        data = pd.read_pickle(data_path)
        # FIX: 预加载为tensor，避免每次 __getitem__ 都做 iloc 行解析
        self.user_ids = torch.tensor(data['uid'].values, dtype=torch.long)
        self.item_ids = torch.tensor(data['iid'].values, dtype=torch.long)
        self.seqs = torch.tensor(np.stack(data['seq'].values), dtype=torch.long)
        self.labels = torch.tensor(data['label'].values, dtype=torch.float)
        logger.info(f"Loaded {len(self.user_ids)} sequence samples from {data_path}")

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'seq': self.seqs[idx],
            'label': self.labels[idx]
        }


def collate_fn(batch):
    """MF数据批处理函数"""
    user_ids = torch.stack([item['user_id'] for item in batch])
    item_ids = torch.stack([item['item_id'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return user_ids, item_ids, labels


def collate_fn_seq(batch):
    """SASRec数据批处理函数"""
    user_ids = torch.stack([item['user_id'] for item in batch])
    item_ids = torch.stack([item['item_id'] for item in batch])
    seqs = torch.stack([item['seq'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return user_ids, item_ids, seqs, labels


class Trainer:
    """训练器"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        lr=1e-3,
        weight_decay=1e-3,
        patience=10,
        eval_type='ranking'  # 'ranking' or 'classification'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.patience = patience
        self.eval_type = eval_type

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 评估器
        self.ranking_evaluator = RankingEvaluator(k_list=[1, 5, 10, 20])

        # Early stopping
        self.best_metric = 0
        self.best_metric_name = 'NDCG@10'  # 主要指标
        self.wait = 0

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for i, batch_data in enumerate(pbar):
            if len(batch_data) == 4:  # SASRec格式
                user_ids, item_ids, seqs, labels = batch_data
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                seqs = seqs.to(self.device)
                labels = labels.to(self.device).long()

                # 前向传播
                logits = self.model(user_ids=user_ids, item_ids=item_ids, user_seqs=seqs)
            else:  # MF格式
                user_ids, item_ids, labels = batch_data
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                labels = labels.to(self.device).long()

                # 前向传播
                logits = self.model(user_ids=user_ids, item_ids=item_ids)

            loss = self.criterion(logits, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # FIX: 不每步都 .item()（会强制GPU→CPU同步），每50步更新一次进度条
            total_loss += loss.detach()
            if i % 50 == 0:
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss.item() / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, data_loader, split_name='Val'):
        """
        评估模型
        支持两种模式:
        1. ranking: 计算Hit@K, NDCG@K, MRR@K (推荐系统标准)
        2. classification: 计算AUC, ACC (二分类)
        """
        self.model.eval()

        if self.eval_type == 'ranking':
            return self._evaluate_ranking(data_loader, split_name)
        else:
            return self._evaluate_classification(data_loader, split_name)

    @torch.no_grad()
    def _evaluate_ranking(self, data_loader, split_name='Val'):
        """
        排序评估（Leave-One-Out场景）
        支持MF和SASRec两种格式
        """
        all_scores = []
        all_true_items = []
        all_candidate_items = []

        # 按用户分组收集数据
        user_data = {}

        pbar = tqdm(data_loader, desc=f'Evaluating {split_name}')
        for batch_data in pbar:
            if len(batch_data) == 4:  # SASRec格式
                user_ids, item_ids, seqs, labels = batch_data
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                seqs = seqs.to(self.device)

                # 获取预测分数
                logits = self.model(user_ids=user_ids, item_ids=item_ids, user_seqs=seqs)
            else:  # MF格式
                user_ids, item_ids, labels = batch_data
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)

                # 获取预测分数
                logits = self.model(user_ids=user_ids, item_ids=item_ids)

            probs = torch.softmax(logits, dim=-1)[:, 1]  # "Yes"的概率

            # 转换为numpy
            user_ids_np = user_ids.cpu().numpy()
            item_ids_np = item_ids.cpu().numpy()
            probs_np = probs.cpu().numpy()
            labels_np = labels.numpy()

            # 按用户分组
            for uid, iid, prob, label in zip(user_ids_np, item_ids_np, probs_np, labels_np):
                if uid not in user_data:
                    user_data[uid] = {
                        'items': [],
                        'scores': [],
                        'labels': []
                    }
                user_data[uid]['items'].append(iid)
                user_data[uid]['scores'].append(prob)
                user_data[uid]['labels'].append(label)

        # 为每个用户提取正样本和候选列表
        for uid, data in user_data.items():
            items = np.array(data['items'])
            scores = np.array(data['scores'])
            labels = np.array(data['labels'])

            # 找到正样本
            pos_idx = np.where(labels == 1)[0]
            if len(pos_idx) > 0:
                true_item = items[pos_idx[0]]
                all_true_items.append(true_item)
                all_candidate_items.append(items)
                all_scores.append(scores)

        # 计算排序指标
        metrics = self.ranking_evaluator.evaluate_batch_ranking(
            all_scores, all_true_items, all_candidate_items
        )

        # 打印结果
        logger.info(format_metrics(metrics, split_name))

        return metrics

    @torch.no_grad()
    def _evaluate_classification(self, data_loader, split_name='Val'):
        """二分类评估（计算AUC和ACC）"""
        all_preds = []
        all_labels = []

        pbar = tqdm(data_loader, desc=f'Evaluating {split_name}')
        for batch_data in pbar:
            if len(batch_data) == 4:  # SASRec格式
                user_ids, item_ids, seqs, labels = batch_data
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                seqs = seqs.to(self.device)

                logits = self.model(user_ids=user_ids, item_ids=item_ids, user_seqs=seqs)
            else:  # MF格式
                user_ids, item_ids, labels = batch_data
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)

                logits = self.model(user_ids=user_ids, item_ids=item_ids)

            probs = torch.softmax(logits, dim=-1)[:, 1]

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算指标
        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

        metrics = {'AUC': auc, 'ACC': acc}

        logger.info(f"{split_name} - AUC: {auc:.4f}, ACC: {acc:.4f}")

        return metrics

    def train(self, num_epochs, save_dir):
        """完整训练流程"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")

            # 训练
            train_loss = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f}")

            # 验证
            val_metrics = self.evaluate(self.val_loader, 'Val')

            # Early stopping（根据评估类型选择主要指标）
            if self.eval_type == 'ranking':
                current_metric = val_metrics[self.best_metric_name]
            else:
                current_metric = val_metrics['AUC']

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.wait = 0

                # 保存最佳模型
                torch.save(
                    self.model.state_dict(),
                    save_dir / 'best_model.pt'
                )
                logger.info(f"✓ Saved best model ({self.best_metric_name if self.eval_type == 'ranking' else 'AUC'}: {self.best_metric:.4f})")
            else:
                self.wait += 1
                logger.info(f"EarlyStopping counter: {self.wait}/{self.patience}")

                if self.wait >= self.patience:
                    logger.info("Early stopping triggered!")
                    break

        # 加载最佳模型并在测试集上评估
        logger.info("\n" + "="*50)
        logger.info("Testing with best model")
        logger.info("="*50)

        self.model.load_state_dict(torch.load(save_dir / 'best_model.pt'))
        test_metrics = self.evaluate(self.test_loader, 'Test')

        # 保存结果
        results = {
            'best_val_metric': self.best_metric,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

        with open(save_dir / 'results.pkl', 'wb') as f:
            pickle.dump(results, f)

        return results


def pretrain_cf(args, meta):
    """预训练协同过滤模型（MF或SASRec）"""
    logger.info("\n" + "="*50)
    logger.info(f"Stage 0: Pre-training CF Model ({args.cf_model})")
    logger.info("="*50)

    # 根据模型类型选择数据集
    if args.cf_model == 'sasrec':
        dataset = SequenceDataset(args.data_dir / f'neg_{args.n_neg}' / 'train_seq.pkl')
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn_seq,
            num_workers=12,
            pin_memory=True,     # FIX: 加速CPU→GPU传输
        )

        # 创建SASRec模型
        cf_model = SASRec(
            n_users=meta['n_users'],
            n_items=meta['n_items'],
            embedding_dim=args.cf_dim,
            max_seq_len=args.max_seq_len,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate
        ).to(args.device)
    else:
        dataset = RecommendationDataset(args.data_dir / f'neg_{args.n_neg}' / 'train.pkl')
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn,
            num_workers=12,
            pin_memory=True,     # FIX: 加速CPU→GPU传输
        )

        # 创建MF模型
        cf_model = MatrixFactorization(
            n_users=meta['n_users'],
            n_items=meta['n_items'],
            embedding_dim=args.cf_dim
        ).to(args.device)

    # 训练
    optimizer = torch.optim.Adam(cf_model.parameters(), lr=args.cf_lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    for epoch in range(args.cf_epochs):
        cf_model.train()
        total_loss = 0

        pbar = tqdm(loader, desc=f'CF Epoch {epoch+1}')

        if args.cf_model == 'sasrec':
            for i, (user_ids, item_ids, seqs, labels) in enumerate(pbar):
                user_ids = user_ids.to(args.device)
                item_ids = item_ids.to(args.device)
                seqs = seqs.to(args.device)
                labels = labels.to(args.device)

                scores = cf_model.predict(seqs, item_ids)
                loss = criterion(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # FIX: 减少 .item() 调用频率，避免每步GPU→CPU同步
                total_loss += loss.detach()
                if i % 50 == 0:
                    pbar.set_postfix({'loss': loss.item()})
        else:
            for i, (user_ids, item_ids, labels) in enumerate(pbar):
                user_ids = user_ids.to(args.device)
                item_ids = item_ids.to(args.device)
                labels = labels.to(args.device)

                scores = cf_model(user_ids, item_ids)
                loss = criterion(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # FIX: 减少 .item() 调用频率，避免每步GPU→CPU同步
                total_loss += loss.detach()
                if i % 50 == 0:
                    pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss.item() / len(loader)
        logger.info(f"CF Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # FIX: checkpoint路径加入数据集名称，多数据集不互相覆盖
            torch.save(cf_model.state_dict(), args.save_dir / f'cf_{args.cf_model}_{args.dataset}.pt')

    logger.info(f"✓ CF model ({args.cf_model}) saved with loss: {best_loss:.4f}")
    return cf_model


def main():
    parser = argparse.ArgumentParser()

    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称，如 beauty, sports, books')
    parser.add_argument('--n_neg', type=int, default=19, choices=[19, 99], help='负样本数')

    # 模型参数
    parser.add_argument('--llm_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                       choices=['meta-llama/Llama-3.2-3B-Instruct', 'facebook/opt-6.7b'])
    parser.add_argument('--cf_model', type=str, default='sasrec',
                       choices=['mf', 'sasrec'], help='CF模型类型')
    parser.add_argument('--cf_dim', type=int, default=64, help='CF嵌入维度')
    parser.add_argument('--n_tokens', type=int, default=1, help='投影token数')

    # SASRec专用参数
    parser.add_argument('--max_seq_len', type=int, default=50, help='最大序列长度')
    parser.add_argument('--num_blocks', type=int, default=2, help='Transformer块数')
    parser.add_argument('--num_heads', type=int, default=1, help='注意力头数')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout率')

    # 训练参数
    parser.add_argument('--stage', type=int, default=1, choices=[0, 1, 2],
                       help='0: 预训练CF, 1: 训练投影层, 2: 微调LLM')
    parser.add_argument('--cf_epochs', type=int, default=50, help='CF训练轮数')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    # FIX: lr 类型从 int 改为 float，原来 type=int 会把 1e-3 截断成 0
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--cf_lr', type=float, default=1e-3, help='CF学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='权重衰减')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    # LoRA参数
    parser.add_argument('--use_lora', action='store_true', help='使用LoRA')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')

    # 其他
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='保存目录')
    parser.add_argument('--cf_ckpt', type=str, default=None, help='预训练CF模型路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--eval_type', type=str, default='ranking',
                       choices=['ranking', 'classification'],
                       help='评估类型: ranking (Hit/NDCG/MRR) or classification (AUC/ACC)')

    args = parser.parse_args()

    # 设置
    args.data_dir = Path(args.data_dir)
    args.save_dir = Path(args.save_dir)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载元数据
    with open(args.data_dir / 'meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    logger.info(f"Dataset: {meta['n_users']} users, {meta['n_items']} items")
    logger.info(f"Negative samples: {args.n_neg}")
    logger.info(f"LLM: {args.llm_name}")
    # FIX: 打印实际学习率，方便排查
    logger.info(f"LR: {args.lr}, CF LR: {args.cf_lr}")

    # FIX: cf_ckpt 默认路径加入数据集名称
    if args.cf_ckpt is None:
        args.cf_ckpt = str(args.save_dir / f'cf_{args.cf_model}_{args.dataset}.pt')

    # Stage 0: 预训练CF
    if args.stage == 0:
        pretrain_cf(args, meta)
        return

    # 加载CF模型
    # FIX: 从checkpoint读取实际的n_items，避免数据集不同导致shape mismatch
    ckpt = torch.load(args.cf_ckpt, map_location='cpu')
    if args.cf_model == 'sasrec':
        n_items_from_ckpt = ckpt['item_embedding.weight'].shape[0] - 1  # 减去 padding_idx=0
        cf_model = SASRec(
            n_users=meta['n_users'],
            n_items=n_items_from_ckpt,
            embedding_dim=args.cf_dim,
            max_seq_len=args.max_seq_len,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate
        )
    else:
        n_items_from_ckpt = ckpt['item_embedding.weight'].shape[0]
        cf_model = MatrixFactorization(
            n_users=meta['n_users'],
            n_items=n_items_from_ckpt,
            embedding_dim=args.cf_dim
        )
    cf_model.load_state_dict(ckpt)
    logger.info(f"✓ Loaded CF model ({args.cf_model}) from {args.cf_ckpt} (n_items={n_items_from_ckpt})")

    # 创建数据加载器
    if args.cf_model == 'sasrec':
        train_loader = DataLoader(
            SequenceDataset(args.data_dir / f'neg_{args.n_neg}' / 'train_seq.pkl'),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_seq,
            num_workers=12, pin_memory=True
        )
        val_loader = DataLoader(
            SequenceDataset(args.data_dir / f'neg_{args.n_neg}' / 'valid_seq.pkl'),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_seq,
            num_workers=12, pin_memory=True
        )
        test_loader = DataLoader(
            SequenceDataset(args.data_dir / f'neg_{args.n_neg}' / 'test_seq.pkl'),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_seq,
            num_workers=12, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            RecommendationDataset(args.data_dir / f'neg_{args.n_neg}' / 'train.pkl'),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
            num_workers=12, pin_memory=True
        )
        val_loader = DataLoader(
            RecommendationDataset(args.data_dir / f'neg_{args.n_neg}' / 'valid.pkl'),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
            num_workers=12, pin_memory=True
        )
        test_loader = DataLoader(
            RecommendationDataset(args.data_dir / f'neg_{args.n_neg}' / 'test.pkl'),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
            num_workers=12, pin_memory=True
        )

    # 创建CoLLM模型
    logger.info("\n" + "="*50)
    logger.info(f"Stage {args.stage}: Building CoLLM Model")
    logger.info("="*50)

    model = CoLLM(
        llm_name=args.llm_name,
        cf_model=cf_model,
        cf_dim=args.cf_dim,
        n_tokens=args.n_tokens,
        freeze_cf=True,
        freeze_llm=(args.stage == 1),
        use_lora=(args.stage == 2 and args.use_lora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )

    # 训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        eval_type=args.eval_type
    )

    stage_dir = args.save_dir / f'stage{args.stage}_neg{args.n_neg}_{args.dataset}'
    results = trainer.train(args.num_epochs, stage_dir)

    logger.info("\n" + "="*50)
    logger.info("Final Results")
    logger.info("="*50)

    if args.eval_type == 'ranking':
        logger.info(f"Best Val {trainer.best_metric_name}: {results['best_val_metric']:.4f}")
        logger.info("\nTest Metrics:")
        for metric_name, value in results['test_metrics'].items():
            logger.info(f"  {metric_name}: {value:.4f}")
    else:
        logger.info(f"Best Val AUC: {results['best_val_metric']:.4f}")
        logger.info(f"Test AUC: {results['test_metrics']['AUC']:.4f}")
        logger.info(f"Test ACC: {results['test_metrics']['ACC']:.4f}")


if __name__ == '__main__':
    main()