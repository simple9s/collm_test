"""
推荐系统评估指标
支持: Hit@K, NDCG@K, MRR@K
K = 1, 5, 10, 20
"""
import numpy as np
from typing import List, Dict, Tuple


class RecommendationMetrics:
    """推荐系统评估指标计算器"""
    
    def __init__(self, k_list=[1, 5, 10, 20]):
        """
        Args:
            k_list: Top-K列表
        """
        self.k_list = k_list
    
    def hit_at_k(self, pred_items: List[int], true_items: List[int], k: int) -> float:
        """
        Hit@K: 预测的Top-K中是否包含真实物品
        
        Args:
            pred_items: 预测的物品列表（按分数排序）
            true_items: 真实交互的物品列表
            k: Top-K
            
        Returns:
            hit: 1 if hit, 0 otherwise
        """
        top_k = pred_items[:k]
        for item in true_items:
            if item in top_k:
                return 1.0
        return 0.0
    
    def ndcg_at_k(self, pred_items: List[int], true_items: List[int], k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        
        Args:
            pred_items: 预测的物品列表（按分数排序）
            true_items: 真实交互的物品列表
            k: Top-K
            
        Returns:
            ndcg: NDCG@K分数
        """
        top_k = pred_items[:k]
        
        # DCG: 实际的累积增益
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in true_items:
                # rel = 1 if relevant, 0 otherwise
                dcg += 1.0 / np.log2(i + 2)  # i+2 because index starts at 0
        
        # IDCG: 理想的累积增益
        idcg = 0.0
        for i in range(min(len(true_items), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        # NDCG
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def mrr_at_k(self, pred_items: List[int], true_items: List[int], k: int) -> float:
        """
        MRR@K: Mean Reciprocal Rank
        
        Args:
            pred_items: 预测的物品列表（按分数排序）
            true_items: 真实交互的物品列表
            k: Top-K
            
        Returns:
            rr: Reciprocal Rank (1/rank of first relevant item)
        """
        top_k = pred_items[:k]
        
        for i, item in enumerate(top_k):
            if item in true_items:
                return 1.0 / (i + 1)  # rank starts at 1
        
        return 0.0
    
    def evaluate_user(
        self, 
        pred_items: List[int], 
        true_items: List[int]
    ) -> Dict[str, float]:
        """
        评估单个用户的所有指标
        
        Args:
            pred_items: 预测的物品列表（按分数排序）
            true_items: 真实交互的物品列表
            
        Returns:
            metrics: 所有指标的字典
        """
        metrics = {}
        
        for k in self.k_list:
            metrics[f'Hit@{k}'] = self.hit_at_k(pred_items, true_items, k)
            metrics[f'NDCG@{k}'] = self.ndcg_at_k(pred_items, true_items, k)
            metrics[f'MRR@{k}'] = self.mrr_at_k(pred_items, true_items, k)
        
        return metrics
    
    def evaluate_batch(
        self,
        all_pred_items: List[List[int]],
        all_true_items: List[List[int]]
    ) -> Dict[str, float]:
        """
        评估一批用户的平均指标
        
        Args:
            all_pred_items: 所有用户的预测物品列表
            all_true_items: 所有用户的真实物品列表
            
        Returns:
            avg_metrics: 平均指标
        """
        all_metrics = []
        
        for pred, true in zip(all_pred_items, all_true_items):
            user_metrics = self.evaluate_user(pred, true)
            all_metrics.append(user_metrics)
        
        # 计算平均值
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics


class RankingEvaluator:
    """
    用于推荐系统的排序评估器
    支持Leave-One-Out场景
    """
    
    def __init__(self, k_list=[1, 5, 10, 20]):
        self.metrics_calculator = RecommendationMetrics(k_list)
        self.k_list = k_list
    
    def evaluate_ranking(
        self,
        scores: np.ndarray,
        true_item: int,
        candidate_items: np.ndarray
    ) -> Dict[str, float]:
        """
        评估单个用户的排序结果（Leave-One-Out场景）
        
        Args:
            scores: 候选物品的分数 [n_candidates]
            true_item: 真实物品ID
            candidate_items: 候选物品ID列表 [n_candidates]
            
        Returns:
            metrics: 评估指标
        """
        # 按分数排序
        sorted_indices = np.argsort(scores)[::-1]  # 降序
        pred_items = candidate_items[sorted_indices].tolist()
        
        # 计算指标
        metrics = self.metrics_calculator.evaluate_user(pred_items, [true_item])
        
        return metrics
    
    def evaluate_batch_ranking(
        self,
        all_scores: List[np.ndarray],
        all_true_items: List[int],
        all_candidate_items: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        批量评估排序结果
        
        Args:
            all_scores: 所有用户的候选物品分数
            all_true_items: 所有用户的真实物品
            all_candidate_items: 所有用户的候选物品列表
            
        Returns:
            avg_metrics: 平均指标
        """
        all_metrics = []
        
        for scores, true_item, candidates in zip(
            all_scores, all_true_items, all_candidate_items
        ):
            metrics = self.evaluate_ranking(scores, true_item, candidates)
            all_metrics.append(metrics)
        
        # 计算平均值
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    格式化输出指标
    
    Args:
        metrics: 指标字典
        prefix: 前缀（如"Val"、"Test"）
        
    Returns:
        formatted: 格式化的字符串
    """
    lines = []
    if prefix:
        lines.append(f"\n{'='*60}")
        lines.append(f"{prefix} Metrics")
        lines.append(f"{'='*60}")
    
    # 按K分组
    k_list = [1, 5, 10, 20]
    
    for k in k_list:
        hit_key = f'Hit@{k}'
        ndcg_key = f'NDCG@{k}'
        mrr_key = f'MRR@{k}'
        
        if hit_key in metrics:
            line = f"K={k:2d} | Hit: {metrics[hit_key]:.4f} | NDCG: {metrics[ndcg_key]:.4f} | MRR: {metrics[mrr_key]:.4f}"
            lines.append(line)
    
    return "\n".join(lines)


# ========== 示例使用 ==========

if __name__ == '__main__':
    # 示例1: 单个用户评估
    print("示例1: 单个用户评估")
    print("-" * 60)
    
    metrics = RecommendationMetrics(k_list=[1, 5, 10, 20])
    
    # 假设预测的Top-20物品（按分数排序）
    pred_items = [5, 12, 3, 8, 15, 20, 7, 9, 1, 14, 11, 6, 18, 2, 10, 4, 13, 16, 19, 17]
    # 真实物品
    true_items = [3, 8]  # 用户真正喜欢的物品
    
    user_metrics = metrics.evaluate_user(pred_items, true_items)
    
    for metric_name, value in user_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    print("\n" + "="*60)
    
    # 示例2: Leave-One-Out场景
    print("\n示例2: Leave-One-Out场景")
    print("-" * 60)
    
    evaluator = RankingEvaluator(k_list=[1, 5, 10, 20])
    
    # 候选物品（1个正样本 + 99个负样本）
    candidate_items = np.array([42] + list(range(100)))  # 42是正样本
    # 模型给出的分数
    scores = np.random.rand(100)
    scores[0] = 0.95  # 给正样本高分
    
    true_item = 42
    
    loo_metrics = evaluator.evaluate_ranking(scores, true_item, candidate_items)
    
    print(format_metrics(loo_metrics, "Leave-One-Out"))
    
    print("\n" + "="*60)
    
    # 示例3: 批量评估
    print("\n示例3: 批量评估")
    print("-" * 60)
    
    n_users = 100
    all_scores = [np.random.rand(100) for _ in range(n_users)]
    all_true_items = [np.random.randint(0, 100) for _ in range(n_users)]
    all_candidates = [np.arange(100) for _ in range(n_users)]
    
    # 给正样本设置高分（模拟好的模型）
    for i in range(n_users):
        true_idx = all_true_items[i]
        all_scores[i][true_idx] = 0.9 + np.random.rand() * 0.1
    
    batch_metrics = evaluator.evaluate_batch_ranking(
        all_scores, all_true_items, all_candidates
    )
    
    print(format_metrics(batch_metrics, "Batch Average"))
