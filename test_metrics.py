#!/usr/bin/env python3
"""
测试评估指标的正确性
"""
import numpy as np
from utils.metrics import RecommendationMetrics, RankingEvaluator, format_metrics


def test_hit_at_k():
    """测试Hit@K"""
    print("="*60)
    print("测试 Hit@K")
    print("="*60)
    
    metrics = RecommendationMetrics(k_list=[1, 5, 10, 20])
    
    # Case 1: 正样本在第1位
    pred_items = [5, 12, 3, 8, 15]
    true_items = [5]
    
    result = metrics.evaluate_user(pred_items, true_items)
    print("\nCase 1: 正样本在第1位")
    print(f"预测: {pred_items}")
    print(f"真实: {true_items}")
    print(f"Hit@1: {result['Hit@1']:.4f} (期望: 1.0)")
    print(f"Hit@5: {result['Hit@5']:.4f} (期望: 1.0)")
    
    # Case 2: 正样本在第3位
    pred_items = [12, 7, 5, 8, 15]
    true_items = [5]
    
    result = metrics.evaluate_user(pred_items, true_items)
    print("\nCase 2: 正样本在第3位")
    print(f"预测: {pred_items}")
    print(f"真实: {true_items}")
    print(f"Hit@1: {result['Hit@1']:.4f} (期望: 0.0)")
    print(f"Hit@5: {result['Hit@5']:.4f} (期望: 1.0)")
    
    # Case 3: 正样本不在Top-10
    pred_items = list(range(100))
    true_items = [99]
    
    result = metrics.evaluate_user(pred_items, true_items)
    print("\nCase 3: 正样本在第100位")
    print(f"Hit@1: {result['Hit@1']:.4f} (期望: 0.0)")
    print(f"Hit@10: {result['Hit@10']:.4f} (期望: 0.0)")
    print(f"Hit@20: {result['Hit@20']:.4f} (期望: 0.0)")


def test_ndcg_at_k():
    """测试NDCG@K"""
    print("\n" + "="*60)
    print("测试 NDCG@K")
    print("="*60)
    
    metrics = RecommendationMetrics(k_list=[1, 5, 10, 20])
    
    # Case 1: 完美排序（正样本在第1位）
    pred_items = [5, 12, 3, 8, 15]
    true_items = [5]
    
    result = metrics.evaluate_user(pred_items, true_items)
    print("\nCase 1: 正样本在第1位")
    print(f"预测: {pred_items}")
    print(f"真实: {true_items}")
    print(f"NDCG@1: {result['NDCG@1']:.4f} (期望: 1.0)")
    print(f"NDCG@5: {result['NDCG@5']:.4f} (期望: 1.0)")
    
    # Case 2: 正样本在第3位
    pred_items = [12, 7, 5, 8, 15]
    true_items = [5]
    
    result = metrics.evaluate_user(pred_items, true_items)
    print("\nCase 2: 正样本在第3位")
    print(f"预测: {pred_items}")
    print(f"真实: {true_items}")
    ndcg_expected = 1.0 / np.log2(4)  # 位置3+1=4
    print(f"NDCG@5: {result['NDCG@5']:.4f} (期望: {ndcg_expected:.4f})")
    
    # Case 3: 多个正样本
    pred_items = [5, 12, 8, 3, 15]
    true_items = [5, 8]
    
    result = metrics.evaluate_user(pred_items, true_items)
    print("\nCase 3: 两个正样本在第1和第3位")
    print(f"预测: {pred_items}")
    print(f"真实: {true_items}")
    dcg = 1.0/np.log2(2) + 1.0/np.log2(4)
    idcg = 1.0/np.log2(2) + 1.0/np.log2(3)
    ndcg_expected = dcg / idcg
    print(f"NDCG@5: {result['NDCG@5']:.4f} (期望: {ndcg_expected:.4f})")


def test_mrr_at_k():
    """测试MRR@K"""
    print("\n" + "="*60)
    print("测试 MRR@K")
    print("="*60)
    
    metrics = RecommendationMetrics(k_list=[1, 5, 10, 20])
    
    # Case 1: 正样本在第1位
    pred_items = [5, 12, 3, 8, 15]
    true_items = [5]
    
    result = metrics.evaluate_user(pred_items, true_items)
    print("\nCase 1: 正样本在第1位")
    print(f"预测: {pred_items}")
    print(f"真实: {true_items}")
    print(f"MRR@5: {result['MRR@5']:.4f} (期望: 1.0)")
    
    # Case 2: 正样本在第3位
    pred_items = [12, 7, 5, 8, 15]
    true_items = [5]
    
    result = metrics.evaluate_user(pred_items, true_items)
    print("\nCase 2: 正样本在第3位")
    print(f"预测: {pred_items}")
    print(f"真实: {true_items}")
    print(f"MRR@5: {result['MRR@5']:.4f} (期望: 0.3333)")
    
    # Case 3: 正样本不在Top-K
    pred_items = list(range(100))
    true_items = [99]
    
    result = metrics.evaluate_user(pred_items, true_items)
    print("\nCase 3: 正样本在第100位")
    print(f"MRR@10: {result['MRR@10']:.4f} (期望: 0.0)")
    print(f"MRR@20: {result['MRR@20']:.4f} (期望: 0.0)")


def test_ranking_evaluator():
    """测试排序评估器（Leave-One-Out场景）"""
    print("\n" + "="*60)
    print("测试 Leave-One-Out 排序评估")
    print("="*60)
    
    evaluator = RankingEvaluator(k_list=[1, 5, 10, 20])
    
    # 模拟场景: 1个正样本 + 99个负样本
    true_item = 42
    candidate_items = np.array([true_item] + list(range(0, 42)) + list(range(43, 100)))
    
    # Case 1: 模型给正样本最高分
    scores = np.random.rand(100)
    scores[0] = 1.0  # 正样本在第0个位置，给最高分
    
    result = evaluator.evaluate_ranking(scores, true_item, candidate_items)
    print("\nCase 1: 正样本得分最高")
    print(f"Hit@1: {result['Hit@1']:.4f} (期望: 1.0)")
    print(f"NDCG@1: {result['NDCG@1']:.4f} (期望: 1.0)")
    print(f"MRR@1: {result['MRR@1']:.4f} (期望: 1.0)")
    
    # Case 2: 正样本得分第5
    scores = np.random.rand(100)
    # 让4个负样本得分更高
    scores[1:5] = np.linspace(0.9, 0.95, 4)
    scores[0] = 0.85  # 正样本得分第5
    
    result = evaluator.evaluate_ranking(scores, true_item, candidate_items)
    print("\nCase 2: 正样本排在第5位")
    print(f"Hit@1: {result['Hit@1']:.4f} (期望: 0.0)")
    print(f"Hit@5: {result['Hit@5']:.4f} (期望: 1.0)")
    print(f"MRR@10: {result['MRR@10']:.4f} (期望: 0.2)")


def test_batch_evaluation():
    """测试批量评估"""
    print("\n" + "="*60)
    print("测试批量评估")
    print("="*60)
    
    evaluator = RankingEvaluator(k_list=[1, 5, 10, 20])
    
    # 模拟100个用户
    n_users = 100
    all_scores = []
    all_true_items = []
    all_candidates = []
    
    for i in range(n_users):
        # 每个用户有100个候选物品（1正 + 99负）
        true_item = np.random.randint(0, 100)
        candidates = np.arange(100)
        
        # 模拟分数（给正样本较高分）
        scores = np.random.rand(100)
        true_idx = np.where(candidates == true_item)[0][0]
        scores[true_idx] = 0.8 + np.random.rand() * 0.2  # 正样本得分0.8-1.0
        
        all_scores.append(scores)
        all_true_items.append(true_item)
        all_candidates.append(candidates)
    
    # 批量评估
    avg_metrics = evaluator.evaluate_batch_ranking(
        all_scores, all_true_items, all_candidates
    )
    
    print("\n批量评估结果 (100个用户):")
    print(format_metrics(avg_metrics, "Average"))


if __name__ == '__main__':
    print("\n" + "="*60)
    print("推荐系统评估指标测试")
    print("="*60)
    
    test_hit_at_k()
    test_ndcg_at_k()
    test_mrr_at_k()
    test_ranking_evaluator()
    test_batch_evaluation()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)
