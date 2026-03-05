"""
Amazon 2018 数据预处理
- 过滤交互次数<5的用户
- 留一法划分数据集
- 支持19和99负样本采样
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import argparse
from pathlib import Path

from utils import download_amazon
import json


class AmazonPreprocessor:
    def __init__(self, min_interactions=5):
        self.min_interactions = min_interactions

    def load_raw_data(self, year, category):
        """
        自动下载 + 读取 Amazon 数据

        Args:
            year: 2018 or 2023
            category: 比如 'Beauty', 'Software'
        """
        print(f"Preparing Amazon {year} - {category} dataset...")

        review_path, meta_path = download_amazon.ensure_amazon_dataset(year, category)

        print(f"Loading review file: {review_path}")

        users = []
        items = []
        ratings = []
        timestamps = []

        # 2018 是 json，每行一个 json
        if year == 2018:
            with open(review_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    users.append(data['reviewerID'])
                    items.append(data['asin'])
                    ratings.append(data.get('overall', 1.0))
                    timestamps.append(data['unixReviewTime'])

        # 2023 是 jsonl
        elif year == 2023:
            with open(review_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    users.append(data['user_id'])
                    items.append(data['parent_asin'])
                    ratings.append(data.get('rating', 1.0))
                    timestamps.append(data['timestamp'])

        df = pd.DataFrame({
            'user': users,
            'item': items,
            'rating': ratings,
            'timestamp': timestamps
        })

        print(f"Loaded {len(df)} interactions")

        return df
    
    def filter_users(self, df):
        """过滤交互次数<5的用户"""
        user_counts = df['user'].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        df_filtered = df[df['user'].isin(valid_users)]
        print(f"After filtering: {len(df_filtered)} interactions, {len(valid_users)} users")
        return df_filtered
    
    def remap_ids(self, df):
        """重新映射用户和物品ID为连续整数"""
        user_map = {u: i for i, u in enumerate(df['user'].unique())}
        item_map = {i: idx for idx, i in enumerate(df['item'].unique())}
        
        df['uid'] = df['user'].map(user_map)
        df['iid'] = df['item'].map(item_map)
        
        print(f"Remapped: {len(user_map)} users, {len(item_map)} items")
        return df, user_map, item_map
    
    def leave_one_out_split(self, df):
        """留一法划分数据集"""
        df = df.sort_values(['uid', 'timestamp'])
        
        train_data = []
        valid_data = []
        test_data = []
        
        for uid, group in df.groupby('uid'):
            interactions = group[['uid', 'iid', 'timestamp']].values
            
            if len(interactions) >= 3:
                train_data.extend(interactions[:-2])
                valid_data.append(interactions[-2])
                test_data.append(interactions[-1])
            elif len(interactions) == 2:
                train_data.extend(interactions[:-1])
                valid_data.append(interactions[-1])
            else:
                train_data.extend(interactions)
        
        train_df = pd.DataFrame(train_data, columns=['uid', 'iid', 'timestamp'])
        valid_df = pd.DataFrame(valid_data, columns=['uid', 'iid', 'timestamp'])
        test_df = pd.DataFrame(test_data, columns=['uid', 'iid', 'timestamp'])
        
        print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
        return train_df, valid_df, test_df
    
    def build_user_history(self, df):
        """构建用户历史交互序列（按时间排序）"""
        user_history = {}
        for uid, group in df.groupby('uid'):
            # 按时间排序
            items = group.sort_values('timestamp')['iid'].tolist()
            user_history[int(uid)] = items
        return user_history
    
    def sample_negatives(self, df, n_items, n_neg=19, user_history=None):
        """负采样（为每个正样本生成n_neg个负样本）"""
        data_with_neg = []
        
        for _, row in df.iterrows():
            uid, pos_iid = int(row['uid']), int(row['iid'])
            
            # 正样本
            data_with_neg.append({
                'uid': uid,
                'iid': pos_iid,
                'label': 1
            })
            
            # 负样本
            user_items = set(user_history[uid]) if user_history else set()
            neg_candidates = list(set(range(n_items)) - user_items - {pos_iid})
            
            neg_samples = np.random.choice(neg_candidates, size=min(n_neg, len(neg_candidates)), replace=False)
            
            for neg_iid in neg_samples:
                data_with_neg.append({
                    'uid': uid,
                    'iid': int(neg_iid),
                    'label': 0
                })
        
        return pd.DataFrame(data_with_neg)
    
    def generate_sequence_data(self, df, user_history, max_seq_len=50):
        """
        生成序列数据（用于SASRec）
        
        Args:
            df: 数据框
            user_history: 用户历史交互字典
            max_seq_len: 最大序列长度
        Returns:
            df_with_seq: 包含序列的数据框
        """
        data_with_seq = []
        
        for _, row in df.iterrows():
            uid, target_iid, label = int(row['uid']), int(row['iid']), int(row['label'])
            
            # 获取用户历史序列（排除当前目标物品）
            full_history = user_history.get(uid, [])
            
            # 找到目标物品在历史中的位置
            try:
                target_idx = full_history.index(target_iid)
                # 取目标物品之前的历史作为序列
                seq = full_history[:target_idx]
            except ValueError:
                # 如果是负样本，使用完整历史
                if label == 0:
                    seq = full_history.copy()
                else:
                    seq = full_history[:-1] if len(full_history) > 0 else []
            
            # 截断或填充序列
            if len(seq) > max_seq_len:
                seq = seq[-max_seq_len:]
            elif len(seq) < max_seq_len:
                # 用0填充（0是padding）
                seq = [0] * (max_seq_len - len(seq)) + seq
            
            data_with_seq.append({
                'uid': uid,
                'iid': target_iid,
                'seq': seq,
                'label': label
            })
        
        return pd.DataFrame(data_with_seq)
    
    def process(self, year, category, output_dir, neg_samples=[19, 99]):
        """完整预处理流程"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载并过滤数据
        df = self.load_raw_data(year, category)
        df = self.filter_users(df)
        
        # 2. 重映射ID
        df, user_map, item_map = self.remap_ids(df)
        
        # 3. 留一法划分
        train_df, valid_df, test_df = self.leave_one_out_split(df)
        
        # 4. 构建用户历史
        full_history = self.build_user_history(df)
        train_history = self.build_user_history(train_df)
        
        n_users = len(user_map)
        n_items = len(item_map)
        
        # 5. 保存元数据
        meta = {
            'n_users': n_users,
            'n_items': n_items,
            'user_map': user_map,
            'item_map': item_map
        }
        with open(output_dir / 'meta.pkl', 'wb') as f:
            pickle.dump(meta, f)
        
        # 6. 对每种负样本数生成数据
        for n_neg in neg_samples:
            neg_dir = output_dir / f'neg_{n_neg}'
            neg_dir.mkdir(exist_ok=True)
            
            print(f"\nGenerating data with {n_neg} negative samples...")
            
            # 训练集负采样
            train_final = self.sample_negatives(train_df, n_items, n_neg, train_history)
            valid_final = self.sample_negatives(valid_df, n_items, n_neg, full_history)
            test_final = self.sample_negatives(test_df, n_items, n_neg, full_history)
            
            # 保存MF格式数据（不含序列）
            train_final.to_pickle(neg_dir / 'train.pkl')
            valid_final.to_pickle(neg_dir / 'valid.pkl')
            test_final.to_pickle(neg_dir / 'test.pkl')
            
            print(f"✓ Saved MF format to {neg_dir}")
            print(f"  Train: {len(train_final)}")
            print(f"  Valid: {len(valid_final)}")
            print(f"  Test: {len(test_final)}")
            
            # 生成SASRec格式数据（含序列）
            max_seq_len = 50
            train_seq = self.generate_sequence_data(train_final, train_history, max_seq_len)
            valid_seq = self.generate_sequence_data(valid_final, full_history, max_seq_len)
            test_seq = self.generate_sequence_data(test_final, full_history, max_seq_len)
            
            # 保存SASRec格式数据
            train_seq.to_pickle(neg_dir / 'train_seq.pkl')
            valid_seq.to_pickle(neg_dir / 'valid_seq.pkl')
            test_seq.to_pickle(neg_dir / 'test_seq.pkl')
            
            print(f"✓ Saved SASRec format to {neg_dir}")
            print(f"  Train seq: {len(train_seq)}")
            print(f"  Valid seq: {len(valid_seq)}")
            print(f"  Test seq: {len(test_seq)}")
        
        print("\nPreprocessing complete!")
        return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True, help='2018 or 2023')
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--min_inter', type=int, default=5)
    parser.add_argument('--neg_samples', type=int, nargs='+', default=[19, 99])
    args = parser.parse_args()

    preprocessor = AmazonPreprocessor(min_interactions=args.min_inter)
    preprocessor.process(args.year, args.category, args.output, args.neg_samples)


if __name__ == '__main__':
    main()

'''
python data/preprocess_amazon.py \
  --year 2018 \
  --category Software \
  --output data/processed/software_2018
  
  
  python data/preprocess_amazon.py \
  --year 2023 \
  --category Beauty \
  --output data/processed/beauty_2023
'''