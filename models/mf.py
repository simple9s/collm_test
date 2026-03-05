"""
SASRec (Self-Attentive Sequential Recommendation)
基于Transformer的序列推荐模型
"""
import torch
import torch.nn as nn
import numpy as np


class PointWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
    
    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, seq_len, hidden_units]
        Returns:
            outputs: [batch_size, seq_len, hidden_units]
        """
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRec(nn.Module):
    """
    SASRec模型
    用于序列推荐，生成用户和物品的嵌入
    """
    
    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=64,
        max_seq_len=50,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.2
    ):
        """
        Args:
            n_users: 用户数量
            n_items: 物品数量
            embedding_dim: 嵌入维度
            max_seq_len: 最大序列长度
            num_blocks: Transformer块数量
            num_heads: 注意力头数
            dropout_rate: Dropout率
        """
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # 物品嵌入（包括padding item 0）
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        
        # 位置嵌入
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Dropout
        self.emb_dropout = nn.Dropout(p=dropout_rate)
        
        # Multi-head Self-Attention块
        self.attention_layers = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.attention_layernorms = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        
        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(embedding_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            
            new_attn_layer = nn.MultiheadAttention(
                embedding_dim,
                num_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_layers.append(new_attn_layer)
            
            new_fwd_layernorm = nn.LayerNorm(embedding_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            
            new_fwd_layer = PointWiseFeedForward(embedding_dim, dropout_rate)
            self.forward_layers.append(new_fwd_layer)
        
        # 最终LayerNorm
        self.last_layernorm = nn.LayerNorm(embedding_dim, eps=1e-8)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, seq_items):
        """
        Args:
            seq_items: [batch_size, seq_len] 物品序列
        Returns:
            seq_output: [batch_size, seq_len, embedding_dim] 序列表示
        """
        batch_size, seq_len = seq_items.shape
        
        # 位置索引
        positions = torch.arange(seq_len, dtype=torch.long, device=seq_items.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # [B, L]
        
        # 物品嵌入 + 位置嵌入
        seq_emb = self.item_embedding(seq_items)  # [B, L, D]
        pos_emb = self.position_embedding(positions)  # [B, L, D]
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)
        
        # 创建因果mask（下三角矩阵）
        # 防止关注未来位置
        attn_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=seq_items.device))
        
        # 通过Transformer块
        for i in range(len(self.attention_layers)):
            # Multi-head Self-Attention
            Q = self.attention_layernorms[i](seq_emb)
            mha_outputs, _ = self.attention_layers[i](Q, Q, Q, attn_mask=attn_mask)
            seq_emb = Q + mha_outputs
            
            # Feed-Forward
            seq_emb = self.forward_layernorms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
        
        seq_output = self.last_layernorm(seq_emb)  # [B, L, D]
        
        return seq_output
    
    def predict(self, seq_items, target_items):
        """
        预测用户对目标物品的偏好
        
        Args:
            seq_items: [batch_size, seq_len] 历史序列
            target_items: [batch_size] 目标物品
        Returns:
            scores: [batch_size] 预测分数
        """
        # 获取序列表示
        seq_output = self.forward(seq_items)  # [B, L, D]
        
        # 取最后一个位置的表示作为用户表示
        user_emb = seq_output[:, -1, :]  # [B, D]
        
        # 获取目标物品嵌入
        target_emb = self.item_embedding(target_items)  # [B, D]
        
        # 计算得分
        scores = (user_emb * target_emb).sum(dim=-1)  # [B]
        
        return scores
    
    def get_user_embedding(self, seq_items):
        """
        获取用户嵌入（基于历史序列）
        
        Args:
            seq_items: [batch_size, seq_len] 历史序列
        Returns:
            user_emb: [batch_size, embedding_dim] 用户嵌入
        """
        seq_output = self.forward(seq_items)  # [B, L, D]
        user_emb = seq_output[:, -1, :]  # 取最后位置 [B, D]
        return user_emb
    
    def get_item_embedding(self, item_ids):
        """
        获取物品嵌入
        
        Args:
            item_ids: [batch_size] 物品ID
        Returns:
            item_emb: [batch_size, embedding_dim] 物品嵌入
        """
        return self.item_embedding(item_ids)


# 简化的MF模型（保持兼容性）
class MatrixFactorization(nn.Module):
    """
    矩阵分解模型（简化版，用于快速实验）
    建议使用SASRec以获得更好的性能
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 初始化
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        """
        Args:
            user_ids: [batch_size]
            item_ids: [batch_size]
        Returns:
            scores: [batch_size]
        """
        user_emb = self.user_embedding(user_ids)  # [B, D]
        item_emb = self.item_embedding(item_ids)  # [B, D]
        
        scores = (user_emb * item_emb).sum(dim=1)  # [B]
        return scores
    
    def get_user_embedding(self, user_ids):
        """获取用户嵌入"""
        return self.user_embedding(user_ids)
    
    def get_item_embedding(self, item_ids):
        """获取物品嵌入"""
        return self.item_embedding(item_ids)
