"""
CoLLM: Collaborative Large Language Model for Recommendation
核心思想：将协同过滤的ID嵌入映射到LLM的语义空间
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ProjectionLayer(nn.Module):
    """将ID嵌入投影到LLM空间"""
    
    def __init__(self, cf_dim, llm_dim, n_tokens=1, hidden_scale=10):
        """
        Args:
            cf_dim: 协同过滤嵌入维度
            llm_dim: LLM隐藏层维度
            n_tokens: 将一个ID嵌入映射为几个token
            hidden_scale: 中间层放大倍数
        """
        super().__init__()
        
        self.n_tokens = n_tokens
        hidden_dim = cf_dim * hidden_scale
        
        self.projection = nn.Sequential(
            nn.Linear(cf_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, llm_dim * n_tokens)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, cf_embeddings):
        """
        Args:
            cf_embeddings: [batch_size, cf_dim]
        Returns:
            projected: [batch_size, n_tokens, llm_dim]
        """
        batch_size = cf_embeddings.size(0)
        projected = self.projection(cf_embeddings)  # [B, llm_dim * n_tokens]
        projected = projected.view(batch_size, self.n_tokens, -1)  # [B, n_tokens, llm_dim]
        return projected


class CoLLM(nn.Module):
    """
    CoLLM完整模型
    
    架构：
    1. 使用预训练的CF模型获取用户/物品ID嵌入
    2. 通过投影层将ID嵌入映射到LLM空间
    3. 将映射后的嵌入插入到提示词中
    4. LLM生成推荐结果（Yes/No）
    """
    
    # FIX: prompt 统一定义为类变量，避免 forward 和缓存逻辑中不一致
    PROMPT_TEXT = (
        "A user has given high ratings to the following books: <USER_EMB>. "
        "Additionally, we have information about the user's preferences. "
        "Using all available information, make a prediction about whether "
        "the user would enjoy the book <ITEM_EMB>? Answer with Yes or No.\nAnswer:"
    )

    def __init__(
        self,
        llm_name,
        cf_model,
        cf_dim=64,
        n_tokens=1,
        freeze_cf=True,
        freeze_llm=False,
        use_lora=True,
        lora_r=8,
        lora_alpha=16
    ):
        super().__init__()

        # 1. 加载LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModel.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )

        llm_dim = self.llm.config.hidden_size

        # 2. 协同过滤模型
        self.cf_model = cf_model
        if freeze_cf:
            for param in self.cf_model.parameters():
                param.requires_grad = False

        # 4. LLM设置
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        if use_lora and not freeze_llm:
            self._setup_lora(lora_r, lora_alpha)

        self.user_projection = ProjectionLayer(cf_dim, llm_dim, n_tokens)  # 保持 float32
        self.item_projection = ProjectionLayer(cf_dim, llm_dim, n_tokens)  # 保持 float32
        self.classifier = nn.Linear(llm_dim, 2)  # 保持 float32
        self._cached_prompt_emb = None

    def _setup_lora(self, r, alpha):
        """配置LoRA"""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

    def _get_prompt_emb(self, batch_size, device):
        """
        FIX: 懒加载并缓存 prompt embedding。
        prompt 文本固定不变，只在第一次调用时 tokenize + embedding，之后直接 expand。
        原来每个 batch 都重新 tokenize [prompt]*batch_size，完全冗余。
        """
        if self._cached_prompt_emb is None:
            tokens = self.tokenizer(
                self.PROMPT_TEXT,
                return_tensors='pt'
            ).to(device)
            with torch.no_grad():
                emb = self.llm.get_input_embeddings()(tokens.input_ids)  # [1, L, D]
            self._cached_prompt_emb = emb  # 缓存

        # expand 到 batch_size，不复制内存
        return self._cached_prompt_emb.expand(batch_size, -1, -1)  # [B, L, D]

    def insert_embeddings(self, input_embeds, user_embeds, item_embeds):
        """
        将用户和物品嵌入插入到提示词中

        Args:
            input_embeds: [B, seq_len, llm_dim]
            user_embeds: [B, n_tokens, llm_dim]
            item_embeds: [B, n_tokens, llm_dim]
        """
        new_embeds = torch.cat([
            user_embeds,    # 用户嵌入
            input_embeds,   # 原始提示词
            item_embeds     # 物品嵌入
        ], dim=1)

        return new_embeds

    def forward(self, user_ids=None, item_ids=None, user_seqs=None):
        """
        Args:
            user_ids: [batch_size] - 用于MF
            item_ids: [batch_size]
            user_seqs: [batch_size, seq_len] - 用于SASRec
        Returns:
            logits: [batch_size, 2]  (Yes/No的logits)
        """
        device = item_ids.device
        batch_size = item_ids.size(0)

        # 1. 获取CF嵌入
        with torch.no_grad():
            if user_seqs is not None:  # SASRec
                user_cf_emb = self.cf_model.get_user_embedding(user_seqs)  # [B, cf_dim]
            else:  # MF
                user_cf_emb = self.cf_model.get_user_embedding(user_ids)   # [B, cf_dim]

            item_cf_emb = self.cf_model.get_item_embedding(item_ids)       # [B, cf_dim]

        # 2. 投影到LLM空间
        user_llm_emb = self.user_projection(user_cf_emb.float()).half()
        item_llm_emb = self.item_projection(item_cf_emb.float()).half()

        # 3. FIX: 使用缓存的 prompt embedding，避免每个 batch 重复计算
        prompt_embeds = self._get_prompt_emb(batch_size, device)  # [B, L, llm_dim]

        # 4. 插入ID嵌入
        combined_embeds = self.insert_embeddings(prompt_embeds, user_llm_emb, item_llm_emb)

        # 5. LLM前向传播
        outputs = self.llm(inputs_embeds=combined_embeds)

        # 6. 取最后一个token的表示用于分类
        last_hidden = outputs.last_hidden_state[:, -1, :]  # [B, llm_dim]

        # 7. 分类
        logits = self.classifier(last_hidden.float())

        return logits

    def predict(self, user_ids=None, item_ids=None, user_seqs=None):
        """预测（推理模式）"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(user_ids, item_ids, user_seqs)
            probs = torch.softmax(logits, dim=-1)
            return probs[:, 1]  # 返回"Yes"的概率