"""Local MiniLM retrieval for training-data preparation."""
from typing import List
import numpy as np
from MGI_module.local_config import EMBEDDING_MODEL_PATH

class RAGProcessor:    
    def __init__(self, model_name: str = None, topk: int = 10, device: str = None):
        from sentence_transformers import SentenceTransformer
        from pathlib import Path
        model_path = Path(model_name or EMBEDDING_MODEL_PATH).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local embedding model not found: {model_path}")
        if topk < 1:
            raise ValueError("topk must be positive")
        self.model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
        self.topk = topk
    
    def retrieve(self, query: str, contexts: List[str]) -> List[str]:
        """
        检索与query最相关的top-k context片段
        
        Args:
            query: 查询问题
            contexts: context片段列表
        
        Returns:
            top-k相关片段列表（按原始顺序排列）
        """
        if not contexts:
            return []
        # 限制topk不超过contexts数量
        k = min(self.topk, len(contexts))
        
        # 计算embeddings
        query_emb = self.model.encode([query])[0]
        context_embs = self.model.encode(contexts)
        
        # 计算余弦相似度
        query_norm = query_emb / max(float(np.linalg.norm(query_emb)), 1e-12)
        context_norms = context_embs / np.maximum(np.linalg.norm(context_embs, axis=1, keepdims=True), 1e-12)
        similarities = np.dot(context_norms, query_norm)
        
        # 获取top-k索引
        topk_indices = np.argsort(similarities)[-k:][::-1]

        # 按原始顺序返回
        sorted_indices = sorted(topk_indices.tolist())
        return [contexts[i] for i in sorted_indices]

    def process(self, query: str, contexts: List[str]) -> str:
        """
        RAG处理：检索并拼接
        """
        retrieved = self.retrieve(query, contexts)
        return "\n\n".join(retrieved)
