# EA_Operators/embedding_utils.py
import numpy as np
import random
import string
from typing import List, Optional
import logging
import torch  # 必须导入 torch 以检测 MPS

logger = logging.getLogger(__name__)


class RobustPromptEmbedder:
    """鲁棒的提示嵌入器，支持多种嵌入方法，并针对 Mac M芯片优化"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_simple: bool = False):
        self.model = None
        self.use_simple = use_simple
        self.embedding_dim = 384  # 默认维度，与MiniLM匹配

        # 检测设备：优先使用 MPS (Mac GPU)，其次 CUDA，最后 CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("🚀 检测到 Mac GPU (MPS)，已启用硬件加速！")
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info("🚀 检测到 NVIDIA GPU (CUDA)，已启用硬件加速！")
        else:
            self.device = "cpu"
            logger.info("⚠️ 未检测到 GPU，使用 CPU 模式。")

        if not use_simple:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"正在加载模型: {model_name} 到 {self.device}...")

                # 加载模型到指定设备
                self.model = SentenceTransformer(model_name, device=self.device)

                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"模型加载成功，嵌入维度: {self.embedding_dim}")
            except Exception as e:
                logger.warning(f"无法加载SentenceTransformer: {e}，使用简单嵌入方法")
                self.use_simple = True
                self._init_simple_embedder()
        else:
            self._init_simple_embedder()

    def _init_simple_embedder(self):
        """初始化简单的基于哈希的嵌入器"""
        logger.info("使用简单嵌入方法")
        self.vocab = self._build_vocab()
        self.embedding_dim = 384  # 保持与MiniLM相同的维度

    def _build_vocab(self):
        """构建简单的词汇表"""
        chars = string.ascii_lowercase + string.digits + string.punctuation + " "
        return {char: idx for idx, char in enumerate(chars)}

    def encode(self, prompts: List[str]) -> np.ndarray:
        """
        生成嵌入向量
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if self.use_simple or self.model is None:
            return self._simple_encode(prompts)

        try:
            # SentenceTransformer 会自动使用初始化时指定的 device (mps)
            embeddings = self.model.encode(prompts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"模型编码失败: {e}，回退到简单编码")
            return self._simple_encode(prompts)

    def _simple_encode(self, prompts: List[str]) -> np.ndarray:
        """简单的编码实现：基于字符频率的特征向量"""
        embeddings = []
        for prompt in prompts:
            vec = np.zeros(self.embedding_dim)
            # 简单的哈希映射
            for i, char in enumerate(prompt.lower()):
                idx = ord(char) % self.embedding_dim
                vec[idx] += 1

            # 归一化
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        return np.array(embeddings)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算余弦相似度"""
        if embedding1.ndim > 1:
            embedding1 = embedding1.flatten()
        if embedding2.ndim > 1:
            embedding2 = embedding2.flatten()

        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class DummyEmbedder:
    """虚拟嵌入器，用于完全禁用嵌入功能 (w/o Mapping)"""

    def __init__(self):
        self.embedding_dim = 384

    def encode(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        # 返回随机向量，保持代码跑通
        return np.random.randn(len(prompts), self.embedding_dim)

    def similarity(self, embedding1, embedding2):
        return random.random()  # 随机相似度