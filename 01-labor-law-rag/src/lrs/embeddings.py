"""向量化层：默认 Sentence-BERT（BAAI/bge-small-zh-v1.5），离线时降级为哈希向量。

为什么要降级实现：
- 面试/答辩现场常常没网，或者 reviewer 不想下 100MB 模型；
- 降级实现必须**诚实**：它只是字符 n-gram 的哈希投影，不具备语义泛化能力，
  评测报告里会单独标注 `embedding_backend`，不会拿它冒充 BGE 的成绩。

bge 系列的检索用法：**query 侧加指令前缀，文档侧不加**（官方推荐），
这里用 BGE_QUERY_INSTRUCTION 实现，避免"直接拿句向量点积"的常见错误。
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod

import lrs.env  # noqa: F401  必须在 sentence_transformers 之前导入，关闭 hf-xet
import numpy as np

logger = logging.getLogger(__name__)

BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
HASH_DIM = 512


class Embedder(ABC):
    """embedder 接口：把文本批量映射为 L2 归一化向量。"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """文档侧向量，shape=(n, dim)，已 L2 归一化。"""

    def embed_query(self, text: str) -> np.ndarray:
        """查询侧向量。默认与文档相同，BGE 实现会加指令前缀。"""
        return self.embed_documents([text])[0]

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_query(t) for t in texts]) if texts else np.zeros((0, self.dim))


class SentenceBERTEmbedder(Embedder):
    """Sentence-BERT（BGE）实现。首次使用会下载模型并缓存到 HF cache。"""

    name = "sentence-transformers"

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32,
                 allow_download: bool = True):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        logger.info("加载 embedding 模型 %s（device=%s）", model_name, device)
        self.model = SentenceTransformer(
            model_name, device=device,
            local_files_only=not allow_download,
        )
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.name = f"sentence-transformers:{model_name}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # 中文检索任务 bge 官方建议 query 加指令前缀
        return self._encode([BGE_QUERY_INSTRUCTION + t for t in texts])


_cjk = re.compile(r"[\u4e00-\u9fff]")
_latin = re.compile(r"[a-zA-Z0-9]+")


def _char_ngrams(text: str, ns=(1, 2, 3)) -> list[str]:
    """中文字符 n-gram + 英文词，用于哈希向量。"""
    toks: list[str] = []
    for n in ns:
        toks.extend(text[i:i + n] for i in range(len(text) - n + 1))
    toks.extend(w.lower() for w in _latin.findall(text))
    return toks


class HashEmbedder(Embedder):
    """离线降级向量：字符 n-gram 的带符号哈希 + IDF 加权近似。

    它没有语义能力（"经济补偿" 与 "赔偿金" 不会相近），只做词面匹配，
    作用是在无模型/无网络时让整条链路（Chroma、RRF、评测）可以跑通和测试。
    """

    name = "hash-ngram(fallback)"
    dim = HASH_DIM

    def _vec(self, text: str) -> np.ndarray:
        import math

        vec = np.zeros(HASH_DIM, dtype=np.float32)
        toks = _char_ngrams(text)
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for tok, c in counts.items():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % HASH_DIM
            sign = 1.0 if (h >> 32) & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(c))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, HASH_DIM), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts])


def build_embedder(model_name: str, device: str = "cpu", batch_size: int = 32,
                   allow_download: bool = True) -> Embedder:
    """构造 embedder；模型不可用时降级为 HashEmbedder 并给出明显 warning。"""
    try:
        return SentenceBERTEmbedder(model_name, device=device, batch_size=batch_size,
                                    allow_download=allow_download)
    except Exception as e:  # ImportError / 下载失败 / 显存不足
        logger.warning("加载 %s 失败（%s: %s），降级为哈希向量。评测报告会标注该 backend。",
                       model_name, type(e).__name__, e)
        return HashEmbedder()
