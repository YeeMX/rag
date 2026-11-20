"""向量库：ChromaDB 持久化 + 余弦相似度检索。

设计要点：
- 自己算 embedding 再写入（`embeddings=`），把向量模型与向量库解耦，
  换模型只需要重建 collection，不影响上层逻辑；
- collection 名带上 backend 与模型名的短哈希，换模型不会读到旧向量（避免"改配置不生效"的坑）；
- 元数据里存 article_id / law / article_no，评测和引用回溯都靠它。
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from lrs.config import CHROMA_DIR
from lrs.embeddings import Embedder
from lrs.schemas import Chunk

logger = logging.getLogger(__name__)


def _collection_name(base: str, embedder: Embedder) -> str:
    sig = hashlib.md5(f"{embedder.name}|{embedder.dim}".encode()).hexdigest()[:8]
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in base)[:40]
    return f"{safe}-{sig}"


class ChromaStore:
    """Chroma 持久化向量库封装。"""

    def __init__(self, embedder: Embedder, persist_dir: Path | str = CHROMA_DIR,
                 collection: str = "labor_law"):
        import chromadb

        self.embedder = embedder
        self.persist_dir = Path(persist_dir)
        self.collection_name = _collection_name(collection, embedder)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "backend": embedder.name},
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def is_empty(self) -> bool:
        return self.count == 0

    def build(self, chunks: list[Chunk], force: bool = False) -> None:
        """全量重建索引。"""
        if force and self.count:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "backend": self.embedder.name},
            )
        if self.count:
            logger.info("collection %s 已有 %d 条，跳过重建", self.collection_name, self.count)
            return
        logger.info("为 %d 个 chunk 生成向量 ...", len(chunks))
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=texts,
            embeddings=[v.tolist() for v in embeddings],
            metadatas=[
                {
                    "article_id": c.article_id,
                    "law": c.law,
                    "article_no": c.article_no,
                    "chapter": c.chapter,
                    "span": c.span,
                }
                for c in chunks
            ],
        )
        logger.info("向量索引构建完成：%d 条", self.count)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """返回 [(chunk_id, cosine 相似度)]。"""
        res = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, max(self.count, 1)),
            include=["distances"],
        )
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        # cosine space 下 distance = 1 - similarity
        return [(cid, float(1.0 - d)) for cid, d in zip(ids, dists)]
