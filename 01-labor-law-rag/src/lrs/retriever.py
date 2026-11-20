"""混合检索：稠密向量 + BM25，用 RRF（Reciprocal Rank Fusion）融合。

为什么用 RRF 而不是加权分数相加：
- 稠密相似度（cosine，有界但尺度随模型漂移）与 BM25（无界）不可直接比较；
- RRF 只用排名：score = Σ 1/(k + rank)，对分数尺度不敏感、无需调权重、鲁棒；
- 论文（Cormack et al., 2009）与大量工程实践都验证它能稳定优于单路召回。
k 取 60 是原论文与业界默认值。

多查询融合：query plan 产生多个检索式（原问题 + 子查询 + 扩展词），
每个检索式各自跑 dense/BM25，再用「同一个 RRF 公式」把多组排名融合成最终排名，
等价于把多路召回的票数累加，避免某个子查询独占结果。
"""

from __future__ import annotations

import logging

import numpy as np

from lrs.embeddings import Embedder
from lrs.lexical import BM25Index
from lrs.schemas import Chunk, QueryPlan, Retrieved
from lrs.vector_store import ChromaStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """dense + BM25 双通道检索器。"""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, bm25: BM25Index,
                 store: ChromaStore, rrf_k: int = 60,
                 dense_weight: float = 1.0, bm25_weight: float = 1.0,
                 dense_enabled: bool = True, bm25_enabled: bool = True):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.embedder = embedder
        self.bm25 = bm25
        self.store = store
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.dense_enabled = dense_enabled
        self.bm25_enabled = bm25_enabled

    # ------------------------------------------------------------------ 单路召回
    def dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        qv = self.embedder.embed_query(query)
        return self.store.search(qv, top_k=top_k)

    def lexical_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        hits = self.bm25.search(query, top_k=top_k)
        return [(self.chunks[i].chunk_id, score) for i, score in hits]

    # ------------------------------------------------------------------ RRF
    @staticmethod
    def rrf_fuse(rankings: list[list[str]], k: int = 60,
                 weights: list[float] | None = None) -> dict[str, float]:
        """多组有序 chunk_id 列表 -> {chunk_id: rrf_score}。"""
        weights = weights or [1.0] * len(rankings)
        fused: dict[str, float] = {}
        for ranking, w in zip(rankings, weights):
            for rank, cid in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + w * (1.0 / (k + rank))
        return fused

    # ------------------------------------------------------------------ 主入口
    def retrieve(self, plan: QueryPlan, top_k: int = 10, *, use_decompose: bool = True,
                 use_expand: bool = True) -> list[Retrieved]:
        """按 query plan 做多路召回 + RRF 融合，返回 top_k 结果。

        查询变体的权重设计（实测调出来的）：
        - 原始问题 1.0：完整、是主信号；
        - 复合查询 0.9：原问题拼接法条术语，补上"拖欠工资→未及时足额支付劳动报酬"
          这类词面鸿沟，信号仍较"锐利"；
        - 分解子查询 0.8：补回多意图里漏掉的部分；
        - 术语扩展词 0.4：很短很泛，只作为弱补充。实测同权时"解除劳动合同"这类泛词
          会靠多路投票把第87条挤出 top-10，因此必须降权。

        每个查询变体各自跑 dense/BM25 两路，所有路一起做加权 RRF。
        """
        weighted_lists: list[tuple[list[str], float]] = []
        variants: list[tuple[str, float]] = [(plan.original, 1.0)]
        if use_expand:
            variants += [(q, 0.9) for q in plan.compound_queries]
        if use_decompose:
            variants += [(q, 0.8) for q in plan.sub_queries]
        if use_expand and plan.expansions:
            variants += [(q, 0.4) for q in plan.expansions]

        dense_scores: dict[str, float] = {}
        lexical_scores: dict[str, float] = {}
        dense_ids: set[str] = set()
        lexical_ids: set[str] = set()
        # 召回池：每个检索式取 30 条（语料 357 条，30 条约覆盖 8%），
        # 保证精准条款不会因为单路排名靠后而被融合淹没；重排阶段再压缩到 10 条。
        pool = max(top_k * 3, 30)
        for q, w in variants:
            if self.dense_enabled:
                d_hits = self.dense_search(q, top_k=pool)
            else:
                d_hits = []
            if self.bm25_enabled:
                l_hits = self.lexical_search(q, top_k=pool)
            else:
                l_hits = []
            if d_hits:
                weighted_lists.append(([cid for cid, _ in d_hits], w * self.dense_weight))
                for cid, s in d_hits:
                    dense_scores[cid] = max(dense_scores.get(cid, -1e9), s)
                dense_ids.update(cid for cid, _ in d_hits)
            if l_hits:
                weighted_lists.append(([cid for cid, _ in l_hits], w * self.bm25_weight))
                for cid, s in l_hits:
                    lexical_scores[cid] = max(lexical_scores.get(cid, -1e9), s)
                lexical_ids.update(cid for cid, _ in l_hits)

        rankings = [r for r, _ in weighted_lists]
        weights = [w for _, w in weighted_lists]
        fused = self.rrf_fuse(rankings, k=self.rrf_k, weights=weights)
        if not fused:
            return []

        # 归一化 RRF 分数，便于展示和理解（不影响排序）
        max_rrf = max(fused.values())
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        results: list[Retrieved] = []
        for rank, (cid, rrf) in enumerate(ordered, start=1):
            chunk = self.by_id[cid]
            source = ("both" if cid in dense_ids and cid in lexical_ids
                      else "dense" if cid in dense_ids else "bm25")
            results.append(
                Retrieved(
                    chunk_id=cid,
                    article_id=chunk.article_id,
                    law=chunk.law,
                    article_no=chunk.article_no,
                    title=f"《{chunk.law.replace('中华人民共和国', '')}》{chunk.article_no}",
                    text=chunk.article_text,
                    rank=rank,
                    source=source,
                    scores={
                        "dense": round(dense_scores.get(cid, 0.0), 4),
                        "bm25": round(lexical_scores.get(cid, 0.0), 4),
                        "rrf": round(rrf / max_rrf, 6) if max_rrf else 0.0,
                    },
                )
            )
        logger.debug("检索 %d 个检索式，融合出 %d 条结果", len(variants), len(results))
        return results
