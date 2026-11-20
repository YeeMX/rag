"""重排：BGE cross-encoder（bge-reranker-base）精排 + 离线降级。

为什么召回之后还要重排：
- 双塔（bi-encoder）把问题与法条**分别**编码，交互只发生在最后一步点积，精度有上限；
- cross-encoder 把 (query, passage) 拼起来一起过 Transformer，能做 token 级交互，
  对"法条术语相近但适用条件不同"的难例区分能力明显更强；
- 代价是慢（每个候选一次前向），所以只用 top-N 候选（默认 10）重排，不做全库。
这是经典的"粗排（双塔/BM25）→ 精排（cross-encoder）"两级检索架构。

降级实现（无网络/无模型时）：RRF 分数 + 词面覆盖率的线性组合。
它不能替代 cross-encoder 的语义交互能力，评测报告会明确标注 rerank_backend。
"""

from __future__ import annotations

import logging
import re

import lrs.env  # noqa: F401  必须在 sentence_transformers 之前导入，关闭 hf-xet
from lrs.lexical import tokenize
from lrs.schemas import Retrieved

logger = logging.getLogger(__name__)

RERANK_MAX_LEN = 512


class CrossEncoderReranker:
    """BAAI/bge-reranker-base 精排。bge reranker 不需要 query 指令前缀。"""

    name = "bge-reranker"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu",
                 max_length: int = RERANK_MAX_LEN):
        from sentence_transformers import CrossEncoder

        logger.info("加载重排模型 %s（device=%s）", model_name, device)
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.name = f"cross-encoder:{model_name}"

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        pairs = [(query, d[:RERANK_MAX_LEN * 3]) for d in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]


class OverlapReranker:
    """离线降级重排：RRF 分 + 查询词覆盖率（纯词面，无语义交互）。"""

    name = "overlap(fallback)"

    def __init__(self, rrf_weight: float = 0.5):
        self.rrf_weight = rrf_weight

    def score(self, query: str, docs: list[str]) -> list[float]:
        q_tokens = set(tokenize(query))
        out = []
        for d in docs:
            d_tokens = set(tokenize(d))
            overlap = len(q_tokens & d_tokens) / (len(q_tokens) or 1)
            out.append(overlap)
        return out


def build_reranker(model_name: str, device: str = "cpu", allow_download: bool = True):
    """优先 cross-encoder，失败降级。"""
    try:
        if not allow_download:
            raise RuntimeError("allow_download=False")
        return CrossEncoderReranker(model_name, device=device)
    except Exception as e:
        logger.warning("加载重排模型 %s 失败（%s: %s），降级为词面覆盖重排。",
                       model_name, type(e).__name__, e)
        return OverlapReranker()


_NORM_RE = re.compile(r"[\s，。；：、（）《》【】“”‘’！？,.;:()\[\]{}<>\"'!?—…·\-]+")


def _norm_query(query: str) -> str:
    return _NORM_RE.sub(" ", query)


def rerank(query: str, candidates: list[Retrieved], reranker,
           top_k: int = 5) -> list[Retrieved]:
    """对候选重排并截断到 top_k；保留各阶段分数轨迹。"""
    if not candidates:
        return []
    reranker = reranker or OverlapReranker()
    docs = [f"{c.title}：{c.text}" for c in candidates]
    scores = reranker.score(query, docs)
    if len(scores) != len(candidates):
        logger.warning("重排返回分数数量不匹配（%d vs %d），回退原顺序",
                       len(scores), len(candidates))
        return candidates[:top_k]
    for c, s in zip(candidates, scores):
        c.scores["rerank"] = round(float(s), 4)
    ranked = sorted(candidates, key=lambda c: c.scores.get("rerank", 0.0), reverse=True)
    for i, c in enumerate(ranked, start=1):
        c.rank = i
    return ranked[:top_k]
