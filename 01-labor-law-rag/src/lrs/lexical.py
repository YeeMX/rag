"""BM25 关键词检索（稀疏通道）。

中文 BM25 的关键在于分词。为了不引入 jieba 这类额外依赖，同时保证召回质量，
这里用「单字 + 相邻二字」混合切分：
- 单字保证召回（"辞退"能被"辞"命中）；
- 二字组保证精度（"经济补偿"拆成 经济/济补/补偿，比只切单字更能区分语义）。

这种切分对法条这种术语密集的短文本效果很好，README 里有和 jieba 思路对比的说明。
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

from lrs.schemas import Chunk

logger = logging.getLogger(__name__)

_PUNCT = re.compile(r"[\s，。；：、（）《》【】“”‘’！？,.;:()\[\]{}<>\"'!?—…·\-]+")
_STOP_TOKENS = {"的了在是我有和就不人都一一个上也很到说要去", "的", "了", "是", "在"}


def tokenize(text: str) -> list[str]:
    """中文混合分词：单字 + 相邻二元组；英文/数字按词。"""
    text = _PUNCT.sub(" ", text.lower())
    tokens: list[str] = []
    for seg in text.split():
        cjk = re.findall(r"[\u4e00-\u9fff]+", seg)
        for run in cjk:
            tokens.extend(run)
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
        tokens.extend(re.findall(r"[a-z0-9]+", seg))
    return tokens


class BM25Index:
    """基于 rank_bm25 的倒排索引；未安装时退化为内置 BM25 实现。"""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("BM25 需要至少一个 chunk")
        self.chunks = chunks
        self.tokens: list[list[str]] = [tokenize(c.text) for c in chunks]
        self.k1 = k1
        self.b = b
        self._backend = "rank_bm25"
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self.tokens, k1=k1, b=b)
        except Exception:  # pragma: no cover - 依赖降级
            logger.warning("未安装 rank_bm25，使用内置 BM25 实现")
            self._bm25 = _SimpleBM25(self.tokens, k1=k1, b=b)
            self._backend = "builtin"

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """返回 [(chunk 下标, 分数)]，按分数降序，只保留 score>0。"""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, float(scores[i])) for i in order if scores[i] > 0]


class _SimpleBM25:
    """内置 BM25Okapi 实现（与 rank_bm25 公式一致），用于依赖缺失时的降级。"""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_len) / max(len(corpus), 1)
        self.doc_freqs: list[Counter] = [Counter(d) for d in corpus]
        df: Counter = Counter()
        for d in corpus:
            df.update(set(d))
        n = len(corpus)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self._doc_tokens = [set(d) for d in corpus]

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * len(self.corpus)
        for i, toks in enumerate(self.corpus):
            freq = Counter(toks)
            dl = self.doc_len[i] or 1
            s = 0.0
            for q in query:
                if q not in self.idf:
                    continue
                f = freq.get(q, 0)
                if f == 0:
                    continue
                s += self.idf[q] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                )
            scores[i] = s
        return scores
