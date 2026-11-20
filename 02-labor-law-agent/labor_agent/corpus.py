"""劳动法法条语料加载与 BM25 检索。

数据来源：``shared/corpus/labor_law_articles.jsonl``（4 部法律共 357 条真实法条）。

设计要点：
- 纯离线：只依赖标准库；``rank_bm25`` 已安装时优先使用，缺失时自动降级为
  本文件内的纯标准库 BM25 实现（公式与 rank_bm25 的 BM25Okapi 一致）。
- 中文分词不依赖 jieba：采用「单字 + 二元组（bigram）」切分。法律文本术语高度
  固定（如「经济补偿」「二倍工资」「竞业限制」），二元组比通用分词更稳，也不会
  引入额外依赖。
- 索引懒加载 + 进程内缓存，import 本模块本身很轻。
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------

CORPUS_PATH_ENV = "LABOR_LAW_CORPUS"
"""环境变量：覆盖法条语料路径（测试/换语料时使用）。"""

LAW_IDS: tuple[str, ...] = ("lcl", "ll", "lda", "sil")

LAW_NAMES: dict[str, str] = {
    "lcl": "中华人民共和国劳动合同法",
    "ll": "中华人民共和国劳动法",
    "lda": "中华人民共和国劳动争议调解仲裁法",
    "sil": "中华人民共和国社会保险法",
}

LAW_SHORT_NAMES: dict[str, str] = {
    "lcl": "劳动合同法",
    "ll": "劳动法",
    "lda": "劳动争议调解仲裁法",
    "sil": "社会保险法",
}

#: 法律别名 -> law_id。注意「劳动法」是「劳动合同法」的子串，解析时必须先长后短。
_LAW_ALIASES: dict[str, str] = {
    "lcl": "lcl",
    "ll": "ll",
    "lda": "lda",
    "sil": "sil",
    "劳动合同法": "lcl",
    "中华人民共和国劳动合同法": "lcl",
    "劳动法": "ll",
    "中华人民共和国劳动法": "ll",
    "劳动争议调解仲裁法": "lda",
    "中华人民共和国劳动争议调解仲裁法": "lda",
    "社会保险法": "sil",
    "中华人民共和国社会保险法": "sil",
}

_CN_DIGITS: dict[str, int] = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS: dict[str, int] = {"十": 10, "百": 100, "千": 1000}
_CN_INT_CHARS = "".join(_CN_DIGITS) + "".join(_CN_UNITS)


class CorpusError(RuntimeError):
    """语料缺失或格式错误。"""


# ---------------------------------------------------------------------------
# 中文数字与条号规范化
# ---------------------------------------------------------------------------


def chinese_to_int(text: str) -> int | None:
    """把中文数字转换为整数，失败返回 ``None``。

    支持 ``十二``、``二十一``、``一百零七`` 等常见写法，足以覆盖四部法律
    （最大条号 107）以及「三个月」「三倍」等法条数字。
    """
    if not text:
        return None
    total = 0
    current = 0
    for ch in text.strip():
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
        else:
            return None
    return total + current


def int_to_chinese(number: int) -> str:
    """把正整数转换为中文数字（1..9999），如 ``47 -> 四十七``。"""
    if not isinstance(number, int):
        raise TypeError(f"number 必须是 int，收到 {type(number).__name__}")
    if number <= 0:
        raise ValueError(f"number 必须为正整数，收到 {number}")
    digits = "零一二三四五六七八九"
    units = ("", "十", "百", "千")
    text = str(number)
    if len(text) > 4:
        raise ValueError(f"暂只支持 1..9999，收到 {number}")
    out: list[str] = []
    zero_pending = False
    for idx, ch in enumerate(text):
        digit = int(ch)
        pos = len(text) - idx - 1
        if digit == 0:
            zero_pending = bool(out)
            continue
        if zero_pending:
            out.append("零")
            zero_pending = False
        # 「十二」而不是「一十二」；但「一百一十二」保留「一」。
        if not (digit == 1 and pos == 1 and not out):
            out.append(digits[digit])
        out.append(units[pos])
    return "".join(out)


def normalize_article_no(value: str | int) -> str | None:
    """把各种条号写法规范成 ``第四十七条`` 形式，无法解析返回 ``None``。

    接受：``47`` / ``"47"`` / ``"第47条"`` / ``"第四十七条"`` / ``"四十七"``。
    """
    if isinstance(value, int):
        if value <= 0:
            return None
        try:
            return f"第{int_to_chinese(value)}条"
        except ValueError:
            return None
    if not isinstance(value, str):
        return None
    text = value.strip().replace(" ", "").replace("　", "")
    if not text:
        return None
    text = re.sub(r"^第", "", text)
    text = re.sub(r"条$", "", text)
    if not text:
        return None
    if text.isdigit():
        number = int(text)
    else:
        number = chinese_to_int(text)
    if number is None or number <= 0:
        return None
    try:
        return f"第{int_to_chinese(number)}条"
    except ValueError:
        return None


def resolve_law_id(value: str | None) -> str | None:
    """把法律 id / 中文名 / 简称解析为 ``law_id``，无法识别返回 ``None``。"""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in _LAW_ALIASES:
        return _LAW_ALIASES[lowered]
    # 长名称优先，避免「劳动法」抢先匹配「劳动合同法」。
    if "劳动合同法" in text:
        return "lcl"
    if "调解仲裁法" in text:
        return "lda"
    if "社会保险法" in text:
        return "sil"
    if "劳动法" in text:
        return "ll"
    return None


# ---------------------------------------------------------------------------
# 分词与 BM25
# ---------------------------------------------------------------------------

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_RUN = re.compile(r"[A-Za-z0-9_.]+")


def tokenize(text: str) -> list[str]:
    """中英混合切词：英文/数字整体保留，中文按「单字 + 二元组」切分。"""
    if not text:
        return []
    tokens: list[str] = [m.group(0).lower() for m in _LATIN_RUN.finditer(text)]
    for match in _CJK_RUN.finditer(text):
        run = match.group(0)
        tokens.extend(run)  # 单字：保证召回
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # 二元组：保证精度
    return tokens


try:  # pragma: no cover - 取决于环境是否安装 rank_bm25
    from rank_bm25 import BM25Okapi as _RankBM25
except ImportError:  # pragma: no cover
    _RankBM25 = None  # type: ignore[assignment]

BM25_BACKEND: str = "rank_bm25" if _RankBM25 is not None else "stdlib"


class _SimpleBM25:
    """纯标准库 BM25，公式对齐 rank_bm25.BM25Okapi（k1=1.5, b=0.75, eps=0.25）。"""

    def __init__(
        self,
        corpus_tokens: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
        eps: float = 0.25,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.eps = eps
        self.corpus_size = len(corpus_tokens)
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus_tokens]
        self.doc_lens: list[int] = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size else 0.0

        doc_freq: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            doc_freq.update(freqs.keys())
        self.idf: dict[str, float] = {}
        negative: list[str] = []
        for token, freq in doc_freq.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[token] = idf
            if idf < 0:
                negative.append(token)
        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0
        for token in negative:
            self.idf[token] = self.eps * self.average_idf

    def get_scores(self, query_tokens: Sequence[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        if not self.avgdl:
            return scores
        for token in query_tokens:
            idf = self.idf.get(token)
            if not idf:
                continue
            for idx, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(token, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_lens[idx] / self.avgdl
                )
                scores[idx] += idf * tf * (self.k1 + 1) / denom
        return scores


class BM25Index:
    """BM25 索引薄封装：优先 rank_bm25，缺失时用标准库实现兜底。"""

    def __init__(self, tokenized_docs: Sequence[Sequence[str]], **kwargs: Any) -> None:
        self._docs = list(tokenized_docs)
        if _RankBM25 is not None:
            self._impl: Any = _RankBM25(self._docs, **kwargs)
            self.backend = "rank_bm25"
        else:
            self._impl = _SimpleBM25(self._docs, **kwargs)
            self.backend = "stdlib"

    def get_scores(self, query_tokens: Sequence[str]) -> list[float]:
        if not query_tokens or not self._docs:
            return [0.0] * len(self._docs)
        return [float(score) for score in self._impl.get_scores(list(query_tokens))]


# ---------------------------------------------------------------------------
# 法条对象与语料库
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Article:
    """一条法条。"""

    id: str
    law_id: str
    law: str
    chapter: str
    article_no: str
    article_num: int
    title: str
    text: str
    paragraphs: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Article":
        required = ("id", "law_id", "law", "article_no", "article_num", "title", "text")
        missing = [key for key in required if key not in raw]
        if missing:
            raise CorpusError(f"法条记录缺少字段 {missing}: {raw.get('id', raw)!r}")
        return cls(
            id=str(raw["id"]),
            law_id=str(raw["law_id"]),
            law=str(raw["law"]),
            chapter=str(raw.get("chapter", "")),
            article_no=str(raw["article_no"]),
            article_num=int(raw["article_num"]),
            title=str(raw["title"]),
            text=str(raw["text"]),
            paragraphs=tuple(str(p) for p in raw.get("paragraphs", [])),
        )

    @property
    def char_len(self) -> int:
        return len(self.text)

    def to_dict(self, text_max_chars: int | None = None) -> dict[str, Any]:
        text = self.text
        if text_max_chars is not None and text_max_chars > 0 and len(text) > text_max_chars:
            text = text[:text_max_chars] + "…"
        return {
            "id": self.id,
            "law_id": self.law_id,
            "law": self.law,
            "law_short": LAW_SHORT_NAMES.get(self.law_id, self.law),
            "chapter": self.chapter,
            "article_no": self.article_no,
            "article_num": self.article_num,
            "title": self.title,
            "text": text,
            "char_len": self.char_len,
        }


def default_corpus_path() -> Path:
    """默认语料路径：环境变量优先，否则按仓库布局定位 ``shared/corpus``。"""
    override = os.getenv(CORPUS_PATH_ENV)
    if override:
        return Path(override)
    # labor_agent/corpus.py -> parents[0]=labor_agent, [1]=项目根, [2]=仓库根
    return Path(__file__).resolve().parents[2] / "shared" / "corpus" / "labor_law_articles.jsonl"


def load_articles(path: str | Path | None = None) -> list[Article]:
    """从 JSONL 读取全部法条。文件缺失/为空时抛出 :class:`CorpusError`。"""
    corpus_path = Path(path) if path is not None else default_corpus_path()
    if not corpus_path.exists():
        raise CorpusError(
            f"法条语料不存在: {corpus_path}\n"
            f"请确认 shared/corpus/labor_law_articles.jsonl 存在，"
            f"或通过环境变量 {CORPUS_PATH_ENV} 指定路径。"
        )
    articles: list[Article] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{corpus_path}:{line_no} JSON 解析失败: {exc}") from exc
            articles.append(Article.from_json(raw))
    if not articles:
        raise CorpusError(f"法条语料为空: {corpus_path}")
    return articles


class Corpus:
    """法条语料库：负责加载、条号索引与 BM25 检索。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_corpus_path()
        self.articles: list[Article] = load_articles(self.path)
        self._by_key: dict[tuple[str, str], Article] = {
            (article.law_id, article.article_no): article for article in self.articles
        }
        tokenized = [
            tokenize(article.text) + tokenize(article.title) for article in self.articles
        ]
        self.index = BM25Index(tokenized)

    # -- 元信息 ---------------------------------------------------------
    @property
    def backend(self) -> str:
        return self.index.backend

    def stats(self) -> dict[str, Any]:
        by_law: dict[str, int] = {}
        for article in self.articles:
            by_law[article.law_id] = by_law.get(article.law_id, 0) + 1
        return {
            "path": str(self.path),
            "total_articles": len(self.articles),
            "by_law": by_law,
            "bm25_backend": self.backend,
        }

    # -- 查询 -----------------------------------------------------------
    def get_article(self, law_id: str | None, article_no: str | int) -> Article | None:
        """按法律 + 条号取法条；``law_id`` 可为简称，``article_no`` 可为多种写法。"""
        resolved_law = resolve_law_id(law_id) if law_id else None
        if resolved_law is None:
            return None
        normalized = normalize_article_no(article_no)
        if normalized is None:
            return None
        return self._by_key.get((resolved_law, normalized))

    def search(
        self,
        query: str,
        top_k: int = 5,
        law_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 检索，返回按得分降序的法条列表（含 ``rank`` / ``score``）。"""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        resolved_law = resolve_law_id(law_id) if law_id else None
        if law_id and resolved_law is None:
            return []
        scores = self.index.get_scores(query_tokens)
        pairs: list[tuple[float, Article]] = [
            (scores[idx], article)
            for idx, article in enumerate(self.articles)
            if (resolved_law is None or article.law_id == resolved_law)
        ]
        # 得分降序；同分时按条号升序，保证结果稳定可复现。
        pairs.sort(key=lambda pair: (-pair[0], pair[1].law_id, pair[1].article_num))
        results: list[dict[str, Any]] = []
        for rank, (score, article) in enumerate(pairs, start=1):
            if score <= 0:
                continue
            item = article.to_dict()
            item["rank"] = rank
            item["score"] = round(score, 4)
            results.append(item)
            if len(results) >= top_k:
                break
        return results


# ---------------------------------------------------------------------------
# 模块级缓存与便捷函数
# ---------------------------------------------------------------------------

_corpus_lock = threading.Lock()
_corpus_cache: Corpus | None = None


def get_corpus(path: str | Path | None = None) -> Corpus:
    """获取（进程内缓存的）语料库实例。"""
    global _corpus_cache
    if path is not None:
        return Corpus(path)
    if _corpus_cache is None:
        with _corpus_lock:
            if _corpus_cache is None:
                _corpus_cache = Corpus()
    return _corpus_cache


def reset_cache() -> None:
    """清空语料缓存（测试用）。"""
    global _corpus_cache
    with _corpus_lock:
        _corpus_cache = None


def search_articles(
    query: str,
    top_k: int = 5,
    law_id: str | None = None,
) -> list[dict[str, Any]]:
    """便捷函数：检索法条，等价于 ``get_corpus().search(...)``。"""
    return get_corpus().search(query, top_k=top_k, law_id=law_id)


def get_article(law_id: str | None, article_no: str | int) -> Article | None:
    """便捷函数：按法律 + 条号取法条。"""
    return get_corpus().get_article(law_id, article_no)


def corpus_stats() -> dict[str, Any]:
    """便捷函数：语料库统计信息。"""
    return get_corpus().stats()
