"""核心数据结构。

整个 pipeline 只依赖这里的 dataclass，方便单独测试每一层、也方便评测时做消融。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LegalArticle:
    """一条法条（对应 shared/corpus/labor_law_articles.jsonl 的一行）。"""

    id: str                 # 全局唯一，如 lcl:第三十九条
    law_id: str             # lcl / ll / lda / sil
    law: str                # 法律全称
    chapter: str            # 章/节
    article_no: str         # 第X条
    article_num: int        # 阿拉伯数字，便于按条号排序
    title: str              # 《劳动合同法》第三十九条
    text: str
    paragraphs: list[str] = field(default_factory=list)

    @property
    def short_law(self) -> str:
        """《中华人民共和国劳动合同法》-> 劳动合同法"""
        return self.law.replace("中华人民共和国", "")


@dataclass
class Chunk:
    """检索单元。默认 1 条法条 = 1 个 chunk；超长法条按字数切分，但保留同一 article_id。"""

    chunk_id: str
    article_id: str
    law: str
    article_no: str
    chapter: str
    text: str               # 带上下文的检索文本（含法律名+条号）
    article_text: str       # 纯法条正文
    span: str = ""           # 超长切分时的第几段，如 "1/2"
    order: int = 0


@dataclass
class Retrieved:
    """一条检索结果及其在各阶段的分数轨迹（用于可解释性与评测）。"""

    chunk_id: str
    article_id: str
    law: str
    article_no: str
    title: str
    text: str
    rank: int = 0
    scores: dict[str, float | str] = field(default_factory=dict)   # {"dense": .7, "rrf": .01, "rerank": .2, "filter": 1.0, "drop_reason": "..."}
    source: str = "hybrid"      # 该结果来自哪一路召回：dense / bm25 / both
    kept: bool = True           # LLM 过滤后是否保留


@dataclass
class QueryPlan:
    """查询理解结果：原问题 -> 子查询 + 扩展词 + 复合查询。"""

    original: str
    sub_queries: list[str]
    expansions: list[str]
    compound_queries: list[str] = field(default_factory=list)
    is_multi_intent: bool = False
    note: str = ""

    @property
    def all_queries(self) -> list[str]:
        return [self.original, *self.sub_queries, *self.compound_queries]


@dataclass
class Citation:
    """答案里的一条引用，必须能回溯到真实法条。"""

    article_id: str
    title: str
    quote: str
    reason: str = ""


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[Retrieved]
    plan: QueryPlan
    llm_mode: str = "mock"       # mock / openai
    latency_ms: dict[str, float] = field(default_factory=dict)

    def top_article_ids(self, k: int) -> list[str]:
        return [r.article_id for r in self.retrieved if r.kept][:k]
