"""语料加载与切分。

语料来源：shared/corpus/labor_law_articles.jsonl（357 条真实法条，见 shared/corpus/provenance.json）。

切分策略：
- 默认「一条法条 = 一个 chunk」，保证引用粒度是"条"，评测指标（MRR/Hit/Recall）也按条计算；
- 仅对超长法条（> MAX_ARTICLE_CHARS）做二次切分，切出来的子块仍映射回同一个 article_id，
  这样评测只需要看 article_id 是否命中，不会被切分方式干扰。
- 检索文本 = 法律名 + 章 + 条号 + 正文，让"标题+内容"一起参与 embedding。

LangChain 用于文本切分（RecursiveCharacterTextSplitter），语料本身用标准库加载，
避免把 LangChain 的 Document 结构泄漏到核心逻辑里。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from lrs.config import ARTICLES_JSONL
from lrs.schemas import Chunk, LegalArticle

logger = logging.getLogger(__name__)

MAX_ARTICLE_CHARS = 480       # 超过则二次切分（357 条里只有少数超长条文会触发）
CHUNK_OVERLAP = 40


def load_articles(path: Path | str = ARTICLES_JSONL) -> list[LegalArticle]:
    """读取 JSONL 法条语料，按 (law 顺序, 条号) 稳定排序。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到语料 {path}。请在项目根目录运行 `python data/corpus/parse_laws.py` 重新生成，"
            "或确认 data/corpus/labor_law_articles.jsonl 存在。"
        )
    law_order = ["lcl", "ll", "lda", "sil"]
    articles: list[LegalArticle] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:  # pragma: no cover - 数据损坏时才触发
                raise ValueError(f"{path}:{line_no} JSON 解析失败: {e}") from e
            articles.append(
                LegalArticle(
                    id=row["id"],
                    law_id=row["law_id"],
                    law=row["law"],
                    chapter=row.get("chapter", ""),
                    article_no=row["article_no"],
                    article_num=int(row["article_num"]),
                    title=row["title"],
                    text=row["text"],
                    paragraphs=row.get("paragraphs", []),
                )
            )
    articles.sort(key=lambda a: (law_order.index(a.law_id) if a.law_id in law_order else 99,
                                 a.article_num))
    logger.info("加载法条 %d 条，来自 %s", len(articles), path)
    return articles


_SHORT_LAW = re.compile(r"《(.+?)》")


def _context_header(article: LegalArticle) -> str:
    """检索文本的上下文头：法律简称 + 条号 + 章名（让"哪部法、哪一条、哪一章"参与检索）。"""
    chapter = article.chapter.split(" / ")[-1].strip() if article.chapter else ""
    parts = [f"{article.short_law}{article.article_no}"]
    if chapter:
        parts.append(chapter)
    return "，".join(parts)


def build_chunks(articles: list[LegalArticle], max_chars: int = MAX_ARTICLE_CHARS) -> list[Chunk]:
    """法条 -> 检索块。短法条整条一个块；超长法条按句子边界切分。"""
    splitter = _make_splitter(max_chars)
    chunks: list[Chunk] = []
    order = 0
    for art in articles:
        header = _context_header(art)
        body = art.text.strip()
        if len(body) <= max_chars or splitter is None:
            pieces = [body]
        else:
            pieces = [p.strip() for p in splitter.split_text(body) if p.strip()]
        for i, piece in enumerate(pieces):
            span = f"{i + 1}/{len(pieces)}" if len(pieces) > 1 else ""
            chunk_id = f"{art.id}#{i}" if span else art.id
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    article_id=art.id,
                    law=art.law,
                    article_no=art.article_no,
                    chapter=art.chapter,
                    text=f"{header}：{piece}",
                    article_text=piece,
                    span=span,
                    order=order,
                )
            )
            order += 1
    logger.info("切分 chunk %d 个（法条 %d 条）", len(chunks), len(articles))
    return chunks


def _make_splitter(max_chars: int):
    """优先用 LangChain 的递归切分器；未安装时退化为按句号切分。"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n", "。", "；", "，", ""],
            keep_separator="end",
        )
    except Exception:  # pragma: no cover - 依赖降级路径
        logger.warning("未安装 langchain_text_splitters，使用内置按句切分")

        class _SimpleSplitter:
            def split_text(self, text: str) -> list[str]:
                sentences = re.split(r"(?<=[。；])", text)
                out, buf = [], ""
                for s in sentences:
                    if len(buf) + len(s) > max_chars and buf:
                        out.append(buf)
                        buf = s
                    else:
                        buf += s
                if buf:
                    out.append(buf)
                return out

        return _SimpleSplitter()


def article_index(articles: list[LegalArticle]) -> dict[str, LegalArticle]:
    """id -> 法条，供引用回溯。"""
    return {a.id: a for a in articles}


def corpus_stats(articles: list[LegalArticle]) -> dict:
    """语料统计，写进 README 与评测报告。"""
    by_law: dict[str, int] = {}
    for a in articles:
        by_law[a.law] = by_law.get(a.law, 0) + 1
    lengths = [len(a.text) for a in articles]
    return {
        "articles": len(articles),
        "chars": sum(lengths),
        "avg_chars": round(sum(lengths) / max(len(lengths), 1), 1),
        "max_chars": max(lengths) if lengths else 0,
        "by_law": by_law,
    }
