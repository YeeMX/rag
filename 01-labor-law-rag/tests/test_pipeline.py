"""集成测试：语料完整性 + 检索链路（用 HashEmbedder + 临时 Chroma，完全离线）。

真实 BGE 模型的端到端测试用 `pytest -m integration` 单独跑（见 test_integration_bge.py）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from lrs.corpus import build_chunks, corpus_stats, load_articles
from lrs.embeddings import HashEmbedder
from lrs.lexical import BM25Index
from lrs.llm import MockLLM
from lrs.query_processing import plan_query
from lrs.retriever import HybridRetriever
from lrs.vector_store import ChromaStore


@pytest.fixture(scope="module")
def corpus():
    return load_articles()


def test_corpus_loads_all_four_laws(corpus):
    assert len(corpus) == 357
    stats = corpus_stats(corpus)
    assert set(stats["by_law"]) == {
        "中华人民共和国劳动合同法", "中华人民共和国劳动法",
        "中华人民共和国劳动争议调解仲裁法", "中华人民共和国社会保险法",
    }
    assert stats["by_law"]["中华人民共和国劳动合同法"] == 98


def test_corpus_article_numbers_contiguous(corpus):
    from collections import defaultdict

    by_law: dict[str, list[int]] = defaultdict(list)
    for a in corpus:
        by_law[a.law_id].append(a.article_num)
    for law_id, nums in by_law.items():
        assert nums == list(range(1, len(nums) + 1)), f"{law_id} 条号不连续"


def test_corpus_texts_are_non_empty(corpus):
    assert all(len(a.text) >= 4 for a in corpus)
    assert all(a.title.startswith("《") for a in corpus)


def test_key_articles_have_expected_keywords(corpus):
    """抽查几条关键法条的内容，确保语料没有错位（防数据管道 bug）。"""
    by_id = {a.id: a for a in corpus}
    assert "二倍" in by_id["lcl:第八十七条"].text          # 违法解除赔偿金 2N
    assert "经济补偿" in by_id["lcl:第四十七条"].text      # 经济补偿计算
    assert "试用期" in by_id["lcl:第十九条"].text
    assert "加班" in by_id["ll:第四十四条"].text or "延长工作时间" in by_id["ll:第四十四条"].text
    assert "仲裁时效" in by_id["lda:第二十七条"].text or "一年" in by_id["lda:第二十七条"].text


@pytest.fixture(scope="module")
def retriever(corpus, tmp_path_factory):
    """Windows 上 Chroma 的文件句柄释放较慢，模块级 fixture 用 tmp_path_factory 且不主动删目录。"""
    chunks = build_chunks(corpus)
    embedder = HashEmbedder()
    tmp = tmp_path_factory.mktemp("chroma-test")
    store = ChromaStore(embedder, persist_dir=tmp, collection="test")
    store.build(chunks)
    bm25 = BM25Index(chunks)
    yield HybridRetriever(chunks, embedder, bm25, store, rrf_k=60)


def test_hybrid_retrieval_smoke(retriever):
    plan = plan_query("公司违法辞退我，经济补偿和赔偿金怎么算？")
    results = retriever.retrieve(plan, top_k=10)
    assert results, "混合检索不应返回空"
    assert all(r.article_id for r in results)
    assert all(0.0 <= float(r.scores["rrf"]) <= 1.0 for r in results)


def test_rrf_fusion_combines_sources(retriever):
    plan = plan_query("未签书面劳动合同的双倍工资")
    results = retriever.retrieve(plan, top_k=10, use_decompose=False, use_expand=False)
    sources = {r.source for r in results}
    # 至少有结果来自 dense 或 bm25；两路建了索引，应能出现 both 或单路
    assert sources & {"dense", "bm25", "both"}


def test_chunks_preserve_article_mapping(corpus):
    chunks = build_chunks(corpus)
    article_ids = {a.id for a in corpus}
    assert all(c.article_id in article_ids for c in chunks)
    # 长条文应被切分且仍映射回同一法条
    multi = [c for c in chunks if c.span]
    assert all(c.article_id in article_ids for c in multi)
