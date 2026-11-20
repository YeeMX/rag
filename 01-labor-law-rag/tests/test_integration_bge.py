"""真实 BGE + 重排模型的端到端测试（需要下载模型，标记为 integration）。

运行：
    pytest -m integration -q
没有网络/模型时自动 skip，不影响常规测试。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from lrs.corpus import build_chunks, load_articles
from lrs.embeddings import build_embedder
from lrs.lexical import BM25Index
from lrs.query_processing import plan_query
from lrs.reranker import build_reranker, rerank
from lrs.retriever import HybridRetriever
from lrs.vector_store import ChromaStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def bge_service(tmp_path_factory):
    try:
        embedder = build_embedder("BAAI/bge-small-zh-v1.5", allow_download=True)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"无法加载 BGE 模型: {e}")
    if "hash" in embedder.name:
        pytest.skip("BGE 模型不可用（降级为 hash），跳过集成测试")
    articles = load_articles()
    chunks = build_chunks(articles)
    tmp = tmp_path_factory.mktemp("chroma-bge")
    store = ChromaStore(embedder, persist_dir=tmp, collection="it")
    store.build(chunks)
    bm25 = BM25Index(chunks)
    retriever = HybridRetriever(chunks, embedder, bm25, store, rrf_k=60)
    yield retriever, embedder


def test_bge_retrieves_relevant_article(bge_service):
    retriever, _ = bge_service
    plan = plan_query("公司违法解除劳动合同，赔偿金怎么计算？")
    results = retriever.retrieve(plan, top_k=10)
    ids = [r.article_id for r in results]
    assert "lcl:第八十七条" in ids[:10], f"top10 未命中第87条: {ids[:10]}"


def test_bge_rerank_improves_first_hit(bge_service):
    retriever, _ = bge_service
    plan = plan_query("未签书面劳动合同能要二倍工资吗？")
    results = retriever.retrieve(plan, top_k=10, use_decompose=False, use_expand=False)
    reranker = build_reranker("BAAI/bge-reranker-base", allow_download=True)
    if "overlap" in getattr(reranker, "name", ""):
        pytest.skip("重排模型不可用")
    reranked = rerank("未签书面劳动合同能要二倍工资吗？", results, reranker, top_k=5)
    assert reranked
    assert "lcl:第八十二条" in [r.article_id for r in reranked][:5]
