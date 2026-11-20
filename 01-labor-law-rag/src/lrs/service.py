"""索引构建与服务装配。

把「加载配置 → 建 embedder → 建 BM25 → 建/读 Chroma → 装配 retriever + reranker + LLM」
收敛到一个入口，CLI、评测、Chainlit 都复用它，避免多套初始化逻辑。
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

from lrs.config import INDEX_DIR, Settings, load_settings
from lrs.corpus import build_chunks, load_articles
from lrs.embeddings import Embedder, build_embedder
from lrs.filter import filter_candidates
from lrs.lexical import BM25Index
from lrs.llm import LLM, build_llm
from lrs.reranker import build_reranker, rerank as do_rerank
from lrs.retriever import HybridRetriever
from lrs.schemas import Citation, QueryPlan, RAGAnswer, Retrieved
from lrs.vector_store import ChromaStore

logger = logging.getLogger(__name__)

BM25_CACHE = INDEX_DIR / "bm25_chunks.pkl"


@dataclass
class RAGService:
    """装配好的检索问答服务。"""

    settings: Settings
    chunks: list
    bm25: BM25Index
    embedder: Embedder
    store: ChromaStore
    retriever: HybridRetriever
    reranker: object
    llm: LLM

    # ------------------------------------------------------------------ 检索
    def retrieve(self, plan: QueryPlan, *, top_k: int | None = None,
                 use_decompose: bool | None = None, use_expand: bool | None = None
                 ) -> list[Retrieved]:
        s = self.settings
        # 检索通道开关跟随 settings（消融评测需要）
        self.retriever.dense_enabled = s.dense_enabled
        self.retriever.bm25_enabled = s.bm25_enabled
        return self.retriever.retrieve(
            plan,
            top_k=top_k or s.top_k,
            use_decompose=s.decompose_enabled if use_decompose is None else use_decompose,
            use_expand=s.expand_enabled if use_expand is None else use_expand,
        )

    # ------------------------------------------------------------------ 全链路
    def answer(self, question: str, *, top_k: int | None = None,
               final_k: int | None = None) -> RAGAnswer:
        s = self.settings
        from lrs.query_processing import plan_query

        t0 = time.perf_counter()
        plan = plan_query(question, decompose=s.decompose_enabled, expand=s.expand_enabled,
                          max_sub_queries=s.max_sub_queries)
        t1 = time.perf_counter()

        candidates = self.retrieve(plan, top_k=top_k or s.top_k)
        t2 = time.perf_counter()

        if s.rerank_enabled and candidates:
            # 重排阶段多留一些候选（rerank_candidates），过滤之后再截断到 final_k，
            # 这样零样本过滤误删时还有机会从后续候选里补足。
            reranked = do_rerank(question, candidates, self.reranker,
                                 top_k=s.rerank_candidates)
        else:
            reranked = candidates
        t3 = time.perf_counter()

        if s.filter_enabled and reranked:
            filtered = filter_candidates(question, reranked, self.llm)
        else:
            filtered = reranked
        filtered = filtered[: final_k or s.final_k]
        t4 = time.perf_counter()

        answer_text, citations = self._generate(question, filtered)
        t5 = time.perf_counter()

        return RAGAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved=reranked,
            plan=plan,
            llm_mode=self.llm.mode,
            latency_ms={
                "plan": round((t1 - t0) * 1000, 1),
                "retrieve": round((t2 - t1) * 1000, 1),
                "rerank": round((t3 - t2) * 1000, 1),
                "filter": round((t4 - t3) * 1000, 1),
                "generate": round((t5 - t4) * 1000, 1),
                "total": round((t5 - t0) * 1000, 1),
            },
        )

    # ------------------------------------------------------------------ 生成
    def _generate(self, question: str, context: list[Retrieved]) -> tuple[str, list[Citation]]:
        from lrs.prompts import ANSWER_PROMPT, format_context

        if not context:
            return ("没有检索到相关法条，请补充关键事实（如入职时间、是否签合同、"
                    "解除方式、工资标准等）后重试。", [])
        prompt = ANSWER_PROMPT.format(context=format_context(context), question=question)
        try:
            text = self.llm.complete(prompt, task="answer")
        except Exception as e:  # pragma: no cover
            logger.warning("生成失败，改用抽取式兜底: %s", e)
            text = "（生成服务不可用）" + "；".join(f"{c.title}：{c.text}" for c in context[:3])

        # 引用回溯：答案中出现的条号必须能在上下文里找到，找不到就不进 citations
        import re

        found: list[Citation] = []
        for c in context:
            short_law = c.law.replace("中华人民共和国", "")
            pattern = re.compile(rf"《?{re.escape(short_law)}》?\s*{re.escape(c.article_no)}")
            if pattern.search(text) or c.article_no in text:
                found.append(Citation(article_id=c.article_id, title=c.title,
                                      quote=c.text[:80] + ("…" if len(c.text) > 80 else "")))
        if not found:  # 模型没写条号时，把注入的材料都作为"参考法条"列出，避免空引用
            found = [Citation(article_id=c.article_id, title=c.title,
                              quote=c.text[:80] + ("…" if len(c.text) > 80 else ""))
                     for c in context[:3]]
        return text, found

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def load_or_build(settings: Settings | None = None, *, force_rebuild: bool = False,
                      prefer_real_llm: bool = False, need_reranker: bool = True
                      ) -> "RAGService":
        s = settings or load_settings()
        articles = load_articles()
        chunks = build_chunks(articles)

        embedder = build_embedder(s.dense_model, device=s.device,
                                  batch_size=s.dense_batch_size,
                                  allow_download=s.allow_model_download)
        store = ChromaStore(embedder, collection=s.collection_name)
        if store.is_empty() or force_rebuild:
            store.build(chunks, force=force_rebuild)
        bm25 = _load_or_build_bm25(chunks, force=force_rebuild)
        retriever = HybridRetriever(chunks, embedder, bm25, store, rrf_k=s.rrf_k,
                                    dense_weight=s.dense_weight, bm25_weight=s.bm25_weight)
        reranker = build_reranker(s.rerank_model, device=s.device,
                                  allow_download=s.allow_model_download) if need_reranker else None
        llm = build_llm(prefer_real=prefer_real_llm)
        return RAGService(settings=s, chunks=chunks, bm25=bm25, embedder=embedder,
                          store=store, retriever=retriever, reranker=reranker, llm=llm)


def _load_or_build_bm25(chunks, force: bool = False) -> BM25Index:
    """BM25 是纯内存结构，缓存到磁盘省去每次重建（357 条其实很快，主要为了评测多次复用）。"""
    if not force and BM25_CACHE.exists():
        try:
            with BM25_CACHE.open("rb") as f:
                cached = pickle.load(f)
            if getattr(cached, "chunks", None) and len(cached.chunks) == len(chunks):
                return cached
        except Exception:  # pragma: no cover - 缓存损坏
            logger.warning("BM25 缓存损坏，重建")
    t0 = time.perf_counter()
    bm25 = BM25Index(chunks)
    BM25_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with BM25_CACHE.open("wb") as f:
        pickle.dump(bm25, f)
    logger.info("BM25 构建完成（%d 条，%.2fs，backend=%s）",
                len(chunks), time.perf_counter() - t0, getattr(bm25, "_backend", "?"))
    return bm25
