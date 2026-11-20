"""检索组件测试：BM25 分词、RRF 融合、零样本过滤解析。"""

from __future__ import annotations

from lrs.filter import _parse_json_loose, filter_candidates
from lrs.lexical import BM25Index, tokenize
from lrs.llm import MockLLM
from lrs.retriever import HybridRetriever
from lrs.schemas import Retrieved


# ---------------------------------------------------------------- BM25
def test_tokenize_mixes_unigram_and_bigram():
    toks = tokenize("经济补偿")
    assert "经" in toks and "济补" in toks and "补偿" in toks


def test_tokenize_drops_punctuation():
    toks = tokenize("公司，违法；解除！")
    assert "，" not in toks and "；" not in toks


def test_bm25_ranks_specific_clause_first(tiny_chunks):
    bm25 = BM25Index(tiny_chunks)
    hits = bm25.search("未签书面劳动合同的二倍工资", top_k=4)
    assert hits, "BM25 应该有结果"
    top_idx, top_score = hits[0]
    assert tiny_chunks[top_idx].article_id.endswith("第82条")
    assert top_score > 0


def test_bm25_empty_query_returns_nothing(tiny_chunks):
    bm25 = BM25Index(tiny_chunks)
    assert bm25.search("   ", top_k=3) == []


# ---------------------------------------------------------------- RRF
def test_rrf_rewards_documents_in_both_lists():
    dense = ["a", "b", "c"]
    lex = ["c", "a", "d"]
    fused = HybridRetriever.rrf_fuse([dense, lex], k=60)
    assert fused["a"] > fused["b"]
    assert fused["c"] > fused["b"]
    # "d" 只出现在一路，分数低于两路都有的
    assert fused["d"] < fused["a"]


def test_rrf_weights_bias(monkeypatch):
    dense = ["a"]
    lex = ["b"]
    fused = HybridRetriever.rrf_fuse([dense, lex], k=60, weights=[2.0, 1.0])
    assert fused["a"] > fused["b"]


# ---------------------------------------------------------------- 过滤
def test_parse_json_loose_handles_markdown_fence():
    raw = '```json\n{"keep": [{"id": 1, "reason": "ok"}], "drop": []}\n```'
    parsed = _parse_json_loose(raw)
    assert parsed["keep"][0]["id"] == 1


def test_parse_json_loose_handles_trailing_comma():
    raw = '{"keep": [{"id": 1,}],}'
    parsed = _parse_json_loose(raw)
    assert parsed is not None


def test_parse_json_loose_returns_none_for_garbage():
    assert _parse_json_loose("我不知道") is None


def _dummy_retrieved(article_id: str, text: str, rerank: float = 0.0) -> Retrieved:
    return Retrieved(chunk_id=article_id, article_id=article_id, law="中华人民共和国劳动合同法",
                     article_no="第1条", title="《劳动合同法》第1条", text=text, rank=1,
                     scores={"rerank": rerank})


def test_filter_keeps_relevant_drops_irrelevant():
    cands = [
        _dummy_retrieved("lcl:第八十七条", "用人单位违法解除劳动合同，应当支付二倍经济补偿标准的赔偿金。",
                         rerank=0.9),
        _dummy_retrieved("sil:第三十八条", "工伤职工治疗工伤的医疗费用，符合规定的从工伤保险基金支付。",
                         rerank=0.2),
    ]
    # top_guaranteed=1：只保证精排第一名，第二名交给 LLM 判断 -> 应被剔除
    kept = filter_candidates("公司违法辞退我，赔偿怎么算？", cands, MockLLM(),
                             min_keep=1, top_guaranteed=1)
    kept_ids = [c.article_id for c in kept]
    assert "lcl:第八十七条" in kept_ids
    assert "sil:第三十八条" not in kept_ids
    assert cands[1].kept is False


def test_filter_guarantees_top_reranked():
    """即使 LLM 全部判错，精排 top-N 也必须保留（防误删 gold）。"""
    class DropAllLLM:
        available = True
        mode = "mock"
        model = "mock"

        def complete(self, prompt, *, task="generic", max_tokens=None, temperature=None):
            return '{"keep": [], "drop": [{"id": 1, "reason": "x"}, {"id": 2, "reason": "y"}]}'

    cands = [
        _dummy_retrieved("lcl:第八十七条", "违法解除赔偿金。", rerank=0.9),
        _dummy_retrieved("lcl:第三十九条", "过失性辞退。", rerank=0.5),
    ]
    kept = filter_candidates("公司违法辞退我", cands, DropAllLLM(), min_keep=2, top_guaranteed=2)
    assert len(kept) == 2


def test_filter_fail_open_on_bad_json():
    class BadLLM:
        available = True
        mode = "bad"
        model = "bad"

        def complete(self, prompt, *, task="generic", max_tokens=None, temperature=None):
            return "抱歉，我无法完成。"

    cands = [_dummy_retrieved("lcl:第四十七条", "经济补偿按工作年限计算。")]
    kept = filter_candidates("经济补偿怎么算？", cands, BadLLM(), min_keep=1)
    assert kept == cands  # 解析失败 => 全部保留


def test_filter_respects_min_keep():
    class DropAllLLM:
        available = True
        mode = "mock"
        model = "mock"

        def complete(self, prompt, *, task="generic", max_tokens=None, temperature=None):
            return '{"keep": [], "drop": [{"id": 1, "reason": "x"}]}'

    cands = [_dummy_retrieved("lcl:第四十七条", "经济补偿按工作年限计算。")]
    kept = filter_candidates("经济补偿怎么算？", cands, DropAllLLM(), min_keep=1)
    assert len(kept) == 1
