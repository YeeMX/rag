"""法条语料加载、条号规范化与 BM25 检索测试。"""

from __future__ import annotations

import pytest

from labor_agent import corpus
from labor_agent.corpus import (
    Article,
    chinese_to_int,
    int_to_chinese,
    normalize_article_no,
    resolve_law_id,
    tokenize,
)


def test_corpus_loads_all_357_articles() -> None:
    stats = corpus.corpus_stats()
    assert stats["total_articles"] == 357
    assert stats["by_law"] == {"lcl": 98, "ll": 107, "lda": 54, "sil": 98}
    assert stats["bm25_backend"] in {"rank_bm25", "stdlib"}


def test_get_article_by_article_number_variants() -> None:
    article = corpus.get_article("lcl", "47")
    assert isinstance(article, Article)
    assert article.article_no == "第四十七条"
    assert "每满一年支付一个月工资" in article.text
    # 中文法律全名、简称、条号中文写法都应能定位
    assert corpus.get_article("中华人民共和国劳动合同法", "第四十七条") is not None
    assert corpus.get_article("劳动合同法", "第47条") is not None
    assert corpus.get_article("lcl", "999") is None


def test_normalize_article_no_variants() -> None:
    for raw in ("47", "第47条", "第四十七条", "四十七", 47):
        assert normalize_article_no(raw) == "第四十七条", raw
    assert normalize_article_no("第一百零七条") == "第一百零七条"
    assert normalize_article_no("abc") is None
    assert normalize_article_no("0") is None


def test_chinese_numeral_roundtrip() -> None:
    for number in range(1, 108):
        assert chinese_to_int(int_to_chinese(number)) == number
    assert int_to_chinese(12) == "十二"
    assert int_to_chinese(20) == "二十"
    assert int_to_chinese(107) == "一百零七"


def test_resolve_law_id_prefers_longer_name() -> None:
    # 「劳动法」是「劳动合同法」的子串，必须优先匹配更长的名称
    assert resolve_law_id("劳动合同法") == "lcl"
    assert resolve_law_id("中华人民共和国劳动合同法") == "lcl"
    assert resolve_law_id("劳动法") == "ll"
    assert resolve_law_id("社会保险法") == "sil"
    assert resolve_law_id("劳动争议调解仲裁法") == "lda"
    assert resolve_law_id("lcl") == "lcl"
    assert resolve_law_id("刑法") is None


def test_tokenize_keeps_cjk_bigrams() -> None:
    tokens = tokenize("经济补偿")
    assert "经济" in tokens
    assert "补偿" in tokens
    assert "经" in tokens  # 单字用于保证召回


def test_search_returns_relevant_article_first() -> None:
    results = corpus.search_articles("违法解除劳动合同 赔偿金", top_k=3)
    assert results
    assert results[0]["title"] == "《劳动合同法》第八十七条"
    for item in results:
        assert {"rank", "score", "article_no", "law", "title", "text"} <= set(item)
        assert item["score"] > 0
    assert [item["rank"] for item in results] == list(range(1, len(results) + 1))


@pytest.mark.parametrize(
    ("query", "expected_title"),
    [
        ("未签书面劳动合同 二倍工资", "《劳动合同法》第八十二条"),
        ("经济补偿 工作年限 每满一年", "《劳动合同法》第四十七条"),
        ("加班 延长工作时间 休息日 法定休假日", "《劳动法》第四十四条"),
    ],
)
def test_search_expected_top_result(query: str, expected_title: str) -> None:
    results = corpus.search_articles(query, top_k=3)
    assert results[0]["title"] == expected_title


def test_search_law_filter_and_top_k() -> None:
    results = corpus.search_articles("解除劳动合同 经济补偿", top_k=4, law_id="ll")
    assert results
    assert len(results) <= 4
    assert all(item["law_id"] == "ll" for item in results)
    assert corpus.search_articles("经济补偿", law_id="not-a-law") == []


def test_search_empty_query_returns_empty_list() -> None:
    assert corpus.search_articles("   ", top_k=3) == []


def test_search_no_match_returns_empty_list() -> None:
    assert corpus.search_articles("zzzzzz", top_k=3) == []
