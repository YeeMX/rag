"""3 个工具的独立测试：金额精确到分/元、边界条件、错误处理、引用核验。"""

from __future__ import annotations

import json

import pytest

from labor_agent import tools


# ---------------------------------------------------------------------------
# search_labor_law
# ---------------------------------------------------------------------------


def test_search_tool_returns_structured_results() -> None:
    result = tools.search_labor_law("违法解除劳动合同 赔偿金", top_k=3)
    assert result["ok"] is True
    assert result["tool"] == "search_labor_law"
    data = result["data"]
    assert data["count"] == 3
    assert data["query"] == "违法解除劳动合同 赔偿金"
    top = data["results"][0]
    assert top["title"] == "《劳动合同法》第八十七条"
    assert top["snippet"]  # 正文片段非空
    assert top["law_short"] == "劳动合同法"
    assert top["rank"] == 1


def test_search_tool_law_filter_and_no_match_hint() -> None:
    filtered = tools.search_labor_law("经济补偿", law_id="ll")
    assert filtered["ok"] is True
    assert all(item["law_id"] == "ll" for item in filtered["data"]["results"])

    empty = tools.search_labor_law("zzzzzz")
    assert empty["ok"] is True
    assert empty["data"]["count"] == 0
    assert "hint" in empty["data"]


@pytest.mark.parametrize(
    ("kwargs", "code", "field"),
    [
        ({"query": ""}, "invalid_arguments", "query"),
        ({"query": "   "}, "invalid_arguments", "query"),
        ({"query": "经济补偿", "top_k": 0}, "invalid_arguments", "top_k"),
        ({"query": "经济补偿", "top_k": 21}, "invalid_arguments", "top_k"),
        ({"query": "经济补偿", "top_k": "5"}, "invalid_arguments", "top_k"),
        ({"query": "经济补偿", "law_id": "刑法"}, "invalid_arguments", "law_id"),
    ],
)
def test_search_tool_rejects_invalid_arguments(kwargs: dict, code: str, field: str) -> None:
    result = tools.search_labor_law(**kwargs)
    assert result["ok"] is False
    assert result["error"]["code"] == code
    assert result["error"]["field"] == field


# ---------------------------------------------------------------------------
# calc_severance：经济补偿金 N / 违法解除 2N
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("years", "months", "expected_n_years", "expected_total"),
    [
        (3, 7, 4.0, 48000.0),   # 满 3 年 + 7 个月（>=6 按一年）→ 4 个月
        (3, 5, 3.5, 42000.0),   # 余 5 个月（<6 按半个月）→ 3.5 个月
        (3, 6, 4.0, 48000.0),   # 恰好 6 个月按一年
        (0, 6, 1.0, 12000.0),   # 不满一年但满 6 个月
        (0, 5, 0.5, 6000.0),    # 不满 6 个月按半个月
        (10, 0, 10.0, 120000.0),
    ],
)
def test_severance_n_rules(
    years: int, months: int, expected_n_years: float, expected_total: float
) -> None:
    result = tools.calc_severance(
        {"scenario": "severance", "monthly_salary": 12000, "years": years, "months": months}
    )
    assert result["ok"] is True
    data = result["data"]
    assert data["n_years"] == expected_n_years
    assert data["total"] == expected_total
    assert data["total_yuan"] == int(expected_total)


def test_severance_high_salary_cap() -> None:
    """月薪 > 社平 3 倍：基数按 3 倍、年限最高 12 年（第 47 条第二款）。"""
    result = tools.calc_severance(
        {
            "scenario": "severance",
            "monthly_salary": 30000,
            "years": 15,
            "months": 0,
            "city_avg_salary": 9000,
        }
    )
    assert result["ok"] is True
    data = result["data"]
    assert data["cap_applied"] is True
    assert data["base_salary"] == 27000.0
    assert data["n_years"] == 12.0
    assert data["total"] == 324000.0
    assert [step["step"] for step in data["steps"]] == [1, 2, 3]
    assert "12 年" in data["steps"][1]["formula"]


def test_severance_no_cap_when_below_three_times_average() -> None:
    result = tools.calc_severance(
        {
            "scenario": "severance",
            "monthly_salary": 20000,
            "years": 15,
            "months": 0,
            "city_avg_salary": 9000,
        }
    )
    data = result["data"]
    assert data["cap_applied"] is False
    assert data["base_salary"] == 20000.0
    assert data["n_years"] == 15.0
    assert data["total"] == 300000.0


def test_severance_without_city_average_is_marked_as_unbounded() -> None:
    result = tools.calc_severance(
        {"scenario": "severance", "monthly_salary": 30000, "years": 15}
    )
    data = result["data"]
    assert data["cap_applied"] is False
    assert data["city_avg_salary"] is None
    assert any("未提供 city_avg_salary" in note for note in data["notes"])


def test_unlawful_termination_is_double_severance() -> None:
    result = tools.calc_severance(
        {
            "scenario": "unlawful_termination",
            "monthly_salary": 12000,
            "years": 3,
            "months": 7,
        }
    )
    data = result["data"]
    assert data["severance_amount"] == 48000.0
    assert data["total"] == 96000.0
    assert data["total_yuan"] == 96000
    assert [step["step"] for step in data["steps"]] == [1, 2, 3, 4]
    assert {"《劳动合同法》第四十七条", "《劳动合同法》第八十七条"} <= {
        item["title"] for item in data["legal_basis"]
    }


# ---------------------------------------------------------------------------
# calc_severance：二倍工资差额
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("months_without_contract", "expected_capped", "expected_total"),
    [
        (5, 5, 75000.0),
        (11, 11, 165000.0),
        (12, 11, 165000.0),
        (14, 11, 165000.0),
        (0, 0, 0.0),
    ],
)
def test_double_wage_capped_at_eleven_months(
    months_without_contract: int, expected_capped: int, expected_total: float
) -> None:
    result = tools.calc_severance(
        {
            "scenario": "double_wage",
            "monthly_salary": 15000,
            "months_without_contract": months_without_contract,
        }
    )
    assert result["ok"] is True
    data = result["data"]
    assert data["capped_months"] == expected_capped
    assert data["total"] == expected_total
    assert data["months_capped"] == (months_without_contract > 11)


# ---------------------------------------------------------------------------
# calc_severance：加班费
# ---------------------------------------------------------------------------


def test_overtime_rates_and_amounts() -> None:
    """月薪 12000：小时工资 = 12000 / 21.75 / 8 ≈ 68.9655 元。"""
    result = tools.calc_severance(
        {
            "scenario": "overtime",
            "monthly_salary": 12000,
            "overtime_hours": {"workday": 10, "rest_day": 8, "holiday": 4},
        }
    )
    assert result["ok"] is True
    data = result["data"]
    assert data["hourly_wage"] == pytest.approx(68.9655, abs=0.0001)
    assert data["amount_breakdown"]["workday"] == 1034.48
    assert data["amount_breakdown"]["rest_day"] == 1103.45
    assert data["amount_breakdown"]["holiday"] == 827.59
    assert data["total"] == 2965.52
    assert data["total_yuan"] == 2966
    # 分档倍率写进计算过程，面试时可讲
    formulas = " ".join(step["formula"] for step in data["steps"])
    assert "1.5 倍" in formulas and "2 倍" in formulas and "3 倍" in formulas
    assert [step["step"] for step in data["steps"]] == [1, 2, 3, 4, 5]


def test_overtime_only_one_category() -> None:
    result = tools.calc_severance(
        {
            "scenario": "overtime",
            "monthly_salary": 12000,
            "overtime_hours": {"holiday": 2},
        }
    )
    data = result["data"]
    assert data["amount_breakdown"]["workday"] == 0.0
    assert data["amount_breakdown"]["rest_day"] == 0.0
    assert data["total"] == pytest.approx(2 * 3 * 68.965517, abs=0.02)


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        (
            {"scenario": "overtime", "monthly_salary": 12000},
            "overtime_hours",
        ),
        (
            {
                "scenario": "overtime",
                "monthly_salary": 12000,
                "overtime_hours": {"workday": 0, "rest_day": 0, "holiday": 0},
            },
            "overtime_hours",
        ),
        (
            {
                "scenario": "overtime",
                "monthly_salary": 12000,
                "overtime_hours": {"workday": -1},
            },
            "overtime_hours.workday",
        ),
        (
            {
                "scenario": "overtime",
                "monthly_salary": 12000,
                "overtime_hours": {"workday": True},
            },
            "overtime_hours.workday",
        ),
    ],
)
def test_overtime_invalid_inputs(payload: dict, field: str) -> None:
    result = tools.calc_severance(payload)
    assert result["ok"] is False
    assert result["error"]["field"] == field


# ---------------------------------------------------------------------------
# calc_severance：参数校验与未知字段
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"monthly_salary": 12000, "years": 3}, "scenario"),
        ({"scenario": "unknown", "monthly_salary": 12000}, "scenario"),
        ({"scenario": "severance", "years": 3}, "monthly_salary"),
        ({"scenario": "severance", "monthly_salary": 0, "years": 3}, "monthly_salary"),
        ({"scenario": "severance", "monthly_salary": -100, "years": 3}, "monthly_salary"),
        ({"scenario": "severance", "monthly_salary": 12000, "years": 0, "months": 0}, "years"),
        ({"scenario": "severance", "monthly_salary": 12000, "years": 1, "months": 12}, "months"),
        ({"scenario": "double_wage", "monthly_salary": 15000}, "months_without_contract"),
        ({"scenario": "severance", "monthly_salary": "12000", "years": 3}, "monthly_salary"),
    ],
)
def test_calc_rejects_invalid_payload(payload: dict, field: str) -> None:
    result = tools.calc_severance(payload)
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments"
    assert result["error"]["field"] == field


def test_calc_rejects_non_dict_payload() -> None:
    result = tools.calc_severance(["not", "a", "dict"])  # type: ignore[arg-type]
    assert result["ok"] is False
    assert result["error"]["field"] == "payload"


def test_calc_warns_on_unknown_fields() -> None:
    result = tools.calc_severance(
        {
            "scenario": "severance",
            "monthly_salary": 12000,
            "years": 3,
            "bonus": 5000,
        }
    )
    assert result["ok"] is True
    assert any("bonus" in warning for warning in result["data"]["warnings"])


# ---------------------------------------------------------------------------
# verify_citation
# ---------------------------------------------------------------------------


def test_verify_article_47_verified() -> None:
    claim = (
        "经济补偿按每满一年支付一个月工资，六个月以上不满一年按一年，"
        "不满六个月支付半个月工资；高收入者按社平三倍封顶，年限最高12年"
    )
    result = tools.verify_citation("lcl", "第四十七条", claim)
    assert result["ok"] is True
    data = result["data"]
    assert data["status"] == "verified"
    assert data["article_found"] is True
    assert data["checked"] >= 6
    assert data["matched"] == data["checked"]
    assert data["article"]["article_no"] == "第四十七条"


def test_verify_article_82_partial_due_to_11_months() -> None:
    result = tools.verify_citation("劳动合同法", "82", "未签书面合同应支付二倍工资，最多11个月")
    data = result["data"]
    assert data["status"] == "partial"
    assert data["article_found"] is True
    assert data["missing"]
    assert any("11个月" in reason for reason in data["missing"])
    # 关键词「二倍工资」应通过同义写法「二倍的工资」命中
    keyword_checks = {item["keyword"]: item["found"] for item in data["checks"]["keywords"]}
    assert keyword_checks.get("二倍工资") is True


def test_verify_wrong_article_is_not_found() -> None:
    """条号存在，但 claim 讲的内容与该条完全无关 → not_found（疑似张冠李戴）。"""
    result = tools.verify_citation("lcl", "第三十九条", "经济补偿按每满一年支付一个月工资，最高12年")
    data = result["data"]
    assert data["status"] == "not_found"
    assert data["article_found"] is True


def test_verify_nonexistent_article() -> None:
    result = tools.verify_citation("lcl", "第九百九十九条", "随便一句话")
    data = result["data"]
    assert data["status"] == "not_found"
    assert data["article_found"] is False
    assert result["ok"] is True  # 核验本身成功执行


def test_verify_invalid_law_or_article_no() -> None:
    invalid_law = tools.verify_citation("刑法", "第一条", "随便")
    assert invalid_law["data"]["status"] == "not_found"
    assert "无法识别的法律标识" in invalid_law["data"]["reasons"][0]

    invalid_no = tools.verify_citation("lcl", "abc", "随便")
    assert invalid_no["data"]["status"] == "not_found"
    assert "无法解析的条号" in invalid_no["data"]["reasons"][0]


def test_verify_rejects_empty_claim() -> None:
    result = tools.verify_citation("lcl", "第四十七条", "   ")
    assert result["ok"] is False
    assert result["error"]["field"] == "claim"


def test_verify_accepts_numeric_article_no() -> None:
    result = tools.verify_citation("lcl", 87, "违法解除应按经济补偿标准的二倍支付赔偿金")
    data = result["data"]
    assert data["article_no"] == "第八十七条"
    assert data["status"] == "verified"


# ---------------------------------------------------------------------------
# 数字要素抽取（verify_citation 的核心子逻辑）
# ---------------------------------------------------------------------------


def test_extract_numeric_claims_units_and_percentages() -> None:
    claims = tools.extract_numeric_claims("支付不低于工资的百分之一百五十，最多11个月，最高12年，半个月")
    keyed = {(item["value"], item["kind"]) for item in claims}
    assert (1.5, "ratio") in keyed      # 百分之一百五十 → 1.5 倍
    assert (11.0, "month") in keyed
    assert (12.0, "year") in keyed
    assert (0.5, "month") in keyed
    percent = tools.extract_numeric_claims("150%")
    assert percent == [{"raw": "150%", "value": 1.5, "kind": "ratio"}]


def test_extract_numeric_claims_masks_article_reference() -> None:
    assert tools.extract_numeric_claims("第三十九条") == []
    claims = tools.extract_numeric_claims("根据第四十七条，补偿12年")
    assert all(item["value"] != 47 for item in claims)


# ---------------------------------------------------------------------------
# 工具注册表 / JSON Schema / 统一调用入口
# ---------------------------------------------------------------------------


def test_tool_schemas_are_openai_function_format() -> None:
    names = [schema["function"]["name"] for schema in tools.TOOL_SCHEMAS]
    assert names == ["search_labor_law", "calc_severance", "verify_citation"]
    for schema in tools.TOOL_SCHEMAS:
        function = schema["function"]
        assert schema["type"] == "function"
        assert function["description"]
        assert function["parameters"]["type"] == "object"
        assert function["parameters"]["required"]
    calc_schema = tools.TOOL_NAME_TO_SCHEMA["calc_severance"]["function"]["parameters"]
    assert set(calc_schema["properties"]["scenario"]["enum"]) == set(tools.SEVERANCE_SCENARIOS)


def test_execute_tool_call_dispatches_to_impl() -> None:
    result = tools.execute_tool_call(
        "calc_severance",
        json.dumps(
            {
                "scenario": "unlawful_termination",
                "monthly_salary": 12000,
                "years": 3,
                "months": 7,
            }
        ),
    )
    assert result["ok"] is True
    assert result["data"]["total"] == 96000.0


def test_execute_tool_call_unknown_tool() -> None:
    result = tools.execute_tool_call("delete_database", "{}")
    assert result["ok"] is False
    assert result["error"]["code"] == "unknown_tool"
    assert "search_labor_law" in result["error"]["message"]


def test_execute_tool_call_bad_json() -> None:
    result = tools.execute_tool_call("search_labor_law", '{"query": "经济补偿", 坏}')
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments_json"


def test_execute_tool_call_non_object_json() -> None:
    result = tools.execute_tool_call("search_labor_law", '["经济补偿"]')
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments"


def test_execute_tool_call_missing_required_argument() -> None:
    result = tools.execute_tool_call("search_labor_law", "{}")
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments"


def test_execute_tool_call_accepts_dict_arguments() -> None:
    result = tools.execute_tool_call(
        "verify_citation",
        {"law_id": "lcl", "article_no": "47", "claim": "经济补偿每满一年支付一个月工资"},
    )
    assert result["ok"] is True
    assert result["data"]["status"] == "verified"
