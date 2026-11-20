"""Agent 可调用的 3 个工具：法条检索、赔偿计算、引用核验。

约定（三种实现共用，面试可讲）：

- 每个工具都有 **OpenAI function calling 格式的 JSON Schema**（``TOOL_SCHEMAS``）；
- 每个工具都是 **独立可测的普通 Python 函数**，不依赖 LLM、不联网；
- 每个工具都返回 **统一结构化信封**：

  ``{"ok": bool, "tool": str, "data": dict | None, "error": {"code", "message", ...} | None}``

- 通过 :func:`execute_tool_call` 统一入口调用：负责解析 JSON 参数、拒绝未知工具、
  把参数错误/内部异常转成结构化错误，保证 Agent 主循环不会因为工具失败而崩溃。

工具返回的金额单位均为「元」，保留两位小数；``total_yuan`` 为四舍五入到元的整数。
"""

from __future__ import annotations

import json
import logging
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable

from . import corpus

_LOGGER = logging.getLogger(__name__)

ToolImpl = Callable[..., dict[str, Any]]


# ---------------------------------------------------------------------------
# 统一返回信封与错误类型
# ---------------------------------------------------------------------------


class ToolInputError(ValueError):
    """工具参数不合法。会被转换成结构化错误，而不是让 Agent 循环崩溃。"""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_arguments",
        field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.field = field


def ok_result(tool: str, data: dict[str, Any]) -> dict[str, Any]:
    """构造成功返回。"""
    return {"ok": True, "tool": tool, "data": data, "error": None}


def error_result(
    tool: str,
    code: str,
    message: str,
    *,
    field: str | None = None,
) -> dict[str, Any]:
    """构造结构化错误返回。"""
    error: dict[str, Any] = {"code": code, "message": message}
    if field:
        error["field"] = field
    return {"ok": False, "tool": tool, "data": None, "error": error}


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

_TWO_PLACES = Decimal("0.01")
_ONE_YUAN = Decimal("1")


def _money(value: float) -> float:
    """金额四舍五入到分（ROUND_HALF_UP，避免 Python round 的银行家舍入）。"""
    if not math.isfinite(value):
        raise ToolInputError("计算结果不是有限数值，请检查输入", code="invalid_arguments")
    return float(Decimal(str(value)).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP))


def _to_yuan(value: float) -> int:
    """金额四舍五入到元。"""
    return int(Decimal(str(value)).quantize(_ONE_YUAN, rounding=ROUND_HALF_UP))


def _fmt_num(value: float) -> str:
    """数字排版：4 -> '4'，3.5 -> '3.5'，12000 -> '12000'。"""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _fmt_money(value: float) -> str:
    return f"{value:.2f}"


def _snippet(text: str, query: str, width: int = 96) -> str:
    """截取包含查询关键词的正文片段，便于直接展示给用户。"""
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return ""
    if len(clean) <= width:
        return clean
    anchor = -1
    for token in re.findall(r"[\u4e00-\u9fff]{2,}", query or ""):
        anchor = clean.find(token)
        if anchor >= 0:
            break
    if anchor < 0:
        return clean[:width] + "…"
    start = max(0, anchor - 24)
    piece = clean[start : start + width]
    prefix = "…" if start > 0 else ""
    suffix = "…" if start + width < len(clean) else ""
    return f"{prefix}{piece}{suffix}"


def _require_number(
    payload: dict[str, Any],
    key: str,
    label: str,
    *,
    gt0: bool = False,
    maximum: float | None = None,
) -> float:
    if key not in payload or payload[key] is None:
        raise ToolInputError(f"缺少必填参数 {key}（{label}）", field=key)
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolInputError(
            f"参数 {key}（{label}）必须是数字，收到 {type(value).__name__}: {value!r}",
            field=key,
        )
    number = float(value)
    if not math.isfinite(number):
        raise ToolInputError(f"参数 {key}（{label}）必须是有限数字", field=key)
    if gt0 and number <= 0:
        raise ToolInputError(f"参数 {key}（{label}）必须大于 0，收到 {number}", field=key)
    if not gt0 and number < 0:
        raise ToolInputError(f"参数 {key}（{label}）不能为负数，收到 {number}", field=key)
    if maximum is not None and number > maximum:
        raise ToolInputError(
            f"参数 {key}（{label}）超过合理上限 {maximum:g}，请检查输入",
            field=key,
        )
    return number


def _optional_number(
    payload: dict[str, Any],
    key: str,
    label: str,
    *,
    gt0: bool = False,
    maximum: float | None = None,
) -> float | None:
    if key not in payload or payload[key] is None:
        return None
    return _require_number(payload, key, label, gt0=gt0, maximum=maximum)


def _require_int(
    payload: dict[str, Any],
    key: str,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if key not in payload or payload[key] is None:
        raise ToolInputError(f"缺少必填参数 {key}（{label}）", field=key)
    value = payload[key]
    if isinstance(value, bool):
        raise ToolInputError(f"参数 {key}（{label}）必须是整数", field=key)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise ToolInputError(
            f"参数 {key}（{label}）必须是整数，收到 {type(value).__name__}: {value!r}",
            field=key,
        )
    if value < minimum:
        raise ToolInputError(f"参数 {key}（{label}）不能小于 {minimum}，收到 {value}", field=key)
    if maximum is not None and value > maximum:
        raise ToolInputError(
            f"参数 {key}（{label}）不能大于 {maximum}，收到 {value}", field=key
        )
    return value


def _optional_int(
    payload: dict[str, Any],
    key: str,
    label: str,
    *,
    default: int = 0,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    """可选整数字段：缺省返回 ``default``，出现则按整数规则校验。"""
    if key not in payload or payload[key] is None:
        return default
    return _require_int(payload, key, label, minimum=minimum, maximum=maximum)


def _legal_basis(*refs: tuple[str, str]) -> list[dict[str, Any]]:
    """从真实语料中取法条原文，作为计算/核验结果的依据（不手写、不编造）。"""
    basis: list[dict[str, Any]] = []
    for law_id, article_no in refs:
        article = corpus.get_article(law_id, article_no)
        if article is None:  # 语料缺失时静默跳过，不影响主功能
            _LOGGER.warning("法条语料中未找到 %s %s", law_id, article_no)
            continue
        basis.append(
            {
                "law_id": law_id,
                "law": article.law,
                "article_no": article.article_no,
                "title": article.title,
                "text": article.text,
            }
        )
    return basis


_DISCLAIMER = (
    "本工具按法条做机械计算，未考虑代通知金、未休年假折算、医疗补助、"
    "举证责任、仲裁时效等个案因素；结果仅供参考，不构成法律意见。"
)

_CALC_KNOWN_FIELDS = frozenset(
    {
        "scenario",
        "monthly_salary",
        "years",
        "months",
        "months_without_contract",
        "overtime_hours",
        "city_avg_salary",
    }
)


# ---------------------------------------------------------------------------
# 工具 1：search_labor_law
# ---------------------------------------------------------------------------

SEARCH_TOOL_NAME = "search_labor_law"


def search_labor_law(
    query: str,
    top_k: int = 5,
    law_id: str | None = None,
) -> dict[str, Any]:
    """检索真实劳动法法条（BM25）。

    Args:
        query: 自然语言问题或关键词，如「违法解除劳动合同 赔偿金」。
        top_k: 返回条数，1-20。
        law_id: 可选法律过滤，``lcl`` / ``ll`` / ``lda`` / ``sil`` 或中文名。

    Returns:
        统一信封；``data.results`` 每项含 ``rank``/``score``/``article_no``/``law``/
        ``title``/``snippet``。未命中时 ``count == 0`` 并给出 ``hint``。
    """
    tool = SEARCH_TOOL_NAME
    try:
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query 必须是非空字符串", field="query")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ToolInputError(
                f"top_k 必须是整数，收到 {type(top_k).__name__}", field="top_k"
            )
        if not 1 <= top_k <= 20:
            raise ToolInputError(f"top_k 必须在 1-20 之间，收到 {top_k}", field="top_k")
        resolved_law: str | None = None
        if law_id is not None and str(law_id).strip():
            resolved_law = corpus.resolve_law_id(str(law_id))
            if resolved_law is None:
                raise ToolInputError(
                    f"无法识别的法律标识 {law_id!r}；可用 lcl/ll/lda/sil 或中文名",
                    field="law_id",
                )
        normalized_query = re.sub(r"\s+", " ", query).strip()
        raw_results = corpus.search_articles(
            normalized_query, top_k=top_k, law_id=resolved_law
        )
        results: list[dict[str, Any]] = []
        for item in raw_results:
            results.append(
                {
                    "rank": item["rank"],
                    "score": item["score"],
                    "id": item["id"],
                    "law_id": item["law_id"],
                    "law": item["law"],
                    "law_short": item["law_short"],
                    "article_no": item["article_no"],
                    "title": item["title"],
                    "snippet": _snippet(item["text"], normalized_query),
                    "char_len": item["char_len"],
                }
            )
        data: dict[str, Any] = {
            "query": normalized_query,
            "top_k": top_k,
            "law_id": resolved_law,
            "count": len(results),
            "results": results,
        }
        if not results:
            data["hint"] = (
                "未检索到相关法条。请换用更通用的法律关键词，"
                "例如「经济补偿」「违法解除」「二倍工资」「加班费」。"
            )
        return ok_result(tool, data)
    except ToolInputError as exc:
        return error_result(tool, exc.code, str(exc), field=exc.field)
    except Exception as exc:  # pragma: no cover - 防御性兜底
        _LOGGER.exception("search_labor_law 内部错误")
        return error_result(tool, "internal_error", f"法条检索内部错误: {exc}")


# ---------------------------------------------------------------------------
# 工具 2：calc_severance
# ---------------------------------------------------------------------------

CALC_TOOL_NAME = "calc_severance"

SEVERANCE_SCENARIOS: tuple[str, ...] = (
    "severance",
    "unlawful_termination",
    "double_wage",
    "overtime",
)

_SCENARIO_LABELS: dict[str, str] = {
    "severance": "经济补偿金 N",
    "unlawful_termination": "违法解除赔偿金 2N",
    "double_wage": "未签书面合同二倍工资差额",
    "overtime": "加班费",
}

#: 加班费倍率（《劳动法》第 44 条）
OVERTIME_RATES: dict[str, tuple[float, str]] = {
    "workday": (1.5, "工作日延时加班"),
    "rest_day": (2.0, "休息日加班且未安排补休"),
    "holiday": (3.0, "法定节假日加班"),
}

#: 月计薪天数 21.75、日标准工时 8 小时（劳社部发〔2008〕3 号）
MONTHLY_PAY_DAYS = 21.75
DAILY_STANDARD_HOURS = 8.0


def _calc_known_field_warnings(payload: dict[str, Any]) -> list[str]:
    unknown = sorted(set(payload) - _CALC_KNOWN_FIELDS)
    if not unknown:
        return []
    return [f"忽略未知字段: {', '.join(unknown)}（可用字段见 calc_severance 的 JSON Schema）"]


def _service_years_detail(years: int, months: int) -> tuple[float, str]:
    """按《劳动合同法》第 47 条把工作年限折算为「N 个月工资」的 N（年数）。"""
    if months >= 6:
        folded = float(years + 1)
        detail = f"{years} 年 {months} 个月 → 满 {years} 年 + 余 {months} 个月（六个月以上不满一年按一年）→ N = {_fmt_num(folded)}"
    elif months > 0:
        folded = years + 0.5
        detail = f"{years} 年 {months} 个月 → 满 {years} 年 + 余 {months} 个月（不满六个月按半个月）→ N = {_fmt_num(folded)}"
    else:
        folded = float(years)
        detail = f"{years} 年整 → N = {_fmt_num(folded)}"
    return folded, detail


def _calc_compensation(
    scenario: str,
    monthly_salary: float,
    city_avg_salary: float | None,
    payload: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    years = _optional_int(payload, "years", "工作年限（整年数）", default=0, minimum=0, maximum=60)
    months = _optional_int(
        payload, "months", "工作年限（不足一年的月数）", default=0, minimum=0, maximum=11
    )
    if years == 0 and months == 0:
        raise ToolInputError(
            "工作年限不能为 0：请提供 years 与 months（3 年 7 个月 → years=3, months=7）",
            field="years",
        )

    folded, service_detail = _service_years_detail(years, months)
    base_salary = monthly_salary
    cap_applied = False
    notes: list[str] = []
    cap_limit = _money(3 * city_avg_salary) if city_avg_salary is not None else None
    effective_years = folded

    if city_avg_salary is None:
        notes.append(
            "未提供 city_avg_salary（当地上年度职工月平均工资），"
            "未应用《劳动合同法》第 47 条的「社平工资 3 倍封顶、年限最高 12 年」；"
            "若你在一二线城市，实际金额可能更低。"
        )
        base_formula = (
            f"月工资 = {_fmt_money(monthly_salary)} 元；未提供社平工资，未触发封顶"
        )
    elif monthly_salary > cap_limit:  # type: ignore[operator]
        cap_applied = True
        base_salary = cap_limit  # type: ignore[assignment]
        effective_years = min(folded, 12.0)
        notes.append(
            "月工资高于当地社平工资 3 倍，已按《劳动合同法》第 47 条第二款"
            "以社平工资 3 倍为基数计算，且计算年限最高不超过 12 年。"
        )
        base_formula = (
            f"月工资 {_fmt_money(monthly_salary)} 元 > 社平 {_fmt_money(city_avg_salary)}"
            f" × 3 = {_fmt_money(cap_limit)} 元 → 基数按 {_fmt_money(cap_limit)} 元计，"
            f"年限封顶 12 年"
        )
    else:
        base_formula = (
            f"月工资 {_fmt_money(monthly_salary)} 元 ≤ 社平 {_fmt_money(city_avg_salary)}"
            f" × 3 = {_fmt_money(cap_limit)} 元，未触发封顶"
        )

    severance_amount = _money(base_salary * effective_years)
    steps: list[dict[str, Any]] = [
        {
            "step": 1,
            "label": "工作年限折算",
            "formula": service_detail,
            "value": effective_years,
            "unit": "年（N 的月数）",
        },
        {
            "step": 2,
            "label": "月工资基数",
            "formula": base_formula,
            "value": _money(base_salary),
            "unit": "元/月",
        },
        {
            "step": 3,
            "label": "经济补偿金 N",
            "formula": f"N × 月工资基数 = {_fmt_num(effective_years)} × {_fmt_money(base_salary)}",
            "value": severance_amount,
            "unit": "元",
        },
    ]

    if scenario == "severance":
        total = severance_amount
        summary = (
            f"经济补偿金 N = {_fmt_num(effective_years)} 个月 × "
            f"{_fmt_money(base_salary)} 元/月 = {_fmt_money(total)} 元"
        )
        legal_refs: tuple[tuple[str, str], ...] = (("lcl", "第四十七条"),)
    else:
        total = _money(severance_amount * 2)
        steps.append(
            {
                "step": 4,
                "label": "违法解除赔偿金 2N",
                "formula": f"2 × 经济补偿金 = 2 × {_fmt_money(severance_amount)}",
                "value": total,
                "unit": "元",
            }
        )
        summary = (
            f"经济补偿金 N = {_fmt_money(severance_amount)} 元；"
            f"违法解除赔偿金 2N = {_fmt_money(total)} 元"
        )
        legal_refs = (("lcl", "第四十七条"), ("lcl", "第八十七条"))
        notes.append(
            "依据《劳动合同法》第 87 条，用人单位违法解除应按第 47 条经济补偿标准的 2 倍支付赔偿金；"
            "若劳动者要求继续履行合同，则可能不适用 2N（第 48 条）。"
        )

    result: dict[str, Any] = {
        "scenario": scenario,
        "scenario_label": _SCENARIO_LABELS[scenario],
        "monthly_salary": _money(monthly_salary),
        "city_avg_salary": _money(city_avg_salary) if city_avg_salary else None,
        "service": {"years": years, "months": months},
        "n_years": effective_years,
        "base_salary": _money(base_salary),
        "cap_applied": cap_applied,
        "severance_amount": severance_amount,
        "total": total,
        "total_yuan": _to_yuan(total),
        "steps": steps,
        "notes": notes,
        "warnings": warnings,
        "legal_basis": _legal_basis(*legal_refs),
        "summary": summary,
        "disclaimer": _DISCLAIMER,
    }
    return result


def _calc_double_wage(
    monthly_salary: float,
    payload: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    months_without = _require_int(
        payload,
        "months_without_contract",
        "未签书面劳动合同的月数",
        minimum=0,
        maximum=120,
    )
    capped_months = min(months_without, 11)
    total = _money(monthly_salary * capped_months)
    notes: list[str] = [
        "《劳动合同法》第 82 条：自用工之日起超过一个月不满一年未订立书面劳动合同的，"
        "应向劳动者每月支付二倍工资；二倍工资中的一倍是正常工资，本工具计算的是"
        "「额外一倍」的差额。",
        "差额最长支持 11 个月（自用工满一个月的次日起算至满一年的前一日）。",
    ]
    if months_without > 11:
        notes.append(
            f"未签月数 {months_without} 个月超过法定上限，超出部分（{months_without - 11} 个月）不再支持。"
        )
    if months_without == 0:
        notes.append("未签月数为 0，无需支付二倍工资差额。")

    steps: list[dict[str, Any]] = [
        {
            "step": 1,
            "label": "可主张月数",
            "formula": (
                f"未签书面合同 {months_without} 个月，法定上限 11 个月 → 计 {capped_months} 个月"
            ),
            "value": capped_months,
            "unit": "个月",
        },
        {
            "step": 2,
            "label": "每月差额",
            "formula": f"月工资 {_fmt_money(monthly_salary)} 元 × 额外一倍 = {_fmt_money(monthly_salary)} 元/月",
            "value": _money(monthly_salary),
            "unit": "元/月",
        },
        {
            "step": 3,
            "label": "二倍工资差额合计",
            "formula": f"{capped_months} 个月 × {_fmt_money(monthly_salary)} 元/月",
            "value": total,
            "unit": "元",
        },
    ]
    summary = (
        f"未签书面劳动合同 {months_without} 个月（按上限计 {capped_months} 个月），"
        f"二倍工资差额 = {capped_months} × {_fmt_money(monthly_salary)} = {_fmt_money(total)} 元"
    )
    return {
        "scenario": "double_wage",
        "scenario_label": _SCENARIO_LABELS["double_wage"],
        "monthly_salary": _money(monthly_salary),
        "months_without_contract": months_without,
        "capped_months": capped_months,
        "months_capped": months_without > 11,
        "total": total,
        "total_yuan": _to_yuan(total),
        "steps": steps,
        "notes": notes,
        "warnings": warnings,
        "legal_basis": _legal_basis(("lcl", "第八十二条")),
        "summary": summary,
        "disclaimer": _DISCLAIMER,
    }


def _calc_overtime(
    monthly_salary: float,
    payload: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    raw_hours = payload.get("overtime_hours")
    if not isinstance(raw_hours, dict):
        raise ToolInputError(
            "overtime 场景必须提供 overtime_hours 对象，"
            '例如 {"workday": 10, "rest_day": 8, "holiday": 4}',
            field="overtime_hours",
        )
    unknown = sorted(set(raw_hours) - set(OVERTIME_RATES))
    if unknown:
        warnings.append(
            f"忽略 overtime_hours 中的未知字段: {', '.join(unknown)}"
            f"（可用: {', '.join(OVERTIME_RATES)}）"
        )
    hours: dict[str, float] = {}
    for key, label in (
        ("workday", "工作日延时加班"),
        ("rest_day", "休息日加班"),
        ("holiday", "法定节假日加班"),
    ):
        value = raw_hours.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolInputError(
                f"overtime_hours.{key}（{label}小时数）必须是数字", field=f"overtime_hours.{key}"
            )
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ToolInputError(
                f"overtime_hours.{key}（{label}小时数）必须是非负有限数字", field=f"overtime_hours.{key}"
            )
        if number > 1000:
            raise ToolInputError(
                f"overtime_hours.{key} 超过合理上限 1000 小时，请检查输入",
                field=f"overtime_hours.{key}",
            )
        hours[key] = number
    if sum(hours.values()) <= 0:
        raise ToolInputError(
            "overtime_hours 至少有一项大于 0（workday / rest_day / holiday）",
            field="overtime_hours",
        )

    daily_wage = monthly_salary / MONTHLY_PAY_DAYS
    hourly_wage = daily_wage / DAILY_STANDARD_HOURS
    steps: list[dict[str, Any]] = [
        {
            "step": 1,
            "label": "小时工资换算",
            "formula": (
                f"月工资 {_fmt_money(monthly_salary)} ÷ {_fmt_num(MONTHLY_PAY_DAYS)} 天"
                f" ÷ {_fmt_num(DAILY_STANDARD_HOURS)} 小时 = {hourly_wage:.4f} 元/小时"
                f"（月计薪天数 21.75，劳社部发〔2008〕3 号）"
            ),
            "value": round(hourly_wage, 4),
            "unit": "元/小时",
        }
    ]
    amounts: dict[str, float] = {}
    for idx, (key, (rate, label)) in enumerate(OVERTIME_RATES.items(), start=2):
        amount = _money(hourly_wage * hours[key] * rate)
        amounts[key] = amount
        steps.append(
            {
                "step": idx,
                "label": f"{label}（{_fmt_num(rate)} 倍）",
                "formula": (
                    f"{_fmt_num(hours[key])} 小时 × {_fmt_num(rate)} 倍 × "
                    f"{hourly_wage:.4f} 元/小时 = {_fmt_money(amount)} 元"
                ),
                "value": amount,
                "unit": "元",
            }
        )
    total = _money(sum(amounts.values()))
    steps.append(
        {
            "step": 5,
            "label": "加班费合计",
            "formula": (
                f"工作日 {_fmt_money(amounts['workday'])} + 休息日 {_fmt_money(amounts['rest_day'])}"
                f" + 法定节假日 {_fmt_money(amounts['holiday'])}"
            ),
            "value": total,
            "unit": "元",
        }
    )
    summary = (
        f"加班费合计 {_fmt_money(total)} 元（工作日 {_fmt_money(amounts['workday'])} 元 + "
        f"休息日 {_fmt_money(amounts['rest_day'])} 元 + 法定节假日 {_fmt_money(amounts['holiday'])} 元）"
    )
    return {
        "scenario": "overtime",
        "scenario_label": _SCENARIO_LABELS["overtime"],
        "monthly_salary": _money(monthly_salary),
        "daily_wage": _money(daily_wage),
        "hourly_wage": round(hourly_wage, 4),
        "overtime_hours": {key: _fmt_num(value) for key, value in hours.items()},
        "amount_breakdown": {"workday": amounts["workday"], "rest_day": amounts["rest_day"], "holiday": amounts["holiday"]},
        "total": total,
        "total_yuan": _to_yuan(total),
        "steps": steps,
        "notes": [
            "《劳动法》第 44 条：工作日延时加班不低于 1.5 倍、休息日加班且不能补休不低于 2 倍、"
            "法定节假日加班不低于 3 倍工资。",
            "休息日加班若已安排补休，无需支付 2 倍工资；法定节假日加班不能以补休替代。",
        ],
        "warnings": warnings,
        "legal_basis": _legal_basis(("ll", "第四十四条")),
        "summary": summary,
        "disclaimer": _DISCLAIMER,
    }


def _calc_severance_impl(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ToolInputError(
            f"payload 必须是 JSON 对象，收到 {type(payload).__name__}", field="payload"
        )
    scenario = payload.get("scenario")
    if scenario not in SEVERANCE_SCENARIOS:
        raise ToolInputError(
            f"scenario 必须是 {list(SEVERANCE_SCENARIOS)} 之一，收到 {scenario!r}",
            field="scenario",
        )
    monthly_salary = _require_number(
        payload, "monthly_salary", "月工资（元）", gt0=True, maximum=10_000_000
    )
    city_avg_salary = _optional_number(
        payload, "city_avg_salary", "当地上年度职工月平均工资（元）", gt0=True, maximum=1_000_000
    )
    warnings = _calc_known_field_warnings(payload)
    if scenario in ("severance", "unlawful_termination"):
        return _calc_compensation(scenario, monthly_salary, city_avg_salary, payload, warnings)
    if scenario == "double_wage":
        return _calc_double_wage(monthly_salary, payload, warnings)
    return _calc_overtime(monthly_salary, payload, warnings)


def calc_severance(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """劳动法赔偿/加班费计算器（按场景分流）。

    支持的 ``scenario``：

    - ``severance``：经济补偿金 N（《劳动合同法》第 47 条）；
    - ``unlawful_termination``：违法解除赔偿金 2N（第 87 条，基数为第 47 条标准）；
    - ``double_wage``：未签书面合同二倍工资差额（第 82 条，最多 11 个月）；
    - ``overtime``：加班费（《劳动法》第 44 条，工作日 1.5 / 休息日 2 / 法定节假日 3 倍）。

    返回的 ``data.steps`` 是逐步计算过程（公式 + 金额），用于面试讲解与结果复核。
    """
    tool = CALC_TOOL_NAME
    try:
        data = _calc_severance_impl(payload if payload is not None else {})
        return ok_result(tool, data)
    except ToolInputError as exc:
        return error_result(tool, exc.code, str(exc), field=exc.field)
    except Exception as exc:  # pragma: no cover - 防御性兜底
        _LOGGER.exception("calc_severance 内部错误")
        return error_result(tool, "internal_error", f"赔偿计算器内部错误: {exc}")


# ---------------------------------------------------------------------------
# 工具 3：verify_citation
# ---------------------------------------------------------------------------

VERIFY_TOOL_NAME = "verify_citation"

_UNIT_KIND: dict[str, str] = {
    "个月": "month",
    "月": "month",
    "年": "year",
    "倍": "ratio",
    "日": "day",
    "天": "day",
    "小时": "hour",
    "元": "money",
}

_KIND_LABELS: dict[str, str] = {
    "month": "个月",
    "year": "年",
    "ratio": "倍",
    "day": "天",
    "hour": "小时",
    "money": "元",
}

#: 条号引用（如「第四十七条」「第82条」「第十二条第一款」）不参与数字比对。
_ARTICLE_REF_RE = re.compile(r"第[0-9零一二三四五六七八九十百两]{1,5}[条款项章节]")
_PERCENT_CN_RE = re.compile(r"百分之([零一二三四五六七八九十百两]+)")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[%％]")
_HALF_RE = re.compile(r"半\s*(个月|月|年|日|天)")
_NUM_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?|[零一二三四五六七八九十百两]+)\s*(个月|月|年|倍|日|天|小时|元)"
)

#: claim 中出现、且用于判断「这句话是否真的在讲这一条」的法律关键词。
LEGAL_KEYWORDS: tuple[str, ...] = (
    "经济补偿金",
    "经济补偿",
    "赔偿金",
    "二倍工资",
    "两倍工资",
    "双倍工资",
    "加班费",
    "加班",
    "月工资",
    "工资",
    "书面劳动合同",
    "书面合同",
    "书面",
    "劳动合同",
    "合同",
    "社会保险",
    "社保",
    "试用期",
    "竞业限制",
    "年休假",
    "工伤",
    "代通知金",
    "医疗期",
    "最低工资",
    "劳动报酬",
    "培训",
    "服务期",
    "违约金",
    "劳务派遣",
    "女职工",
    "未成年工",
    "休息日",
    "法定节假日",
    "工作日",
    "解除",
    "终止",
    "补偿",
    "赔偿",
)


#: 关键词同义写法：claim 与法条原文用词可能不同（如「二倍工资」vs「二倍的工资」），
#: 命中任一写法即视为该关键词出现在法条中，减少误判为 partial。
KEYWORD_VARIANTS: dict[str, tuple[str, ...]] = {
    "经济补偿金": ("经济补偿金", "经济补偿"),
    "经济补偿": ("经济补偿",),
    "赔偿金": ("赔偿金", "赔偿"),
    "二倍工资": ("二倍工资", "二倍的工资", "二倍"),
    "两倍工资": ("两倍工资", "二倍的工资", "二倍"),
    "双倍工资": ("双倍工资", "二倍的工资", "二倍"),
    "加班费": ("加班费", "加班", "工资报酬"),
    "加班": ("加班",),
    "月工资": ("月工资", "工资"),
    "工资": ("工资", "劳动报酬"),
    "书面劳动合同": ("书面劳动合同", "书面合同", "书面"),
    "书面合同": ("书面合同", "书面劳动合同"),
    "书面": ("书面",),
    "劳动合同": ("劳动合同", "合同"),
    "合同": ("合同",),
    "社会保险": ("社会保险", "社保"),
    "社保": ("社保", "社会保险"),
    "法定节假日": ("法定节假日", "法定休假日", "法定假日"),
    "休息日": ("休息日",),
}


def _keyword_found(keyword: str, article_text: str) -> bool:
    """关键词是否出现在法条原文中（容忍常见同义写法）。"""
    for variant in KEYWORD_VARIANTS.get(keyword, (keyword,)):
        if variant in article_text:
            return True
    return False


def extract_numeric_claims(text: str) -> list[dict[str, Any]]:
    """从文本中提取带单位的数字要素（条号引用会被屏蔽，不参与比对）。

    例如 ``「六个月以上不满一年，支付半个月工资，最高12年」`` 会提取出
    ``6 个月``、``1 年``、``0.5 个月``、``12 年``。百分比统一换算为倍率
    （``150%`` / ``百分之一百五十`` → 1.5 倍），便于和法条原文比对。
    """
    if not text:
        return []
    masked = _ARTICLE_REF_RE.sub("〔条号〕", text)
    claims: list[dict[str, Any]] = []
    seen: set[tuple[float, str]] = set()

    def add(raw: str, value: float, kind: str) -> None:
        if value is None or not math.isfinite(value) or value <= 0:
            return
        key = (round(value, 4), kind)
        if key in seen:
            return
        seen.add(key)
        claims.append({"raw": raw.strip(), "value": round(value, 4), "kind": kind})

    for match in _PERCENT_CN_RE.finditer(masked):
        number = corpus.chinese_to_int(match.group(1))
        if number is not None:
            add(match.group(0), number / 100.0, "ratio")
    for match in _PERCENT_RE.finditer(masked):
        add(match.group(0), float(match.group(1)) / 100.0, "ratio")
    for match in _HALF_RE.finditer(masked):
        add(match.group(0), 0.5, _UNIT_KIND[match.group(1)])
    for match in _NUM_UNIT_RE.finditer(masked):
        raw_number, unit = match.group(1), match.group(2)
        if re.fullmatch(r"\d+(?:\.\d+)?", raw_number):
            value = float(raw_number)
        else:
            parsed = corpus.chinese_to_int(raw_number)
            if parsed is None:
                continue
            value = float(parsed)
        add(match.group(0), value, _UNIT_KIND[unit])
    return claims


_LAW_NAME_RE = re.compile(
    "|".join(
        sorted(
            (
                "中华人民共和国劳动争议调解仲裁法",
                "中华人民共和国劳动合同法",
                "中华人民共和国社会保险法",
                "中华人民共和国劳动法",
                "劳动争议调解仲裁法",
                "劳动合同法",
                "社会保险法",
                "劳动法",
            ),
            key=len,
            reverse=True,
        )
    )
)


def _strip_law_names(text: str) -> str:
    """去掉 claim 里的法律名称，避免「《劳动合同法》」中的「劳动合同」被当成关键词。"""
    cleaned = re.sub(r"《[^》]{0,40}》", " ", text or "")
    return _LAW_NAME_RE.sub(" ", cleaned)


def extract_keywords(claim: str) -> list[str]:
    """提取 claim 中出现的关键法律词，并去掉被更长关键词包含的短词。"""
    text = _strip_law_names(claim)
    hits = sorted(
        (keyword for keyword in LEGAL_KEYWORDS if keyword in text),
        key=lambda keyword: (-len(keyword), LEGAL_KEYWORDS.index(keyword)),
    )
    kept: list[str] = []
    for keyword in hits:
        if any(keyword in longer for longer in kept):
            continue
        kept.append(keyword)
    return kept


def _verify_not_found(
    tool: str,
    claim: str,
    reasons: list[str],
    *,
    law_id: str | None = None,
    article_no: str | None = None,
) -> dict[str, Any]:
    return ok_result(
        tool,
        {
            "status": "not_found",
            "law_id": law_id,
            "article_no": article_no,
            "article_found": False,
            "claim": claim,
            "checks": {"numbers": [], "keywords": []},
            "matched": 0,
            "checked": 0,
            "missing": list(reasons),
            "reasons": reasons,
            "article": None,
        },
    )


def verify_citation(
    law_id: str | None = None,
    article_no: str | int | None = None,
    claim: str | None = None,
) -> dict[str, Any]:
    """核验一条法条引用：条号是否存在 + claim 里的数字/关键词是否真的出现在法条中。

    Args:
        law_id: 法律标识或中文名，如 ``lcl`` / ``劳动合同法``。
        article_no: 条号，支持 ``47`` / ``"第47条"`` / ``"第四十七条"``。
        claim: 待核验的引用内容（一句话或一段话）。

    Returns:
        统一信封，``data.status`` 取值：

        - ``verified``：条号存在，且 claim 中可提取的数字、关键词全部出现在该法条中；
        - ``partial``：条号存在，部分要素缺失，或 claim 中没有可核验要素（需人工复核）；
        - ``not_found``：条号不存在，或条号存在但 claim 要素与该条完全对不上（疑似张冠李戴）。

        注意：``status=not_found`` 仍是「核验成功执行」，``ok`` 为 ``True``；
        ``ok=False`` 只表示参数错误/内部错误。
    """
    tool = VERIFY_TOOL_NAME
    try:
        claim_text = claim.strip() if isinstance(claim, str) else ""
        if not claim_text:
            raise ToolInputError("claim 必须是非空字符串（待核验的引用内容）", field="claim")
        if law_id is None or not str(law_id).strip():
            raise ToolInputError("law_id 不能为空，如 lcl / 劳动合同法", field="law_id")
        if article_no is None or str(article_no).strip() == "":
            raise ToolInputError("article_no 不能为空，如 47 / 第四十七条", field="article_no")

        resolved_law = corpus.resolve_law_id(str(law_id))
        if resolved_law is None:
            return _verify_not_found(
                tool,
                claim_text,
                [f"无法识别的法律标识 {law_id!r}；可用 lcl/ll/lda/sil 或中文名（如「劳动合同法」）"],
            )
        normalized_no = corpus.normalize_article_no(article_no)
        if normalized_no is None:
            return _verify_not_found(
                tool,
                claim_text,
                [f"无法解析的条号 {article_no!r}；请使用 47 / 第47条 / 第四十七条 等写法"],
                law_id=resolved_law,
            )

        article = corpus.get_article(resolved_law, normalized_no)
        law_short = corpus.LAW_SHORT_NAMES.get(resolved_law, resolved_law)
        if article is None:
            return _verify_not_found(
                tool,
                claim_text,
                [f"{law_short}中不存在{normalized_no}（条号超出该法律条文范围，或条号写法有误）"],
                law_id=resolved_law,
                article_no=normalized_no,
            )

        claim_numbers = extract_numeric_claims(claim_text)
        article_numbers = extract_numeric_claims(article.text)
        article_keys = {(item["value"], item["kind"]) for item in article_numbers}
        number_checks: list[dict[str, Any]] = []
        for item in claim_numbers:
            number_checks.append(
                {
                    **item,
                    "found": (item["value"], item["kind"]) in article_keys,
                }
            )

        claim_keywords = extract_keywords(claim_text)
        keyword_checks: list[dict[str, Any]] = [
            {"keyword": keyword, "found": _keyword_found(keyword, article.text)}
            for keyword in claim_keywords
        ]

        matched = sum(1 for check in number_checks + keyword_checks if check["found"])
        checked = len(number_checks) + len(keyword_checks)
        missing: list[str] = []
        for check in number_checks:
            if not check["found"]:
                missing.append(
                    f"claim 中的数字「{check['raw']}」（{_fmt_num(check['value'])} "
                    f"{_KIND_LABELS[check['kind']]}）未在{article.title}中出现"
                )
        for check in keyword_checks:
            if not check["found"]:
                missing.append(
                    f"claim 中的关键词「{check['keyword']}」未在{article.title}中出现"
                )

        reasons: list[str] = [f"条号存在：{article.title}（{article.chapter}）"]
        if checked == 0:
            status = "partial"
            reasons.append(
                "claim 中未提取到带单位的数字或法律关键词，无法自动核验，需人工复核。"
            )
        elif matched == checked:
            status = "verified"
            reasons.append(
                f"claim 中可提取的 {checked} 个要素（数字/关键词）全部出现在法条原文中。"
            )
        elif matched == 0:
            status = "not_found"
            reasons.append(
                "条号存在，但 claim 中的数字/关键词在该法条原文中完全找不到，引用很可能张冠李戴。"
            )
        else:
            status = "partial"
            reasons.append(
                f"claim 中 {checked} 个可核验要素有 {checked - matched} 个未在法条原文中找到，"
                "需人工复核（也可能是同义表述差异）。"
            )
        reasons.extend(missing)

        return ok_result(
            tool,
            {
                "status": status,
                "law_id": resolved_law,
                "law": article.law,
                "law_short": law_short,
                "article_no": article.article_no,
                "citation": article.title,
                "article_found": True,
                "claim": claim_text,
                "checks": {
                    "numbers": number_checks,
                    "keywords": keyword_checks,
                },
                "matched": matched,
                "checked": checked,
                "missing": missing,
                "reasons": reasons,
                "article": article.to_dict(text_max_chars=600),
            },
        )
    except ToolInputError as exc:
        return error_result(tool, exc.code, str(exc), field=exc.field)
    except Exception as exc:  # pragma: no cover - 防御性兜底
        _LOGGER.exception("verify_citation 内部错误")
        return error_result(tool, "internal_error", f"引用核验内部错误: {exc}")


# ---------------------------------------------------------------------------
# 统一调用入口 + OpenAI function calling Schema
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, ToolImpl] = {
    SEARCH_TOOL_NAME: search_labor_law,
    CALC_TOOL_NAME: calc_severance,
    VERIFY_TOOL_NAME: verify_citation,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": SEARCH_TOOL_NAME,
            "description": (
                "检索中国劳动法真实法条（4 部法律共 357 条）。输入自然语言问题或法律关键词，"
                "返回最相关法条的条号、法律名称和正文片段。当用户问「法律怎么规定/有没有依据/"
                "是哪一条」时使用；不要凭空编造条号或法条内容。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索词或自然语言问题，如「违法解除劳动合同 赔偿金」「未签书面合同 二倍工资」",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "返回条数，默认 5",
                    },
                    "law_id": {
                        "type": "string",
                        "enum": ["lcl", "ll", "lda", "sil"],
                        "description": "可选限定法律：lcl=劳动合同法，ll=劳动法，lda=劳动争议调解仲裁法，sil=社会保险法；不填则检索全部",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": CALC_TOOL_NAME,
            "description": (
                "劳动法赔偿/加班费计算器，返回逐步计算过程（公式 + 金额）。scenario 取值："
                "severance=经济补偿金 N（劳动合同法第47条）；"
                "unlawful_termination=违法解除赔偿金 2N（第87条）；"
                "double_wage=未签书面合同二倍工资差额（第82条，最多11个月）；"
                "overtime=加班费（劳动法第44条：工作日1.5倍/休息日2倍/法定节假日3倍）。"
                "涉及金额时优先调用，不要自己心算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario": {
                        "type": "string",
                        "enum": list(SEVERANCE_SCENARIOS),
                        "description": "计算场景",
                    },
                    "monthly_salary": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "月工资（元）：解除前 12 个月平均工资，或主张二倍工资/加班费所依据的月工资",
                    },
                    "years": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "工作年限的整年数（3 年 7 个月 → 3）",
                    },
                    "months": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 11,
                        "description": "工作年限中不满一年的月数（3 年 7 个月 → 7）",
                    },
                    "months_without_contract": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "未签书面劳动合同的月数（double_wage 场景）",
                    },
                    "overtime_hours": {
                        "type": "object",
                        "description": "加班小时数（overtime 场景）：工作日延时/休息日/法定节假日",
                        "properties": {
                            "workday": {"type": "number", "minimum": 0, "description": "工作日延时加班小时数"},
                            "rest_day": {"type": "number", "minimum": 0, "description": "休息日加班小时数"},
                            "holiday": {"type": "number", "minimum": 0, "description": "法定节假日加班小时数"},
                        },
                        "additionalProperties": False,
                    },
                    "city_avg_salary": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "当地上年度职工月平均工资（元），用于《劳动合同法》第47条 3 倍封顶；不填则按无封顶计算",
                    },
                },
                "required": ["scenario", "monthly_salary"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": VERIFY_TOOL_NAME,
            "description": (
                "核验一条法条引用是否属实：先确认条号是否真实存在，再检查 claim 中的关键数字/关键词"
                "是否真的出现在该法条原文中，返回 verified / partial / not_found 与逐项理由。"
                "在回答中引用法条前调用它做自检，可有效避免编造条号或数字。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "law_id": {
                        "type": "string",
                        "description": "法律标识或简称：lcl/ll/lda/sil，或「劳动合同法」「劳动法」「社会保险法」「劳动争议调解仲裁法」",
                    },
                    "article_no": {
                        "type": ["string", "integer"],
                        "description": "条号，支持 47 / \"第47条\" / \"第四十七条\"",
                    },
                    "claim": {
                        "type": "string",
                        "description": "待核验的引用内容，如「未签书面合同最多支持11个月二倍工资」",
                    },
                },
                "required": ["law_id", "article_no", "claim"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_NAME_TO_SCHEMA: dict[str, dict[str, Any]] = {
    schema["function"]["name"]: schema for schema in TOOL_SCHEMAS
}


def get_tool_schemas() -> list[dict[str, Any]]:
    """返回 OpenAI function calling 格式的工具定义（返回副本，调用方可安全修改）。"""
    return json.loads(json.dumps(TOOL_SCHEMAS, ensure_ascii=False))


def execute_tool_call(
    name: str,
    arguments: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """统一工具调用入口（Agent 主循环只依赖这一个函数）。

    负责：
    - 拒绝未知工具名（``unknown_tool``）；
    - 解析 JSON 字符串参数，解析失败返回 ``invalid_arguments_json``；
    - 参数不是对象、字段不合法 → ``invalid_arguments``；
    - 任何未预期异常 → ``internal_error``。
    任何情况下都返回结构化信封，不向 Agent 主循环抛异常。
    """
    tool = str(name or "").strip()
    if tool not in TOOL_REGISTRY:
        return error_result(
            tool,
            "unknown_tool",
            f"未知工具 {tool!r}；可用工具: {', '.join(TOOL_REGISTRY)}",
        )
    if arguments is None:
        args: Any = {}
    elif isinstance(arguments, dict):
        args = dict(arguments)
    elif isinstance(arguments, str):
        raw = arguments.strip()
        if not raw:
            args = {}
        else:
            try:
                args = json.loads(raw)
            except json.JSONDecodeError as exc:
                return error_result(
                    tool,
                    "invalid_arguments_json",
                    f"工具参数不是合法 JSON：{exc.msg}（位置 {exc.pos}）；原始参数: {raw[:200]!r}",
                )
    else:
        return error_result(
            tool,
            "invalid_arguments",
            f"工具参数必须是 JSON 字符串或对象，收到 {type(arguments).__name__}",
        )
    if not isinstance(args, dict):
        return error_result(
            tool,
            "invalid_arguments",
            f"工具参数必须是 JSON 对象，收到 JSON {type(args).__name__}",
        )
    try:
        # calc_severance 的契约是「一个结构化 payload 对象」（见函数签名与 README），
        # 而 JSON Schema 描述的也是这个对象，因此整体传入；其余工具按字段展开传入。
        if tool == CALC_TOOL_NAME:
            return calc_severance(args)
        return TOOL_REGISTRY[tool](**args)
    except TypeError as exc:
        return error_result(tool, "invalid_arguments", f"工具参数不合法：{exc}")
    except Exception as exc:  # pragma: no cover - 防御性兜底
        _LOGGER.exception("工具 %s 未捕获异常", tool)
        return error_result(tool, "internal_error", f"工具执行异常：{exc}")


def tool_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """把工具返回压缩成适合写进 trace 的摘要（不丢关键信息）。"""
    if not payload.get("ok"):
        return {"ok": False, "error": payload.get("error")}
    data = payload.get("data") or {}
    tool = payload.get("tool")
    if tool == SEARCH_TOOL_NAME:
        return {
            "ok": True,
            "count": data.get("count"),
            "top": [
                {
                    "title": item["title"],
                    "score": item["score"],
                    "snippet": item["snippet"],
                }
                for item in (data.get("results") or [])[:5]
            ],
        }
    if tool == CALC_TOOL_NAME:
        return {
            "ok": True,
            "scenario": data.get("scenario"),
            "total": data.get("total"),
            "summary": data.get("summary"),
            "steps": data.get("steps"),
            "notes": data.get("notes"),
            "warnings": data.get("warnings"),
        }
    if tool == VERIFY_TOOL_NAME:
        return {
            "ok": True,
            "status": data.get("status"),
            "article_found": data.get("article_found"),
            "citation": data.get("citation") or data.get("article_no"),
            "matched": data.get("matched"),
            "checked": data.get("checked"),
            "reasons": data.get("reasons"),
            "checks": data.get("checks"),
        }
    return {"ok": True, "data": data}
