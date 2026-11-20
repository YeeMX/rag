"""LLM 抽象层：离线可用的 MockLLM + 可选的 OpenAI 兼容真实模型。

设计：
- :class:`BaseLLM` 只暴露一个方法 ``complete(messages, tools) -> LLMResponse``，
  上层（function calling 循环 / LangGraph 节点）不关心背后是 mock 还是真模型。
- :class:`MockLLM` 是**确定性的规则引擎**，用关键词规则「扮演」模型做工具选择，
  完全离线、不需要 API key。它在每条输出里都显式标注 ``MOCK``，不会假装是真模型。
- :class:`OpenAICompatLLM` 在设置 ``OPENAI_API_KEY``（可选 ``OPENAI_BASE_URL`` /
  ``OPENAI_MODEL``）时启用，走标准 OpenAI tools/function calling 协议。

真实模型没有配置时，:func:`build_llm` 会回退到 MockLLM，保证 demo 永远能跑。
"""

from __future__ import annotations

import abc
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import corpus

_LOGGER = logging.getLogger(__name__)

MOCK_BANNER = "【MOCK 离线模拟输出】"
MOCK_THOUGHT_PREFIX = "（MOCK 规则引擎）"
"""MockLLM 用来标注自身身份的前缀，测试会断言输出里包含它。"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """一次工具调用请求（arguments 保持 JSON 字符串，解析交给主循环/工具入口）。"""

    id: str
    name: str
    arguments_json: str

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments_json},
        }


@dataclass
class LLMResponse:
    """模型一次回复：要么是自然语言，要么是若干工具调用（OpenAI 协议里两者可并存）。"""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    mock: bool = False
    provider: str = "unknown"
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    latency_ms: float | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    """模型调用失败（网络/鉴权/响应格式异常等）。"""


class LLMUnavailableError(LLMError):
    """真实模型不可用（缺少 API key / 缺少 openai 包）。"""


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseLLM(abc.ABC):
    """LLM 抽象：上层只依赖这个接口。"""

    name: str = "base"

    @property
    @abc.abstractmethod
    def is_mock(self) -> bool:
        """是否为离线模拟模型。"""

    @abc.abstractmethod
    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """给定对话历史与工具定义，返回一次模型回复。"""

    def describe(self) -> str:
        kind = "MOCK（离线规则模拟，非真实大模型）" if self.is_mock else "真实/兼容模型"
        return f"LLM={self.name}｜{kind}"


# ---------------------------------------------------------------------------
# MockLLM：离线、确定性的规则引擎
# ---------------------------------------------------------------------------

_NUM_TOKEN = r"(?:\d+(?:\.\d+)?|[零一二三四五六七八九十百两]+)"

_CITY_AVG_RE = re.compile(
    rf"(?:当地)?(?:社平工资|社会平均工资|职工月平均工资|上年度职工月平均工资|平均工资|社平)"
    rf"\s*(?:是|为|约|大概)?\s*({_NUM_TOKEN})\s*(?:元)?"
)
_SALARY_RE = re.compile(
    rf"(?:月薪|月工资|月收入|税后月薪|工资)\s*(?:是|为|约|大概|有)?\s*(?:人民币)?\s*"
    rf"({_NUM_TOKEN})\s*(万|元|块)?"
)
_SERVICE_RE = re.compile(
    rf"({_NUM_TOKEN})\s*年(?:半|(?:零|又|个)?\s*({_NUM_TOKEN})\s*个?月)?"
)
_MONTHS_WITHOUT_CONTRACT_RE = (
    re.compile(
        rf"({_NUM_TOKEN})\s*个?月[^，。；,;]{{0,12}}?(?:未签|没签|没有签|未订立|未签订)"
    ),
    re.compile(
        rf"(?:未签|没签|没有签|未订立|未签订)[^，。；,;]{{0,12}}?({_NUM_TOKEN})\s*个?月"
    ),
)
_ARTICLE_RE = re.compile(r"第\s*([0-9]{1,3}|[零一二三四五六七八九十百两]{1,5})\s*条")
_OVERTIME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "workday": ("工作日", "延时", "平时"),
    "rest_day": ("休息日", "周末", "周六", "周日"),
    "holiday": ("法定节假日", "节假日", "法定假日"),
}
_OT_HOURS_TPL = r"{word}[^，。；,;]{{0,10}}?(\d+(?:\.\d+)?)\s*(?:个)?小时"
_LAW_MENTION_RE = re.compile(
    r"(中华人民共和国劳动合同法|劳动合同法|中华人民共和国劳动法|劳动争议调解仲裁法|"
    r"中华人民共和国社会保险法|社会保险法|劳动法)"
)
_VERIFY_INTENT_WORDS = ("核验", "验证", "引用", "是否准确", "是否属实", "真的", "准确吗", "对吗", "有没有这条")
_LEGAL_BASIS_WORDS = ("依据", "哪一条", "第几条", "法条", "法律依据", "条文", "相关规定", "合法吗")
_SMALL_TALK_RE = re.compile(
    r"^(你好|您好|hi|hello|嗨|在吗|谢谢|感谢|再见|你是谁|介绍一下你)[呀啊哈呢吗！!。.？?\s]*$",
    re.IGNORECASE,
)


def _to_number(token: str) -> float | None:
    """把阿拉伯数字或中文数字 token 转成 float。"""
    if not token:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)
    parsed = corpus.chinese_to_int(token)
    return float(parsed) if parsed is not None else None


def _model_call_reason(tool_names: Sequence[str]) -> str:
    reasons = {
        "calc_severance": "问题涉及具体金额，需要按法条精算",
        "search_labor_law": "需要检索真实法条作为依据",
        "verify_citation": "需要核验引用是否属实",
    }
    return "；".join(reasons.get(name, name) for name in tool_names)


@dataclass
class _MockPlan:
    """MockLLM 的「思考结果」：要调用哪些工具、以及一段可展示的思考说明。"""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    thought: str = ""


class MockLLM(BaseLLM):
    """离线规则 Mock：用关键词规则模拟「模型决定调用哪些工具」。

    它是一种**确定性的规则引擎**，不是神经网络，也不联网：

    - 识别赔偿/加班/二倍工资等意图并抽取金额、年限、加班小时数；
    - 必要时一次回复里发起多个工具调用（例如「算赔偿 + 找法律依据」）；
    - 工具返回后，用模板把结果拼装成最终回答，并显式标注 ``MOCK``。

    无法识别的信息不会瞎编：缺少必要参数时会改为检索法条，并在回答中说明缺什么。
    """

    name = "mock-rules"

    @property
    def is_mock(self) -> bool:
        return True

    # -- 主入口 ---------------------------------------------------------
    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        question = self._last_user_question(messages)
        if not question:
            return self._response(
                content=f"{MOCK_BANNER}没有收到用户问题，请补充要咨询的劳动法问题。"
            )

        tool_payloads = self._collect_tool_payloads(messages)
        if tool_payloads:
            # 已经拿到工具结果：这一轮扮演「基于工具结果作答」。
            return self._response(content=self._compose_answer(tool_payloads))

        plan = self._plan(question)
        if not plan.calls:
            return self._response(content=self._small_talk_answer(question))

        round_no = self._assistant_tool_round(messages) + 1
        tool_calls = [
            ToolCall(
                id=f"call_{round_no}_{index}",
                name=name,
                arguments_json=json.dumps(args, ensure_ascii=False),
            )
            for index, (name, args) in enumerate(plan.calls, start=1)
        ]
        return self._response(content=plan.thought, tool_calls=tool_calls)

    # -- 消息解析 -------------------------------------------------------
    @staticmethod
    def _last_user_question(messages: Sequence[dict[str, Any]]) -> str:
        for message in reversed(list(messages)):
            if message.get("role") == "user":
                content = message.get("content")
                return content if isinstance(content, str) else ""
        return ""

    @staticmethod
    def _collect_tool_payloads(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "tool" in payload:
                payloads.append(payload)
        return payloads

    @staticmethod
    def _assistant_tool_round(messages: Sequence[dict[str, Any]]) -> int:
        return sum(
            1
            for message in messages
            if message.get("role") == "assistant" and message.get("tool_calls")
        )

    @staticmethod
    def _response(
        content: str | None,
        tool_calls: list[ToolCall] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=content,
            tool_calls=list(tool_calls or []),
            mock=True,
            provider="mock-rules",
        )

    # -- 意图识别与参数抽取 ---------------------------------------------
    def _plan(self, question: str) -> _MockPlan:
        if _SMALL_TALK_RE.match(question.strip()) or len(question.strip()) <= 2:
            return _MockPlan(calls=[], thought=f"{MOCK_THOUGHT_PREFIX}这是寒暄/说明性输入，无需调用工具。")

        verify_call = self._plan_verify(question)
        if verify_call is not None:
            return _MockPlan(
                calls=[verify_call],
                thought=f"{MOCK_THOUGHT_PREFIX}这是法条引用核验请求，调用 verify_citation 逐项比对原文。",
            )

        scenario = self._detect_scenario(question)
        wants_legal_basis = any(word in question for word in _LEGAL_BASIS_WORDS)
        if scenario is not None:
            args, missing = self._extract_calc_args(question, scenario)
            if not missing:
                thought = (
                    f"{MOCK_THOUGHT_PREFIX}识别为「{self._scenario_label(scenario)}」场景，"
                    f"调用 calc_severance 精算"
                )
                calls: list[tuple[str, dict[str, Any]]] = [("calc_severance", args)]
                if wants_legal_basis:
                    thought += "；问题还要求法律依据，同时调用 search_labor_law"
                    calls.append(("search_labor_law", self._legal_basis_args(scenario)))
                thought += "。"
                return _MockPlan(calls=calls, thought=thought)
            thought = (
                f"{MOCK_THOUGHT_PREFIX}识别为「{self._scenario_label(scenario)}」场景，"
                f"但缺少必要信息（{'、'.join(missing)}），无法直接计算；改为检索相关法条，"
                f"并在回答中说明需要补充的信息。"
            )
            return _MockPlan(
                calls=[("search_labor_law", self._legal_basis_args(scenario))],
                thought=thought,
            )

        return _MockPlan(
            calls=[("search_labor_law", {"query": question.strip(), "top_k": 5})],
            thought=f"{MOCK_THOUGHT_PREFIX}未识别到可直接计算的场景，调用 search_labor_law 检索相关法条。",
        )

    def _plan_verify(self, question: str) -> tuple[str, dict[str, Any]] | None:
        if not any(word in question for word in _VERIFY_INTENT_WORDS):
            return None
        article_match = _ARTICLE_RE.search(question)
        if article_match is None:
            return None
        law_match = _LAW_MENTION_RE.search(question)
        law_id = corpus.resolve_law_id(law_match.group(1)) if law_match else "lcl"
        article_no = corpus.normalize_article_no(article_match.group(1)) or article_match.group(1)
        if "：" in question:
            claim = question.split("：", 1)[1].strip()
        elif ":" in question:
            claim = question.split(":", 1)[1].strip()
        else:
            claim = question.strip()
        if not law_id:
            law_id = "lcl"
        return (
            "verify_citation",
            {"law_id": law_id, "article_no": article_no, "claim": claim or question.strip()},
        )

    def _detect_scenario(self, question: str) -> str | None:
        lowered = question.lower()
        if "加班" in question:
            return "overtime"
        if "二倍工资" in question or "双倍工资" in question:
            return "double_wage"
        if ("未签" in question or "没签" in question or "未订立" in question) and (
            "合同" in question or "劳动合同" in question
        ):
            return "double_wage"
        if "违法" in question and any(word in question for word in ("辞退", "解除", "开除", "终止")):
            return "unlawful_termination"
        if "2n" in lowered or "二n" in lowered:
            return "unlawful_termination"
        if any(
            word in question
            for word in ("经济补偿", "补偿金", "裁员", "协商解除", "辞退", "解除", "开除", "赔偿")
        ):
            return "severance"
        return None

    @staticmethod
    def _scenario_label(scenario: str) -> str:
        return {
            "severance": "经济补偿金 N",
            "unlawful_termination": "违法解除赔偿金 2N",
            "double_wage": "未签书面合同二倍工资",
            "overtime": "加班费",
        }.get(scenario, scenario)

    @staticmethod
    def _legal_basis_args(scenario: str) -> dict[str, Any]:
        queries = {
            "severance": "经济补偿 工作年限 每满一年 一个月工资",
            "unlawful_termination": "违法解除劳动合同 赔偿金 经济补偿 第四十七条 第八十七条",
            "double_wage": "未订立书面劳动合同 二倍工资",
            "overtime": "加班 延长工作时间 休息日 法定休假日 工资报酬 百分之一百五十",
        }
        return {"query": queries[scenario], "top_k": 3}

    def _extract_money(self, question: str) -> tuple[float | None, float | None]:
        """返回 (月工资, 当地社平工资)。"""
        text = question
        city_avg: float | None = None
        city_match = _CITY_AVG_RE.search(text)
        if city_match:
            city_avg = _to_number(city_match.group(1))
            text = text[: city_match.start()] + "★" + text[city_match.end() :]

        salary: float | None = None
        salary_match = _SALARY_RE.search(text)
        if salary_match:
            value = _to_number(salary_match.group(1))
            unit = salary_match.group(2)
            if value is not None and unit == "万":
                value *= 10_000
            salary = value
        return salary, city_avg

    def _extract_service(self, question: str) -> tuple[int | None, int | None]:
        match = _SERVICE_RE.search(question)
        if not match:
            return None, None
        years_value = _to_number(match.group(1))
        if years_value is None:
            return None, None
        years = int(years_value)
        raw = match.group(0)
        if "半" in raw:
            months = 6
        else:
            months_value = _to_number(match.group(2)) if match.group(2) else 0.0
            months = int(months_value or 0)
        if months is not None and not 0 <= months <= 11:
            return None, None
        return years, months

    def _extract_months_without_contract(self, question: str) -> int | None:
        for pattern in _MONTHS_WITHOUT_CONTRACT_RE:
            match = pattern.search(question)
            if match:
                value = _to_number(match.group(1))
                if value is not None:
                    return int(value)
        return None

    def _extract_overtime_hours(self, question: str) -> dict[str, float]:
        hours: dict[str, float] = {}
        for key, words in _OVERTIME_KEYWORDS.items():
            for word in words:
                match = re.search(_OT_HOURS_TPL.format(word=word), question)
                if match:
                    hours[key] = float(match.group(1))
                    break
        return hours

    def _extract_calc_args(
        self, question: str, scenario: str
    ) -> tuple[dict[str, Any], list[str]]:
        salary, city_avg = self._extract_money(question)
        args: dict[str, Any] = {"scenario": scenario}
        missing: list[str] = []
        if salary is None:
            missing.append("月工资 monthly_salary")
        else:
            args["monthly_salary"] = salary
        if city_avg is not None:
            args["city_avg_salary"] = city_avg

        if scenario in ("severance", "unlawful_termination"):
            years, months = self._extract_service(question)
            if years is None:
                missing.append("工作年限（years/months）")
            elif years == 0 and (months or 0) == 0:
                missing.append("工作年限（不能为 0）")
            else:
                args["years"] = years
                args["months"] = int(months or 0)
        elif scenario == "double_wage":
            months_without = self._extract_months_without_contract(question)
            if months_without is None:
                missing.append("未签书面合同月数 months_without_contract")
            else:
                args["months_without_contract"] = months_without
        elif scenario == "overtime":
            hours = self._extract_overtime_hours(question)
            if not hours:
                missing.append("加班小时数 overtime_hours")
            else:
                args["overtime_hours"] = hours
        return args, missing

    # -- 回答拼装 -------------------------------------------------------
    def _small_talk_answer(self, question: str) -> str:
        return (
            f"{MOCK_BANNER}这是离线规则 Mock，不会像真实大模型一样自由对话。\n"
            "我能演示的工具链路：\n"
            "1. 赔偿计算：如「公司违法辞退我，月薪12000，工作3年7个月，能赔多少？」\n"
            "2. 法条检索：如「未签书面合同有什么法律后果？」\n"
            "3. 引用核验：如「核验这句引用：《劳动合同法》第八十二条说未签合同最多11个月二倍工资」\n"
            "提示：提问里带上「月薪」「工作年限」「加班小时数」等数字，计算会更准确。"
        )

    def _compose_answer(self, payloads: Sequence[dict[str, Any]]) -> str:
        lines: list[str] = [
            f"{MOCK_BANNER}以下内容由离线规则引擎根据工具返回结果拼装，"
            "**不是真实大模型的推理输出**，仅用于演示工具调用链路。",
            "",
        ]
        for payload in payloads:
            tool = payload.get("tool", "unknown")
            if not payload.get("ok"):
                error = payload.get("error") or {}
                lines.append(f"⚠ 工具 `{tool}` 执行失败：{error.get('message', '未知错误')}")
                lines.append("")
                continue
            data = payload.get("data") or {}
            if tool == "calc_severance":
                lines.extend(self._format_calc(data))
            elif tool == "search_labor_law":
                lines.extend(self._format_search(data))
            elif tool == "verify_citation":
                lines.extend(self._format_verify(data))
        lines.append("")
        lines.append(
            "（以上为规则模板输出。真实上线版本会由 LLM 组织自然语言，"
            "但金额与法条仍应来自工具返回，避免模型编造。）"
        )
        return "\n".join(lines)

    @staticmethod
    def _format_calc(data: dict[str, Any]) -> list[str]:
        lines = [
            f"## {data.get('scenario_label', '计算结果')}",
            f"- 结论：{data.get('summary', '')}",
            f"- 精确金额：{data.get('total', 0):.2f} 元（约 {data.get('total_yuan', 0)} 元）",
            "",
            "计算过程：",
        ]
        for step in data.get("steps") or []:
            lines.append(f"{step['step']}. {step['label']}：{step['formula']}")
        basis = data.get("legal_basis") or []
        if basis:
            lines.append("")
            lines.append("法律依据（来自真实语料）：")
            for item in basis:
                lines.append(f"- {item['title']}：{item['text'][:120]}…")
        notes = data.get("notes") or []
        if notes:
            lines.append("")
            lines.append("注意：")
            for note in notes:
                lines.append(f"- {note}")
        return lines

    @staticmethod
    def _format_search(data: dict[str, Any]) -> list[str]:
        results = data.get("results") or []
        lines = [f"## 法条检索（命中 {data.get('count', 0)} 条）"]
        if not results:
            lines.append(f"- 未命中。{data.get('hint', '')}")
            return lines
        for item in results[:3]:
            lines.append(f"- **{item['title']}**（BM25 {item['score']}）：{item['snippet']}")
        return lines

    @staticmethod
    def _format_verify(data: dict[str, Any]) -> list[str]:
        status = data.get("status")
        status_label = {
            "verified": "✅ 核验通过（条号存在，claim 要素与原文一致）",
            "partial": "⚠ 部分核验（条号存在，但部分要素未能对应，需人工复核）",
            "not_found": "❌ 未通过（条号不存在，或内容与该条完全对不上）",
        }.get(status, status)
        lines = ["## 法条引用核验", f"- 结论：{status_label}"]
        checks = data.get("checks") or {}
        numbers = checks.get("numbers") or []
        keywords = checks.get("keywords") or []
        if numbers:
            lines.append(
                "- 数字比对："
                + "、".join(
                    f"{item['raw']}{'✓' if item['found'] else '✗'}" for item in numbers
                )
            )
        if keywords:
            lines.append(
                "- 关键词比对："
                + "、".join(
                    f"{item['keyword']}{'✓' if item['found'] else '✗'}" for item in keywords
                )
            )
        for reason in data.get("reasons") or []:
            lines.append(f"- {reason}")
        return lines


# ---------------------------------------------------------------------------
# 真实模型：OpenAI 兼容接口
# ---------------------------------------------------------------------------


class OpenAICompatLLM(BaseLLM):
    """OpenAI 兼容 Chat Completions（标准 tools / function calling）。

    环境变量：

    - ``OPENAI_API_KEY``（必需）
    - ``OPENAI_BASE_URL``（可选，兼容 DeepSeek / vLLM / Ollama 等 OpenAI 兼容端点）
    - ``OPENAI_MODEL``（可选，默认 ``gpt-4o-mini``）
    - ``OPENAI_TIMEOUT``（可选，秒，默认 60）
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url if base_url is not None else os.getenv("OPENAI_BASE_URL")
        self.timeout = timeout if timeout is not None else float(os.getenv("OPENAI_TIMEOUT", "60"))
        if not self.api_key:
            raise LLMUnavailableError(
                "缺少 OPENAI_API_KEY：请设置环境变量后重试，或使用 --mock 走离线模式。"
            )
        try:
            from openai import OpenAI  # 延迟导入：未安装 openai 时其他模块仍可用
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise LLMUnavailableError(
                "未安装 openai 包，无法调用真实模型；请使用 --mock 离线模式。"
            ) from exc
        client_kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = OpenAI(**client_kwargs)
        self.name = f"openai-compatible:{self.model}"

    @property
    def is_mock(self) -> bool:
        return False

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0,
        }
        if tools:
            request["tools"] = list(tools)
            request["tool_choice"] = "auto"
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:  # openai 的各种异常统一包装
            raise LLMError(f"真实模型调用失败：{type(exc).__name__}: {exc}") from exc

        choices = getattr(response, "choices", None) or []
        if not choices:  # pragma: no cover - 防御性兜底
            raise LLMError("模型返回空 choices")
        message = choices[0].message
        tool_calls: list[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            tool_calls.append(
                ToolCall(
                    id=str(getattr(call, "id", "") or f"call_{len(tool_calls)}"),
                    name=str(getattr(call.function, "name", "")),
                    arguments_json=str(getattr(call.function, "arguments", "") or ""),
                )
            )
        usage = None
        if getattr(response, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }
        return LLMResponse(
            content=getattr(message, "content", None),
            tool_calls=tool_calls,
            mock=False,
            provider=self.name,
            usage=usage,
        )


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def has_openai_credentials() -> bool:
    """当前环境是否配置了 OPENAI_API_KEY。"""
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def build_llm(backend: str = "auto") -> BaseLLM:
    """构造 LLM。

    Args:
        backend: ``mock``（强制离线） / ``openai``（强制真实，失败抛错） /
            ``auto``（有 key 用真实模型，否则回退 Mock，保证永远可跑）。
    """
    backend = (backend or "auto").strip().lower()
    if backend == "mock":
        return MockLLM()
    if backend == "openai":
        return OpenAICompatLLM()
    if backend == "auto":
        if has_openai_credentials():
            try:
                return OpenAICompatLLM()
            except LLMUnavailableError as exc:
                _LOGGER.warning("真实模型不可用，回退 MockLLM：%s", exc)
                return MockLLM()
        return MockLLM()
    raise ValueError(f"未知 backend {backend!r}，可选 auto / mock / openai")
