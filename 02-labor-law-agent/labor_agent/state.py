"""两种 Agent 实现（原生 function calling / LangGraph）共享的运行状态与工具执行逻辑。

设计目的（面试可讲）：方案 A 和方案 B 只负责「控制流」，下面这些能力完全复用一份代码：

- :class:`AgentState`：同一份状态结构（TypedDict，可被 LangGraph 直接用作 channel）；
- :func:`assistant_message`：把 :class:`~labor_agent.llm.LLMResponse` 转成 OpenAI 消息；
- :func:`execute_pending_calls`：执行工具调用、回填 tool 消息、写 trace；
- :func:`request_forced_answer` / :func:`fallback_answer`：轮数预算耗尽时的兜底；
- :class:`AgentResult`：两种实现统一的返回结构与 JSON 序列化。

这样对比两种实现时，差异只剩「谁决定下一跳」和「状态怎么流转」，而不是两套重复代码。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypedDict

from . import tools
from .llm import BaseLLM, LLMError, LLMResponse

SYSTEM_PROMPT = """你是一名严谨的中国劳动法咨询 Agent，服务对象是普通劳动者。

工作规则：
1. 涉及具体金额（经济补偿、赔偿金、二倍工资、加班费）时，必须调用 calc_severance 工具计算，
   不要自己心算，也不要把没有依据的数字写进回答。
2. 涉及法律依据、条号、规则时，必须调用 search_labor_law 检索真实法条；只引用工具返回的法条，
   严禁编造条号或法条内容。
3. 回答中引用具体法条前，优先调用 verify_citation 核验条号与关键数字是否属实。
4. 工具返回错误或信息不足时，要如实告诉用户缺什么信息、建议补充什么，不要猜测。
5. 用简洁的中文回答，先给结论再给依据；最后附一句「以上为参考分析，不构成法律意见」。
"""

FORCE_ANSWER_PROMPT = (
    "工具调用轮数预算已用尽。请立即基于目前已获得的工具结果给出最终答案，不要再调用任何工具；"
    "无法确定的信息必须明确说明，不得编造条号或金额。"
)

DEFAULT_MAX_STEPS = 6


class AgentState(TypedDict, total=False):
    """两种实现共用的状态。字段尽量扁平，便于序列化与在 LangGraph 里流转。"""

    question: str
    framework: str
    mock: bool
    messages: list[dict[str, Any]]
    iterations: int
    reflections: int
    pending_calls: list[dict[str, str]]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    draft_answer: str | None
    final_answer: str | None
    force_answer: bool
    forced: bool
    error: str | None
    trace: list[dict[str, Any]]


@dataclass
class AgentResult:
    """一次 Agent 运行的完整结果（含可回放的 trace）。"""

    question: str
    answer: str
    framework: str
    mock: bool
    iterations: int
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    error: str | None = None
    forced: bool = False

    @property
    def used_tools(self) -> list[str]:
        """按调用顺序去重后的工具名列表。"""
        names: list[str] = []
        for call in self.tool_calls:
            name = call.get("name")
            if isinstance(name, str) and name not in names:
                names.append(name)
        return names

    @classmethod
    def from_state(cls, state: AgentState, *, forced: bool = False) -> "AgentResult":
        return cls(
            question=state.get("question", ""),
            answer=state.get("final_answer") or "",
            framework=state.get("framework", "unknown"),
            mock=bool(state.get("mock")),
            iterations=int(state.get("iterations", 0)),
            tool_calls=list(state.get("tool_calls", [])),
            tool_results=list(state.get("tool_results", [])),
            trace=list(state.get("trace", [])),
            messages=list(state.get("messages", [])),
            error=state.get("error"),
            forced=forced or bool(state.get("forced")),
        )

    def to_dict(self, *, include_messages: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "framework": self.framework,
            "mock": self.mock,
            "iterations": self.iterations,
            "used_tools": self.used_tools,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "trace": self.trace,
            "error": self.error,
            "forced": self.forced,
        }
        if include_messages:
            data["messages"] = self.messages
        return data


# ---------------------------------------------------------------------------
# 状态初始化与 trace
# ---------------------------------------------------------------------------


def new_state(question: str, framework: str) -> AgentState:
    """创建初始状态：system prompt + user 问题 + 空 trace。"""
    return AgentState(
        question=question,
        framework=framework,
        mock=False,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        iterations=0,
        reflections=0,
        pending_calls=[],
        tool_calls=[],
        tool_results=[],
        draft_answer=None,
        final_answer=None,
        force_answer=False,
        forced=False,
        error=None,
        trace=[],
    )


def add_trace(state: AgentState, event_type: str, title: str, **detail: Any) -> dict[str, Any]:
    """追加一条 trace 事件并返回它（detail 需要可 JSON 序列化）。"""
    event: dict[str, Any] = {
        "index": len(state.get("trace", [])) + 1,
        "type": event_type,
        "title": title,
    }
    if detail:
        event["detail"] = detail
    state.setdefault("trace", []).append(event)
    return event


def assistant_message(response: LLMResponse) -> dict[str, Any]:
    """把模型回复转成 OpenAI 兼容的 assistant 消息。"""
    message: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
    if response.tool_calls:
        message["tool_calls"] = [call.to_openai_dict() for call in response.tool_calls]
    return message


def pending_calls_from_response(response: LLMResponse) -> list[dict[str, str]]:
    """把模型回复里的工具调用转成内部 pending 结构。"""
    return [
        {"id": call.id, "name": call.name, "arguments_json": call.arguments_json}
        for call in response.tool_calls
    ]


# ---------------------------------------------------------------------------
# 工具执行（两种实现共用）
# ---------------------------------------------------------------------------


def _arguments_preview(arguments_json: str) -> Any:
    """trace 里展示的参数：能解析就展示对象，否则展示原始字符串。"""
    try:
        parsed = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return arguments_json
    return parsed


def execute_pending_calls(state: AgentState) -> None:
    """执行 ``state['pending_calls']`` 中的全部工具调用，并回填消息 / 结果 / trace。

    工具失败不会中断流程：``execute_tool_call`` 永远返回结构化信封，错误会作为
    tool 消息交还给模型，由模型决定下一步。
    """
    calls = list(state.get("pending_calls") or [])
    if not calls:
        return
    state["pending_calls"] = []
    for call in calls:
        call_id = str(call.get("id") or f"call_{len(state['tool_results']) + 1}")
        name = str(call.get("name") or "")
        arguments_json = str(call.get("arguments_json") or "")
        preview = _arguments_preview(arguments_json)
        payload = tools.execute_tool_call(name, arguments_json)
        state.setdefault("tool_results", []).append(payload)
        state.setdefault("tool_calls", []).append(
            {"id": call_id, "name": name, "arguments": preview}
        )
        state.setdefault("messages", []).append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(payload, ensure_ascii=False),
            }
        )
        add_trace(
            state,
            "tool",
            f"工具返回：{name}",
            name=name,
            arguments=preview,
            ok=bool(payload.get("ok")),
            result=tools.tool_summary(payload),
        )


# ---------------------------------------------------------------------------
# 兜底与收尾
# ---------------------------------------------------------------------------


def request_forced_answer(llm: BaseLLM, state: AgentState) -> str | None:
    """轮数耗尽后，再请求一次「禁止调用工具」的最终回答；模型仍要调工具则返回 None。"""
    messages = list(state.get("messages", [])) + [
        {"role": "system", "content": FORCE_ANSWER_PROMPT}
    ]
    try:
        response = llm.complete(messages, tools=[])
    except LLMError as exc:
        add_trace(state, "error", "强制收尾时模型调用失败", message=str(exc))
        return None
    if response.tool_calls:
        return None
    content = (response.content or "").strip()
    return content or None


def fallback_answer(state: AgentState) -> str:
    """没有拿到模型最终回答时的确定性兜底：把已有工具结果如实汇报给用户。"""
    lines = [
        "（系统兜底）模型未能在预算内给出最终答案。以下是本轮已获取的工具结果摘要，仅供参考：",
    ]
    results = state.get("tool_results") or []
    if not results:
        lines.append("- 本轮没有成功执行任何工具。")
    for payload in results:
        tool = payload.get("tool", "unknown")
        if not payload.get("ok"):
            error = payload.get("error") or {}
            lines.append(f"- `{tool}` 执行失败：{error.get('message', '未知错误')}")
            continue
        data = payload.get("data") or {}
        if tool == tools.CALC_TOOL_NAME and data.get("summary"):
            lines.append(f"- {data['summary']}")
        elif tool == tools.SEARCH_TOOL_NAME:
            titles = "；".join(
                item["title"] for item in (data.get("results") or [])[:3]
            )
            lines.append(f"- 检索到 {data.get('count', 0)} 条法条：{titles or '无'}")
        elif tool == tools.VERIFY_TOOL_NAME:
            lines.append(f"- 引用核验结论：{data.get('status')}")
    lines.append("建议补充月薪、工作年限、加班小时数，或明确要引用的法条后重试。")
    return "\n".join(lines)


def finish(state: AgentState, answer: str | None, *, forced: bool = False) -> AgentResult:
    """写入最终回答、追加 answer trace，并返回 :class:`AgentResult`。"""
    text = (answer or "").strip()
    if not text:
        text = fallback_answer(state)
        forced = True
    state["final_answer"] = text
    state["forced"] = forced
    state.setdefault("messages", []).append({"role": "assistant", "content": text})
    add_trace(
        state,
        "answer",
        "最终回答",
        framework=state.get("framework"),
        mock=bool(state.get("mock")),
        forced=forced,
        answer_chars=len(text),
    )
    return AgentResult.from_state(state, forced=forced)


def validate_question(question: str) -> str:
    """校验并规整用户问题。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    return question.strip()
