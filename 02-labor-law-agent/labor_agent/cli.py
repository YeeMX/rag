"""命令行入口：``python -m labor_agent.cli ask/demo/graph``。

默认是 **Mock 模式**（离线规则模拟，不需要 API key），保证任何环境都能完整演示。
``--openai`` 才会调用真实模型（需要 ``OPENAI_API_KEY``）。

输出设计目标：trace 好看、适合截图放到 README / 面试时现场演示。
颜色仅在终端（isatty）且未设置 ``NO_COLOR`` 时启用；重定向到文件时是纯文本。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from typing import Any, Sequence

from .function_calling import run_function_calling
from .langgraph_agent import (
    LangGraphUnavailable,
    graph_mermaid,
    is_langgraph_available,
    run_langgraph,
)
from .llm import (
    BaseLLM,
    LLMError,
    LLMUnavailableError,
    MockLLM,
    build_llm,
)
from .state import AgentResult

RULE = "─" * 74


# ---------------------------------------------------------------------------
# 终端输出小工具
# ---------------------------------------------------------------------------


def _configure_stdout() -> None:
    """Windows 控制台默认 GBK，含中文输出前必须切到 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - pytest 捕获对象等非标准流
            pass


class Palette:
    """极简 ANSI 配色（不依赖 rich，重定向时自动禁用）。"""

    CODES = {
        "bold": "1",
        "dim": "2",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
    }

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        prefix = "".join(f"\033[{self.CODES.get(style, style)}m" for style in styles)
        return f"{prefix}{text}\033[0m"


def _display_width(text: str) -> int:
    """终端显示宽度（中文算 2 列），用于对齐表格。"""
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def _clip(text: str, limit: int) -> str:
    """按显示宽度截断（中文安全），超出加省略号。"""
    text = " ".join(str(text).split())
    if _display_width(text) <= limit:
        return text
    out = ""
    used = 0
    for char in text:
        char_width = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if used + char_width > limit - 1:
            break
        out += char
        used += char_width
    return out + "…"


# ---------------------------------------------------------------------------
# trace 渲染
# ---------------------------------------------------------------------------


def _render_tool_result(detail: dict[str, Any], color: Palette) -> list[str]:
    name = detail.get("name", "")
    result = detail.get("result") or {}
    ok = bool(detail.get("ok"))
    lines: list[str] = []
    if not ok:
        error = result.get("error") or {}
        lines.append(
            color(f"    ❌ 失败 [{error.get('code', 'error')}] {error.get('message', '')}", "red")
        )
        return lines
    if name == "search_labor_law":
        lines.append(color(f"    ✓ 命中 {result.get('count', 0)} 条法条", "green"))
        for item in result.get("top") or []:
            score_text = f"BM25={item.get('score')}"
            lines.append(
                f"    · {color(str(item.get('title', '')), 'bold')} "
                f"{color(score_text, 'dim')}"
            )
            lines.append(f"      {_clip(item.get('snippet', ''), 88)}")
    elif name == "calc_severance":
        lines.append(color(f"    ✓ {result.get('summary', '')}", "green"))
        for step in result.get("steps") or []:
            lines.append(
                f"    {step.get('step')}. {step.get('label', '')}："
                f"{_clip(step.get('formula', ''), 88)}"
            )
        for warning in result.get("warnings") or []:
            lines.append(color(f"    ⚠ {warning}", "yellow"))
    elif name == "verify_citation":
        status = str(result.get("status", ""))
        status_color = {"verified": "green", "partial": "yellow", "not_found": "red"}.get(
            status, "white"
        )
        lines.append(
            color(f"    ✓ 核验结论：{status}", status_color)
            + f"｜条号存在={result.get('article_found')}｜"
            + f"要素 {result.get('matched')}/{result.get('checked')}"
        )
        for reason in result.get("reasons") or []:
            lines.append(f"      · {_clip(reason, 90)}")
    else:  # pragma: no cover - 兜底展示
        lines.append("    " + _clip(json.dumps(result, ensure_ascii=False), 120))
    return lines


def _render_planner_trace(event: dict[str, Any], color: Palette) -> list[str]:
    detail = event.get("detail") or {}
    lines: list[str] = []
    content = detail.get("content")
    if content:
        label = "思考" if detail.get("tool_calls") else "模型输出"
        lines.append(f"    {color(label, 'cyan')}：{_clip(content, 110)}")
    for call in detail.get("tool_calls") or []:
        arguments = call.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        lines.append(
            f"    {color('→ 工具调用', 'magenta')} {call.get('name')}"
            f"({_clip(arguments, 96)})"
        )
    if not lines:
        lines.append(color("    （本轮未返回文本与工具调用）", "dim"))
    return lines


def _render_trace_event(event: dict[str, Any], color: Palette) -> list[str]:
    index = event.get("index", "?")
    event_type = event.get("type")
    title = str(event.get("title", ""))
    detail = event.get("detail") or {}
    lines: list[str] = []

    if event_type == "start":
        icon = "🚀"
        lines.append(f"[{index}] {icon} {color(title, 'bold', 'blue')}")
        if detail.get("llm"):
            lines.append(f"    LLM：{detail['llm']}")
        lines.append(
            f"    框架：{detail.get('framework')}｜最大轮数：{detail.get('max_steps')}"
        )
    elif event_type == "planner":
        icon = "🧠"
        lines.append(f"[{index}] {icon} {color(title, 'bold', 'cyan')}")
        lines.extend(_render_planner_trace(event, color))
    elif event_type == "tool":
        icon = "🛠"
        lines.append(f"[{index}] {icon} {color(title, 'bold', 'magenta')}")
        arguments = detail.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        lines.append(f"    参数：{_clip(arguments, 110)}")
        lines.extend(_render_tool_result(detail, color))
    elif event_type == "reflect":
        icon = "🔁"
        lines.append(f"[{index}] {icon} {color(title, 'bold', 'yellow')}")
        lines.append(
            f"    reflections={detail.get('reflections')}｜max_steps={detail.get('max_steps')}"
        )
    elif event_type == "answer":
        icon = "✅"
        lines.append(
            f"[{index}] {icon} {color(title, 'bold', 'green')}"
            f"（{detail.get('answer_chars', 0)} 字，forced={detail.get('forced')}）"
        )
    elif event_type == "error":
        icon = "❌"
        lines.append(f"[{index}] {icon} {color(title, 'bold', 'red')}")
        if detail.get("message"):
            lines.append(f"    {_clip(str(detail['message']), 110)}")
        if detail.get("code"):
            lines.append(f"    code={detail['code']}")
    else:  # pragma: no cover - 未知事件类型
        lines.append(f"[{index}] {title}")
    return lines


def render_result(
    result: AgentResult,
    *,
    color: Palette | None = None,
    show_trace: bool = True,
) -> str:
    """把一个 :class:`AgentResult` 渲染成适合终端阅读 / 截图的文本。"""
    color = color or Palette(False)
    lines: list[str] = [color(RULE, "dim")]
    mode = (
        color("MOCK 离线模拟（规则引擎，非真实大模型）", "yellow")
        if result.mock
        else color("真实模型", "green")
    )
    lines.append(f"{color('❓ 问题', 'bold')}：{result.question}")
    lines.append(
        f"{color('🧭 框架', 'bold')}：{result.framework}｜模式：{mode}"
        f"｜模型轮数：{result.iterations}｜工具调用：{len(result.tool_calls)} 次"
    )
    lines.append(color(RULE, "dim"))
    if show_trace:
        for event in result.trace:
            lines.extend(_render_trace_event(event, color))
        lines.append(color(RULE, "dim"))
    lines.append(color("💬 最终回答", "bold", "green"))
    lines.append(result.answer if result.answer else "（无最终回答）")
    lines.append(color(RULE, "dim"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# demo 题库
# ---------------------------------------------------------------------------

DEMO_QUESTIONS: list[dict[str, Any]] = [
    {
        "question": "公司违法辞退我，月薪12000，工作3年7个月，能赔多少？依据哪一条？",
        "framework": "function_calling",
        "note": "一次回复发起 2 个工具调用：精算 2N + 检索法条",
    },
    {
        "question": "公司经济性裁员，我月薪30000，工作15年，当地社平工资9000，经济补偿怎么算？",
        "framework": "function_calling",
        "note": "触发第 47 条高薪封顶：基数按社平 3 倍、年限最高 12 年",
    },
    {
        "question": "入职14个月一直没签书面劳动合同，月薪15000，能主张多少二倍工资差额？",
        "framework": "function_calling",
        "note": "第 82 条：二倍工资差额，最多 11 个月",
    },
    {
        "question": "我工作日加班10小时、休息日加班8小时、法定节假日加班4小时，月薪12000，加班费多少？",
        "framework": "function_calling",
        "note": "劳动法第 44 条：1.5 倍 / 2 倍 / 3 倍三档加班费",
    },
    {
        "question": "帮我核验这条引用是否准确：《劳动合同法》第八十二条说未签书面合同最多支付11个月二倍工资？",
        "framework": "langgraph",
        "note": "引用核验：条号存在，但「11 个月」与原文对不上 → partial",
    },
]


# ---------------------------------------------------------------------------
# 参数与运行
# ---------------------------------------------------------------------------


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--framework",
        choices=["auto", "function_calling", "langgraph"],
        default="auto",
        help="Agent 实现：function_calling（方案 A）/ langgraph（方案 B）/ auto（默认；ask 用方案 A，demo 自动混用）",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--mock",
        action="store_true",
        help="强制离线 Mock 模式（默认行为，不需要 API key）",
    )
    group.add_argument(
        "--openai",
        action="store_true",
        help="使用真实模型（需 OPENAI_API_KEY，可选 OPENAI_BASE_URL / OPENAI_MODEL）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="模型决策轮数上限（默认 function_calling=6，langgraph=4）",
    )
    parser.add_argument("--no-color", action="store_true", help="关闭 ANSI 颜色")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m labor_agent.cli",
        description="劳动法咨询 Agent：3 个工具 + function calling / LangGraph 两种实现",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="问一个劳动法问题")
    ask.add_argument("question", help="问题文本，例如：公司违法辞退我，月薪12000，工作3年7个月，能赔多少？")
    _add_common_flags(ask)
    ask.add_argument("--json", action="store_true", help="输出完整 JSON（含 messages/tool_results），便于调试")

    demo = sub.add_parser("demo", help="跑 5 个内置问题并打印完整 trace")
    _add_common_flags(demo)
    demo.add_argument("--json", action="store_true", help="以 JSON Lines 输出每个结果")

    graph = sub.add_parser("graph", help="打印 LangGraph 图的 mermaid 定义")
    graph.add_argument("--max-steps", type=int, default=4, help="图中标注的轮数上限")
    return parser


def _make_color(no_color: bool) -> Palette:
    enabled = (
        sys.stdout.isatty()
        and not no_color
        and not os.getenv("NO_COLOR")
    )
    return Palette(enabled)


def _make_llm(use_openai: bool) -> BaseLLM:
    if use_openai:
        return build_llm("openai")
    return MockLLM()  # 默认永远走离线 mock，避免误用真实 API 产生费用


def _resolve_max_steps(raw: int | None, framework: str) -> int:
    if raw is not None:
        if raw < 1:
            raise ValueError("--max-steps 必须 >= 1")
        return raw
    return 4 if framework == "langgraph" else 6


def _run(question: str, framework: str, llm: BaseLLM, max_steps: int) -> AgentResult:
    if framework == "langgraph":
        return run_langgraph(question, llm, max_steps=max_steps)
    return run_function_calling(question, llm, max_steps=max_steps)


def _cmd_ask(args: argparse.Namespace) -> int:
    color = _make_color(args.no_color)
    try:
        llm = _make_llm(args.openai)
    except LLMUnavailableError as exc:
        print(color(f"❌ 无法启用真实模型：{exc}", "red"), file=sys.stderr)
        return 2
    framework = "function_calling" if args.framework == "auto" else args.framework
    if framework == "langgraph" and not is_langgraph_available():
        print(
            color(
                "❌ 当前环境未安装 langgraph，无法运行方案 B；"
                "请改用 --framework function_calling（功能等价）。",
                "red",
            ),
            file=sys.stderr,
        )
        return 2
    try:
        max_steps = _resolve_max_steps(args.max_steps, framework)
        result = _run(args.question, framework, llm, max_steps)
    except ValueError as exc:
        print(color(f"❌ 参数错误：{exc}", "red"), file=sys.stderr)
        return 2
    except LLMError as exc:
        print(color(f"❌ 模型调用失败：{exc}", "red"), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result.to_dict(include_messages=True), ensure_ascii=False, indent=2))
    else:
        print(render_result(result, color=color))
    return 0 if result.error is None else 1


def _run_one_demo(
    item: dict[str, Any],
    llm: BaseLLM,
    *,
    forced_framework: str | None,
    max_steps_raw: int | None,
    index: int,
) -> tuple[AgentResult | None, str, str | None]:
    """跑单个 demo 问题；返回 (结果, 实际框架, 降级说明)。"""
    framework = forced_framework or item.get("framework", "function_calling")
    note = None
    if framework == "langgraph" and not is_langgraph_available():
        note = "langgraph 不可用，已自动降级为 function_calling"
        framework = "function_calling"
    try:
        max_steps = _resolve_max_steps(max_steps_raw, framework)
        result = _run(str(item["question"]), framework, llm, max_steps)
        return result, framework, note
    except LLMError as exc:  # 单个问题失败不影响其它问题
        print(f"❌ demo 第 {index} 题模型调用失败：{exc}", file=sys.stderr)
        return None, framework, note


def _cmd_demo(args: argparse.Namespace) -> int:
    color = _make_color(args.no_color)
    try:
        llm = _make_llm(args.openai)
    except LLMUnavailableError as exc:
        print(color(f"❌ 无法启用真实模型：{exc}", "red"), file=sys.stderr)
        return 2
    forced_framework = None if args.framework == "auto" else args.framework
    if forced_framework == "langgraph" and not is_langgraph_available():
        print(
            color("❌ 当前环境未安装 langgraph，无法强制使用方案 B。", "red"),
            file=sys.stderr,
        )
        return 2

    print(color("=" * 74, "dim"))
    print(color("劳动法咨询 Agent · Demo", "bold", "cyan"))
    print(f"LLM：{llm.describe()}")
    print(
        f"模式：{'MOCK 离线模拟（不联网、不需要 API key）' if llm.is_mock else '真实模型'}"
        f"｜问题数：{len(DEMO_QUESTIONS)}"
    )
    print("覆盖：法条检索 / 赔偿计算 / 引用核验 3 个工具 + function calling / LangGraph 2 种实现")
    print(color("=" * 74, "dim"))

    results: list[tuple[int, AgentResult | None, str, str]] = []
    for index, item in enumerate(DEMO_QUESTIONS, start=1):
        print()
        print(color(f"━━━ Demo {index}/{len(DEMO_QUESTIONS)}：{item['note']} ━━━", "bold", "blue"))
        result, framework, downgrade = _run_one_demo(
            item,
            llm,
            forced_framework=forced_framework,
            max_steps_raw=args.max_steps,
            index=index,
        )
        if downgrade:
            print(color(f"⚠ {downgrade}", "yellow"))
        if result is None:
            continue
        results.append((index, result, framework, downgrade or ""))
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False))
        else:
            print(render_result(result, color=color))

    print()
    print(color("=" * 74, "dim"))
    print(color("Demo 汇总（真实运行结果）", "bold", "cyan"))
    header = (
        _pad("#", 3)
        + _pad("框架", 18)
        + _pad("工具调用", 34)
        + _pad("轮数", 5)
        + _pad("金额/结论", 40)
    )
    print(color(header, "bold"))
    for index, result, framework, _ in results:
        tools_text = ", ".join(result.used_tools) or "（无）"
        conclusion = _conclusion_of(result)
        row = (
            _pad(str(index), 3)
            + _pad(framework, 18)
            + _pad(_clip(tools_text, 32), 34)
            + _pad(str(result.iterations), 5)
            + _clip(conclusion, 40)
        )
        print(row)
    print(color("=" * 74, "dim"))
    if llm.is_mock:
        print(
            color(
                "提示：以上为 MOCK 离线模拟输出，用于演示工具调用链路，不是真实大模型回答。",
                "yellow",
            )
        )
        print(
            "     设置 OPENAI_API_KEY 并加 --openai 可切换真实模型："
            "python -m labor_agent.cli demo --openai"
        )
    return 0


def _conclusion_of(result: AgentResult) -> str:
    """从工具结果里提炼一句结论，用于 demo 汇总表。"""
    for payload in result.tool_results:
        data = payload.get("data") or {}
        if payload.get("tool") == "calc_severance" and data.get("total") is not None:
            return f"{data.get('total', 0):.2f} 元"
        if payload.get("tool") == "verify_citation" and data.get("status"):
            return f"引用核验: {data['status']}"
    if result.answer:
        first_line = next(
            (line for line in result.answer.splitlines() if line.strip()),
            "",
        )
        return first_line
    return "（无结论）"


def _cmd_graph(args: argparse.Namespace) -> int:
    if not is_langgraph_available():
        print(
            "❌ 当前环境未安装 langgraph，无法输出图结构；请改用方案 A（function_calling）。",
            file=sys.stderr,
        )
        return 2
    try:
        print(graph_mermaid(max_steps=args.max_steps))
    except LangGraphUnavailable as exc:  # pragma: no cover - 竞态兜底
        print(f"❌ {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdout()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "ask":
        return _cmd_ask(args)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "graph":
        return _cmd_graph(args)
    parser.print_help(sys.stderr)  # pragma: no cover - argparse 已保证有子命令
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
