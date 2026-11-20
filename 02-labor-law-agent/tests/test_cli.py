"""CLI 端到端测试：ask / demo / graph 子命令，以及 ``python -m`` 入口。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from labor_agent.cli import main
from labor_agent.langgraph_agent import is_langgraph_available

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTION = "公司违法辞退我，月薪12000，工作3年7个月，能赔多少？"


def test_ask_mock_prints_full_trace(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ask", QUESTION, "--no-color"])
    out = capsys.readouterr().out
    assert code == 0
    assert "MOCK 离线模拟" in out
    assert "工具返回：calc_severance" in out
    assert "96000.00" in out
    assert "💬 最终回答" in out


def test_ask_json_output_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ask", QUESTION, "--json"])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert data["framework"] == "function_calling"
    assert data["mock"] is True
    assert "96000.00" in data["answer"]
    assert data["tool_calls"]
    assert data["messages"]


@pytest.mark.skipif(not is_langgraph_available(), reason="langgraph 未安装")
def test_ask_langgraph_framework(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ask", QUESTION, "--framework", "langgraph", "--no-color"])
    out = capsys.readouterr().out
    assert code == 0
    assert "langgraph" in out
    assert "96000.00" in out


def test_demo_runs_five_questions_and_summary(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["demo", "--no-color"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("━━━ Demo ") == 5
    assert "Demo 汇总" in out
    assert "96000.00 元" in out
    assert "引用核验: partial" in out
    assert "不是真实大模型回答" in out


def test_graph_subcommand_prints_mermaid(capsys: pytest.CaptureFixture[str]) -> None:
    if not is_langgraph_available():
        pytest.skip("langgraph 未安装")
    code = main(["graph"])
    out = capsys.readouterr().out
    assert code == 0
    assert "graph TD" in out
    for node in ("planner", "tools", "reflect", "answer"):
        assert node in out


def test_openai_mode_without_key_fails_friendly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(["ask", QUESTION, "--openai"])
    err = capsys.readouterr().err
    assert code == 2
    assert "OPENAI_API_KEY" in err


def test_invalid_max_steps_returns_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ask", QUESTION, "--max-steps", "0"])
    err = capsys.readouterr().err
    assert code == 2
    assert "max-steps" in err


def test_python_dash_m_entry_point() -> None:
    """真实子进程验证 ``python -m labor_agent.cli ask`` 可运行（README 里的命令）。"""
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "labor_agent.cli",
            "ask",
            QUESTION,
            "--no-color",
            "--json",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["framework"] == "function_calling"
    assert "96000.00" in data["answer"]
