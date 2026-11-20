"""可复现评测入口：一条命令跑「消融 + 答案层指标」。

用法：
    python scripts/run_eval.py                 # 5 种配置消融 + 答案生成（默认 Mock LLM）
    python scripts/run_eval.py --limit 5       # 冒烟
    python scripts/run_eval.py --real-llm      # 使用真实 LLM（需 OPENAI_API_KEY）
    python scripts/run_eval.py --judge         # 额外跑 LLM-as-judge

实际逻辑在 scripts/evaluate.py（可以直接调用那里做更细粒度的控制）。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import lrs.env  # noqa: F401  必须在 sentence-transformers 之前导入（关闭 hf-xet）
import evaluate  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if "--answers" not in sys.argv and "--skip-ablations" not in sys.argv:
        sys.argv.append("--answers")
    return evaluate.main()


if __name__ == "__main__":
    raise SystemExit(main())
