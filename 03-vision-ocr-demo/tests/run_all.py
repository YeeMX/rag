"""一次运行 tests/ 下全部 unittest 用例。

用法（在 03-vision-ocr-demo 目录下，vision 环境即可；不加载 paddle/torch 模型）::

    python tests/run_all.py

退出码 0 表示全部通过。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent


def main() -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    sys.path.insert(0, str(PROJECT_ROOT))
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(TESTS_DIR), pattern="test_*.py", top_level_dir=str(TESTS_DIR))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    print("=" * 60)
    print(f"用例总数: {result.testsRun}，失败: {len(result.failures)}，错误: {len(result.errors)}，"
          f"跳过: {len(result.skipped)}")
    print("=" * 60)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
