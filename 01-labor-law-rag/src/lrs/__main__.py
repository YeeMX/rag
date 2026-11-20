"""`python -m lrs` 入口：等价于 scripts/cli.py。"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent          # .../src/lrs
_PROJECT = _SRC.parents[1]                      # 项目根目录
if str(_PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "scripts"))


def main() -> int:
    import runpy

    cli = _PROJECT / "scripts" / "cli.py"
    sys.argv = [str(cli), *sys.argv[1:]]
    runpy.run_path(str(cli), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
