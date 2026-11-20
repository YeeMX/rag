"""支持 ``python -m labor_agent`` 直接进入 CLI。"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
