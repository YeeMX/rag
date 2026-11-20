"""语料自举：法条 JSONL 缺失时，从公开仓库下载四部法律全文并重新解析。

正常情况下 `data/corpus/labor_law_articles.jsonl` 已随仓库分发，不需要运行本脚本。
只有在语料损坏 / 只想保留原始文本时使用：

    python data/corpus/bootstrap.py          # 缺文件才下载
    python data/corpus/bootstrap.py --force  # 强制重新下载并解析

数据来源（公开整理）：https://github.com/FFMRyan/Chinese-Labor-Law
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
JSONL = HERE / "labor_law_articles.jsonl"
BASE = "https://raw.githubusercontent.com/FFMRyan/Chinese-Labor-Law/master/"

LAWS = [
    "中华人民共和国劳动合同法",
    "中华人民共和国劳动法",
    "中华人民共和国劳动争议调解仲裁法",
    "中华人民共和国社会保险法",
]


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for law in LAWS:
        dest = RAW_DIR / f"{law}.md"
        if dest.exists() and not force:
            print(f"已存在，跳过：{dest.name}")
            continue
        url = BASE + urllib.parse.quote(f"{law}.md")
        print(f"下载 {url}")
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            dest.write_bytes(resp.read())
        print(f"  -> {dest} ({dest.stat().st_size / 1024:.1f} KB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    args = parser.parse_args()
    try:
        if JSONL.exists() and not args.force:
            print(f"语料已存在：{JSONL}，无需下载。加 --force 可强制重建。")
            return 0
        download(force=args.force)
        print("解析法条 ...")
        return subprocess.call([sys.executable, str(HERE / "parse_laws.py")])
    except Exception as e:  # pragma: no cover - 网络异常
        print(f"下载失败：{e}\n可手动把四部法律的 Markdown 放到 {RAW_DIR} 后运行 parse_laws.py",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
