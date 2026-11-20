"""把原始 Markdown 法条文本解析成结构化 JSONL。

数据来源：https://github.com/FFMRyan/Chinese-Labor-Law （公开整理的四部劳动法系法律全文）
原始文件位于 raw/，本脚本只做清洗与结构化，不修改法条文字本身。

输出：labor_law_articles.jsonl，每行一条法条，字段：
    id           全局唯一 id，形如 lcl:第39条（lcl=劳动合同法）
    law_id       法律英文短 id
    law          法律全称
    chapter      所属章/节（若有）
    article_no   条号，形如 第三十九条
    article_num  阿拉伯数字条号
    title        检索用标题（法律 + 条号）
    text         法条正文（已去掉 Markdown 与 &emsp; 等排版字符）
    paragraphs   正文按自然段切分的列表
    char_len     正文字符数

用法：
    python parse_laws.py            # 解析 raw/ 下全部 .md，写入 labor_law_articles.jsonl
    python parse_laws.py --stats    # 只打印统计信息
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
OUT_PATH = Path(__file__).resolve().parent / "labor_law_articles.jsonl"

# 法律全称 -> (law_id, 短名)
LAW_META: dict[str, tuple[str, str]] = {
    "中华人民共和国劳动合同法": ("lcl", "劳动合同法"),
    "中华人民共和国劳动法": ("ll", "劳动法"),
    "中华人民共和国劳动争议调解仲裁法": ("lda", "劳动争议调解仲裁法"),
    "中华人民共和国社会保险法": ("sil", "社会保险法"),
}

CN_NUM = "一二三四五六七八九十百零〇两"
ARTICLE_RE = re.compile(rf"\*\*(第[{CN_NUM}]+条)\*\*\s*(.*)$")
CHAPTER_RE = re.compile(r"^#{2,6}\s+(.+)$")
# 章节标题行，如 “第六章  监督检查”“第一节  集体合同”
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百]+[章节]\s*")


def cn_to_int(cn: str) -> int:
    """把 '三十九' / '一百零一' 这类中文数字转成 int。"""
    digits = {c: i for i, c in enumerate("零一二三四五六七八九")}
    digits["〇"] = 0
    digits["两"] = 2
    cn = cn.replace("第", "").replace("条", "")
    if "百" in cn:
        left, _, right = cn.partition("百")
        hundreds = digits.get(left, 1) if left else 1
        val = hundreds * 100
        if right.startswith("零"):
            right = right[1:]
        if right.startswith("十"):
            val += 10 + (digits.get(right[1], 0) if len(right) > 1 else 0)
        elif right:
            val += int("".join(str(digits[c]) for c in right))
        return val
    if "十" in cn:
        left, _, right = cn.partition("十")
        tens = digits.get(left, 1) if left else 1
        return tens * 10 + (digits.get(right, 0) if right else 0)
    return int("".join(str(digits[c]) for c in cn)) if cn else 0


def clean_line(line: str) -> str:
    """去掉 Markdown/HTML 排版字符，保留中文标点。"""
    line = line.replace("&emsp;", "").replace("\u3000", " ").replace("&nbsp;", " ")
    line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)  # **加粗** -> 纯文本
    line = re.sub(r"^#{1,6}\s*", "", line)        # 标题井号
    return line.strip()


@dataclass
class Article:
    id: str
    law_id: str
    law: str
    chapter: str
    article_no: str
    article_num: int
    title: str
    text: str
    paragraphs: list[str]
    char_len: int


def parse_law(path: Path) -> tuple[str, str, list[Article], str]:
    """解析单部法律，返回 (law, law_id, 法条列表, 法律沿革说明)。"""
    law = path.stem
    law_id, short = LAW_META.get(law, (law, law))
    lines = path.read_text(encoding="utf-8").splitlines()

    articles: list[Article] = []
    chapter = ""
    current: dict | None = None
    preamble: list[str] = []
    in_toc = False

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = re.sub(r"\s+", " ", " ".join(current["paragraphs"])).strip()
        text = re.sub(r"\s+([，。；：、？！])", r"\1", text)
        paragraphs = [p for p in current["paragraphs"] if p]
        num = cn_to_int(current["article_no"])
        articles.append(
            Article(
                id=f"{law_id}:{current['article_no']}",
                law_id=law_id,
                law=law,
                chapter=current["chapter"],
                article_no=current["article_no"],
                article_num=num,
                title=f"《{short}》{current['article_no']}",
                text=text,
                paragraphs=paragraphs,
                char_len=len(text),
            )
        )
        current = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        # 目录区：跳过（目录行形如 “目　　录” / “目 录” / 带 &emsp; 前缀）
        plain = re.sub(r"\s+", "", clean_line(line))
        if plain == "目录":
            in_toc = True
            continue
        if in_toc:
            # 目录里的章标题没有 "#" 前缀；正文里的章标题是 Markdown 标题
            if CHAPTER_RE.match(line):
                in_toc = False  # 进入正文章节
            else:
                continue
        if not preamble and line.startswith("（") and "通过" in line:
            preamble.append(clean_line(line))
            continue
        m_chapter = CHAPTER_RE.match(line)
        if m_chapter:
            flush()
            title = clean_line(m_chapter.group(1))
            if "第一章" in title or "总" in title:
                chapter = title
            else:
                chapter = f"{chapter} / {title}" if chapter and SECTION_RE.match(title) is None else title
            continue
        m_article = ARTICLE_RE.search(line)
        if m_article:
            flush()
            current = {
                "article_no": m_article.group(1),
                "chapter": chapter,
                "paragraphs": [],
            }
            body = clean_line(m_article.group(2))
            if body:
                current["paragraphs"].append(body)
            elif m_article.group(2).strip():
                current["paragraphs"].append(clean_line(m_article.group(2)))
            continue
        if current is None:
            continue  # 目录以外的页眉等噪声
        body = clean_line(line)
        if body and not SECTION_RE.match(body):
            current["paragraphs"].append(body)

    flush()
    return law, law_id, articles, (preamble[0] if preamble else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true", help="只打印统计，不写文件")
    args = parser.parse_args()

    all_articles: list[Article] = []
    provenance = []
    # 固定顺序，保证 jsonl 稳定可复现
    for law in LAW_META:
        path = RAW_DIR / f"{law}.md"
        if not path.exists():
            raise FileNotFoundError(f"缺少原始法律文本：{path}")
        law_name, law_id, articles, preamble = parse_law(path)
        all_articles.extend(articles)
        provenance.append(
            {
                "law": law_name,
                "law_id": law_id,
                "source": f"https://github.com/FFMRyan/Chinese-Labor-Law/blob/master/{law_name}.md",
                "preamble": preamble,
                "article_count": len(articles),
            }
        )

    by_law: dict[str, int] = {}
    for a in all_articles:
        by_law[a.law] = by_law.get(a.law, 0) + 1
    total_chars = sum(a.char_len for a in all_articles)
    avg = total_chars / max(len(all_articles), 1)

    print("== 语料统计 ==")
    for law, n in by_law.items():
        print(f"  {law}: {n} 条")
    print(f"  法条总数: {len(all_articles)}，总字数: {total_chars}，平均每条 {avg:.1f} 字")
    nums = [a.article_num for a in all_articles if a.law_id == "lcl"]
    print(f"  劳动合同法条号连续性: {nums[0]}..{nums[-1]} 共 {len(nums)} 条 "
          f"{'（连续）' if nums == list(range(nums[0], nums[-1] + 1)) else '（存在跳号）'}")

    if args.stats:
        return

    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for a in all_articles:
            f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
    prov_path = OUT_PATH.with_name("provenance.json")
    prov_path.write_text(
        json.dumps(
            {
                "note": "法条原文来自公开整理的法律文本仓库，仅用于检索实验，不作为法律意见。",
                "files": provenance,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {OUT_PATH}（{OUT_PATH.stat().st_size / 1024:.1f} KB）与 {prov_path.name}")


if __name__ == "__main__":
    main()
