"""pytest 公共 fixture。

单元测试不依赖真实模型/网络：用 HashEmbedder + 内存/临时目录的 Chroma + MockLLM，
保证 `pytest` 在任何机器上几秒内跑完。
集成测试（需要 BGE）通过 `pytest -m integration` 单独跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lrs.schemas import Chunk, LegalArticle  # noqa: E402

SAMPLE_LAW = "中华人民共和国劳动合同法"


def make_article(num: int, text: str, law_id: str = "lcl") -> LegalArticle:
    law = SAMPLE_LAW if law_id == "lcl" else "中华人民共和国劳动法"
    return LegalArticle(
        id=f"{law_id}:第{num}条",
        law_id=law_id,
        law=law,
        chapter="第四章  劳动合同的解除和终止",
        article_no=f"第{num}条",
        article_num=num,
        title=f"《{law.replace('中华人民共和国', '')}》第{num}条",
        text=text,
        paragraphs=[text],
    )


@pytest.fixture()
def tiny_chunks() -> list[Chunk]:
    """3 条法条的迷你语料，用于检索/融合的单元测试。"""
    arts = [
        make_article(47, "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。"),
        make_article(87, "用人单位违反本法规定解除或者终止劳动合同的，应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。"),
        make_article(82, "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。"),
        make_article(44, "有下列情形之一的，用人单位应当按照下列标准支付高于劳动者正常工作时间工资的工资报酬：（一）安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬；", law_id="ll"),
    ]
    chunks = []
    for i, a in enumerate(arts):
        chunks.append(Chunk(
            chunk_id=a.id, article_id=a.id, law=a.law, article_no=a.article_no,
            chapter=a.chapter, text=f"{a.title}：{a.text}", article_text=a.text, order=i,
        ))
    return chunks


@pytest.fixture()
def benchmark_cases() -> list[dict]:
    import json

    path = PROJECT_ROOT / "data" / "benchmark.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
