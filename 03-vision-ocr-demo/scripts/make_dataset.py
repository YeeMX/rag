"""生成「中文法律文书扫描件」合成评估集（带像素级精确标注）。

用法（在 03-vision-ocr-demo 目录下，用 .venv-vision 解释器）::

    python scripts/make_dataset.py --out data --num 32 --num-train 24 --seed 42

产出目录结构::

    data/
      images/train/*.png      合成文书图片
      images/val/*.png
      labels/train/*.txt      YOLO 格式标注（cls cx cy w h，归一化）
      labels/val/*.txt
      annotations.json        每张图的精确 bbox、区域全文、整图全文、字段 ground truth
      data.yaml               ultralytics 数据集配置

设计要点（面试可讲）：
1. 因为图片是自己「画」出来的，所以每个版面区域的 bbox 是**精确 ground truth**，
   不受人工标注误差影响，适合把「检测 → OCR → 抽取」整条链路的误差拆开看。
2. 版面元素固定四类：``title``（标题）、``paragraph``（正文段落）、``table``（表格）、
   ``stamp``（红色圆形印章），对应劳动合同扫描件最常见的结构。
3. 难例（旋转 2~5°、低对比度 + 噪声、印章遮挡正文、表格跨行合并）全部合成，
   标注方式与正常样本一致，用来观察检测 / OCR 在退化条件下的表现。
4. 生成过程完全由 ``--seed`` 决定，同一 seed 结果可复现。

依赖：Pillow + numpy（两个视觉环境都已安装），不依赖 cv2。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

if sys.platform == "win32":
    try:  # Windows 控制台默认 GBK，中文输出会乱码
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = ["title", "paragraph", "table", "stamp"]
CLASS_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

PAGE_W: int = 1000          # A4 比例 1:1.414（1000x1414 ≈ 120dpi）
PAGE_H: int = 1414
TITLE_SIZE: int = 46
BODY_SIZE: int = 21
TABLE_SIZE: int = 18
STAMP_FONT_SIZE: int = 15

PAGE_BG: tuple[int, int, int] = (252, 252, 252)
TEXT_COLOR: tuple[int, int, int] = (20, 20, 20)
LOW_CONTRAST_TEXT: tuple[int, int, int] = (150, 150, 150)
LOW_CONTRAST_BG: tuple[int, int, int] = (238, 238, 238)
STAMP_RED: tuple[int, int, int] = (198, 32, 40)

FONT_CANDIDATES: list[str] = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
]

#: 正常样本中允许出现的 4 张难例的位置与类型（其余为正常样本）。
#: 约定：0..num_train-1 是 train，之后为 val。这里故意让 train/val 各有 2 张难例。
HARD_CASE_PLAN: dict[int, str] = {
    5: "low_contrast",     # 低对比度 + 噪声（train）
    13: "table_merge",     # 表格跨行合并（train）
    26: "rotation",        # 旋转 2~5°（val）
    30: "stamp_occlusion", # 印章遮挡正文（val）
}

COMPANIES: list[tuple[str, str]] = [
    ("北京华信科技有限公司", "华信科技"),
    ("上海鼎盛商贸有限公司", "鼎盛商贸"),
    ("广州云启信息技术有限公司", "云启信息"),
    ("深圳市恒达建设集团有限公司", "恒达建设"),
    ("杭州明远文化传媒有限公司", "明远传媒"),
    ("成都锦江餐饮管理有限公司", "锦江餐饮"),
    ("武汉长江物流有限公司", "长江物流"),
    ("南京金陵药业股份有限公司", "金陵药业"),
]

PERSON_NAMES: list[str] = [
    "张伟", "李娜", "王强", "刘敏", "陈晨",
    "杨帆", "周婷", "吴昊", "徐静", "孙磊",
]

LEGAL_REPS: list[str] = ["张建国", "李国强", "王志刚", "刘德明", "陈建华", "赵永强"]

ADDRESSES: list[str] = [
    "北京市朝阳区建国路88号院3号楼",
    "上海市浦东新区张江路168号2幢",
    "广州市天河区体育西路103号",
    "深圳市南山区科技园南区18栋",
    "杭州市西湖区文三路90号",
    "成都市武侯区天府大道666号",
    "武汉市江汉区解放大道100号",
    "南京市鼓楼区中山北路26号",
]

CITIES: list[str] = ["北京市", "上海市", "广州市", "深圳市", "杭州市", "成都市"]

JOB_TITLES: list[str] = [
    "软件测试", "行政办公", "财务会计", "人事专员", "新媒体运营", "项目管理",
]

TITLES: list[str] = ["劳动合同书", "劳动合同", "劳动合同文本", "聘用合同书"]

SEAL_TYPES: list[str] = ["合同专用章", "人事专用章", "公章"]

OPTIONAL_CLAUSES: list[str] = [
    "甲方依法为乙方缴纳社会保险费，乙方个人应缴纳部分由甲方从其工资中代扣代缴。",
    "乙方应当遵守甲方依法制定的各项规章制度和劳动纪律，服从甲方管理，履行岗位职责。",
    "乙方应当对甲方的商业秘密及与知识产权相关的保密事项承担保密义务，未经甲方书面同意不得泄露。",
    "甲乙双方解除或终止劳动合同，应当按照《中华人民共和国劳动合同法》的规定办理相关手续。",
]


# ---------------------------------------------------------------------------
# 字体与数据结构
# ---------------------------------------------------------------------------

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def resolve_font_path() -> str:
    """返回第一个存在的中文字体路径；一个都找不到时抛错。"""
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "找不到中文 TrueType 字体，尝试过: " + ", ".join(FONT_CANDIDATES)
    )


def load_font(size: int, font_path: str | None = None) -> ImageFont.FreeTypeFont:
    """加载并缓存指定字号的中文字体（PIL 画中文必须显式 truetype）。"""
    path = font_path or resolve_font_path()
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


@dataclass
class Region:
    """一个版面区域（ground truth）。"""

    cls: str                          # 类别名，见 CLASS_NAMES
    bbox: tuple[float, float, float, float]  # 左上 / 右下，像素坐标
    text: str                         # 区域内渲染的文字（多行用 \n）

    def to_dict(self, region_id: str) -> dict[str, Any]:
        return {
            "id": region_id,
            "class": self.cls,
            "bbox": [round(v, 1) for v in self.bbox],
            "text": self.text,
        }


@dataclass
class ParagraphLines:
    """排版后的一个段落：实际绘制的每一行。"""

    lines: list[str]


# ---------------------------------------------------------------------------
# 文本排版工具
# ---------------------------------------------------------------------------


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    """按像素宽度做中日韩逐字换行（原文里的 ``\\n`` 保留为强制换行）。"""
    out: list[str] = []
    for raw in text.split("\n"):
        if not raw:
            out.append("")
            continue
        cur = ""
        for ch in raw:
            if cur and draw.textlength(cur + ch, font=font) > max_width:
                out.append(cur)
                cur = ch
            else:
                cur += ch
        out.append(cur)
    return out


def line_height(font: ImageFont.FreeTypeFont, extra: int = 3) -> int:
    """单行占用的垂直像素高度（ascent + descent + 额外行距）。"""
    ascent, descent = font.getmetrics()
    return ascent + descent + extra


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    return float(draw.textlength(text, font=font))


def clamp_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """把 bbox 裁剪到图片范围内。"""
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


# ---------------------------------------------------------------------------
# 内容生成：一份劳动合同的随机字段
# ---------------------------------------------------------------------------


def make_id_card(rng: random.Random) -> str:
    """伪造一个 18 位身份证号（仅用于渲染文字，不是真实个人信息）。"""
    region = f"{rng.randint(110000, 659000):06d}"
    birth = f"{rng.randint(1985, 2002):04d}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
    tail = f"{rng.randint(0, 999):03d}{rng.randint(0, 9)}"
    return region + birth + tail


def fmt_date(d: date) -> str:
    return f"{d.year}年{d.month:02d}月{d.day:02d}日"


def build_content(rng: random.Random, *, merged_table: bool = False) -> dict[str, Any]:
    """随机生成一份劳动合同的结构化内容 + 渲染文本。"""
    company, short_name = rng.choice(COMPANIES)
    person = rng.choice(PERSON_NAMES)
    legal_rep = rng.choice(LEGAL_REPS)
    addr_a = rng.choice(ADDRESSES)
    addr_b = rng.choice(ADDRESSES)
    city = rng.choice(CITIES)
    job = rng.choice(JOB_TITLES)
    title = rng.choice(TITLES)
    seal_type = rng.choice(SEAL_TYPES)

    years = rng.choice([1, 2, 3, 5])
    start = date(rng.randint(2022, 2026), rng.randint(1, 12), rng.randint(1, 28))
    # 合同结束日 = 起始日 + N 年 - 1 天（day<=28 时 replace 一定合法）
    end = start.replace(year=start.year + years) - timedelta(days=1)
    sign_date = start - timedelta(days=rng.randint(1, 20))

    salary = rng.choice([6000, 6500, 7000, 8000, 9000, 10000, 12000, 15000, 18000, 20000])
    probation = rng.choice([1, 2, 3, 6])
    probation_salary = int(round(salary * 0.8, -2))
    pay_day = rng.choice([5, 10, 15, 20, 25])

    clauses: list[str] = [
        f"第一条 本合同为固定期限劳动合同，合同期限为{years}年，自{fmt_date(start)}起至{fmt_date(end)}止。",
        f"第二条 乙方同意在甲方从事{job}岗位工作，工作地点为{city}，"
        f"甲方可根据生产经营需要依法调整乙方的工作岗位。",
        f"第三条 甲方于每月{pay_day}日前以货币形式支付乙方工资，"
        f"月工资为人民币{salary}元，且不低于当地最低工资标准。",
        f"第四条 乙方试用期为{probation}个月，试用期月工资为人民币{probation_salary}元，"
        f"试用期包含在劳动合同期限内。",
    ]
    # 额外条款 0~2 条（面试时可以说「版面元素数量随机，避免模型过拟合固定模板」）
    for extra in rng.sample(OPTIONAL_CLAUSES, k=rng.randint(0, 2)):
        clauses.append(f"第{'一二三四五六七八九十'[len(clauses)]}条 {extra}")

    table: dict[str, Any] = {
        "headers": ["项目", "金额（元/月）", "备注"],
        "rows": [
            ["基本工资", f"{int(round(salary * 0.6, -2))}", "按月发放"],
            ["岗位工资", f"{int(round(salary * 0.3, -2))}", "按月发放"],
            ["绩效工资", f"{salary - int(round(salary * 0.6, -2)) - int(round(salary * 0.3, -2))}", "按考核发放"],
            ["合计", f"{salary}", "税前"],
        ],
        "col_fracs": [0.34, 0.36, 0.30],
        "merges": [],
    }
    if merged_table:
        # 难例：备注列第 1、2 行合并为跨行单元格
        table["rows"][1][2] = "按月发放"
        table["merges"] = [(1, 2, 2, 1)]
        table["rows"][2][2] = ""  # 被合并覆盖的单元格

    party_a = [
        f"甲方（用人单位）：{company}",
        f"住　　所：{addr_a}",
        f"法定代表人：{legal_rep}",
    ]
    party_b = [
        f"乙方（劳动者）：{person}",
        f"身份证号：{make_id_card(rng)}",
        f"住　　所：{addr_b}",
    ]
    sign_lines = [
        f"甲方（盖章）：{company}",
        f"乙方（签字）：{person}",
        f"签订日期：{fmt_date(sign_date)}",
    ]

    fields: dict[str, Any] = {
        "party_a": company,
        "party_b": person,
        "contract_years": years,
        "contract_start": fmt_date(start),
        "contract_end": fmt_date(end),
        "monthly_salary": salary,
        "probation_months": probation,
        "probation_salary": probation_salary,
        "job": job,
        "sign_date": fmt_date(sign_date),
        "stamp_company": short_name,
    }

    return {
        "title": title,
        "party_a": party_a,
        "party_b": party_b,
        "clauses": clauses,
        "table": table,
        "sign": sign_lines,
        "stamp": (short_name, seal_type),
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# 渲染器
# ---------------------------------------------------------------------------


class DocumentRenderer:
    """在内存画布上按模板渲染一份文书，并记录每个区域的精确 bbox。"""

    def __init__(self, rng: random.Random, *, low_contrast: bool = False) -> None:
        self.rng = rng
        self.bg = LOW_CONTRAST_BG if low_contrast else PAGE_BG
        self.fg = LOW_CONTRAST_TEXT if low_contrast else TEXT_COLOR
        self.img = Image.new("RGB", (PAGE_W, PAGE_H), self.bg)
        self.draw = ImageDraw.Draw(self.img)
        self.regions: list[Region] = []

    # -- 量测 ---------------------------------------------------------------

    def measure_paragraph(self, raw_lines: Sequence[str], width: int) -> int:
        """估算一个段落块的高度（像素）。"""
        font = load_font(BODY_SIZE)
        lh = line_height(font)
        n = sum(len(wrap_text(self.draw, line, font, width)) for line in raw_lines)
        return n * lh

    def measure_table(self, spec: dict[str, Any]) -> int:
        font = load_font(TABLE_SIZE)
        lh = line_height(font)
        return lh + 28 + len(spec["rows"]) * (lh + 20)

    # -- 绘制 ---------------------------------------------------------------

    def draw_title(self, y: int, text: str) -> tuple[Region, int]:
        font = load_font(TITLE_SIZE)
        lh = line_height(font)
        w = text_width(self.draw, text, font)
        x = (PAGE_W - w) / 2.0
        self.draw.text((x, y), text, font=font, fill=self.fg)
        bbox = clamp_box((x - 3.0, y - 2.0, x + w + 3.0, y + lh + 2.0), PAGE_W, PAGE_H)
        region = Region("title", bbox, text)
        self.regions.append(region)
        return region, lh

    def draw_paragraph(
        self, x: int, y: int, width: int, raw_lines: Sequence[str]
    ) -> tuple[Region, int]:
        font = load_font(BODY_SIZE)
        lh = line_height(font)
        lines: list[str] = []
        for raw in raw_lines:
            lines.extend(wrap_text(self.draw, raw, font, width))
        max_w = max((text_width(self.draw, ln, font) for ln in lines), default=0.0)
        for i, ln in enumerate(lines):
            self.draw.text((x, y + i * lh), ln, font=font, fill=self.fg)
        bbox = clamp_box(
            (x - 2.0, y - 2.0, x + max_w + 3.0, y + len(lines) * lh + 2.0),
            PAGE_W,
            PAGE_H,
        )
        region = Region("paragraph", bbox, "\n".join(lines))
        self.regions.append(region)
        return region, len(lines) * lh

    # -- 表格 ---------------------------------------------------------------

    @staticmethod
    def _table_geometry(
        spec: dict[str, Any], width: int
    ) -> tuple[list[int], list[int], int]:
        """返回 (列宽列表, 行高列表[表头行 + 正文行], 表格总高)。"""
        font = load_font(TABLE_SIZE)
        lh = line_height(font)
        fracs = spec["col_fracs"]
        col_w: list[int] = [int(round(width * f)) for f in fracs]
        col_w[-1] += width - sum(col_w)  # 吸收舍入误差
        row_h: list[int] = [lh + 28] + [lh + 20] * len(spec["rows"])
        return col_w, row_h, sum(row_h)

    @staticmethod
    def _owner_map(
        spec: dict[str, Any]
    ) -> dict[tuple[int, int], tuple[int, int]]:
        """把「被合并覆盖的格子」映射到合并区域的左上角 owner 格子。

        行号约定：0 是表头行，1.. 是正文行。merges 用正文行号 (0-based) 表示。
        """
        n_rows = 1 + len(spec["rows"])
        n_cols = len(spec["headers"])
        owner = {(r, c): (r, c) for r in range(n_rows) for c in range(n_cols)}
        for (r0, c0, rowspan, colspan) in spec.get("merges", []):
            gr0 = r0 + 1  # 正文行号 -> 网格行号
            for dr in range(rowspan):
                for dc in range(colspan):
                    owner[(gr0 + dr, c0 + dc)] = (gr0, c0)
        return owner

    def draw_table(
        self, x: int, y: int, spec: dict[str, Any]
    ) -> tuple[Region, int]:
        font = load_font(TABLE_SIZE)
        lh = line_height(font)
        headers: list[str] = spec["headers"]
        rows: list[list[str]] = spec["rows"]
        width = int(spec["_width"])
        col_w, row_h, total_h = self._table_geometry(spec, width)
        n_rows = len(row_h)
        n_cols = len(headers)
        owner = self._owner_map(spec)
        x_edges = [x]
        for w in col_w:
            x_edges.append(x_edges[-1] + w)
        y_edges = [y]
        for h in row_h:
            y_edges.append(y_edges[-1] + h)

        # 外框
        self.draw.rectangle([x, y, x_edges[-1], y_edges[-1]], outline=self.fg, width=3)
        # 横向分隔线：被合并单元格覆盖的线段不画
        for r in range(1, n_rows):
            y_line = y_edges[r]
            for c in range(n_cols):
                if owner[(r - 1, c)] != owner[(r, c)]:
                    self.draw.line([(x_edges[c], y_line), (x_edges[c + 1], y_line)], fill=self.fg, width=2)
        # 纵向分隔线
        for c in range(1, n_cols):
            x_line = x_edges[c]
            for r in range(n_rows):
                if owner[(r, c - 1)] != owner[(r, c)]:
                    self.draw.line([(x_line, y_edges[r]), (x_line, y_edges[r + 1])], fill=self.fg, width=2)

        # 单元格文字（仅 owner 格子画一次）
        cell_texts: list[str] = []
        for r in range(n_rows):
            for c in range(n_cols):
                if owner[(r, c)] != (r, c):
                    continue
                # 该 owner 的跨度
                span_r = 1
                while r + span_r < n_rows and owner[(r + span_r, c)] == (r, c):
                    span_r += 1
                span_c = 1
                while c + span_c < n_cols and owner[(r, c + span_c)] == (r, c):
                    span_c += 1
                text = headers[c] if r == 0 else rows[r - 1][c]
                cell_texts.append(text)
                if not text:
                    continue
                cx = (x_edges[c] + x_edges[c + span_c]) / 2.0
                cy = (y_edges[r] + y_edges[r + span_r]) / 2.0
                self.draw.text((cx, cy), text, font=font, fill=self.fg, anchor="mm")

        bbox = clamp_box((float(x), float(y), float(x_edges[-1]), float(y_edges[-1])), PAGE_W, PAGE_H)
        region = Region("table", bbox, "\n".join(cell_texts))
        self.regions.append(region)
        return region, total_h

    # -- 印章 ---------------------------------------------------------------

    def draw_stamp(
        self, cx: float, cy: float, r: float, line1: str, line2: str
    ) -> Region:
        """画红色圆形印章：外圈 + 五角星 + 两行文字。"""
        red = STAMP_RED
        self.draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=red,
            width=max(3, int(r * 0.06)),
        )
        # 五角星
        star_cy = cy - r * 0.36
        outer = r * 0.17
        inner = r * 0.07
        pts: list[tuple[float, float]] = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rad = outer if i % 2 == 0 else inner
            pts.append((cx + rad * math.cos(ang), star_cy + rad * math.sin(ang)))
        self.draw.polygon(pts, fill=red)

        font = load_font(STAMP_FONT_SIZE)
        self.draw.text((cx, cy + r * 0.02), line1, font=font, fill=red, anchor="mm")
        self.draw.text((cx, cy + r * 0.30), line2, font=font, fill=red, anchor="mm")

        bbox = clamp_box((cx - r, cy - r, cx + r, cy + r), PAGE_W, PAGE_H)
        region = Region("stamp", bbox, f"{line1}\n{line2}")
        self.regions.append(region)
        return region


# ---------------------------------------------------------------------------
# 版面布局
# ---------------------------------------------------------------------------


def render_document(rng: random.Random, hard_case: str | None) -> tuple[Image.Image, list[Region], dict[str, Any], float]:
    """渲染一张文书。

    Returns:
        (图片, 区域列表, 渲染内容字典, 旋转角度)
    """
    merged_table = hard_case == "table_merge"
    content = build_content(rng, merged_table=merged_table)
    renderer = DocumentRenderer(rng, low_contrast=(hard_case == "low_contrast"))

    top = rng.randint(56, 84)
    bottom = 44
    left = rng.randint(80, 118)
    right = rng.randint(80, 118)
    para_w = PAGE_W - left - right

    # 标题
    title_region, title_h = renderer.draw_title(top, content["title"])

    # 正文块顺序：甲方信息 → 乙方信息 → 条款 → 工资构成表 → 落款
    clauses = list(content["clauses"])

    def make_blocks(clause_list: Sequence[str]) -> list[tuple[str, Any]]:
        blocks: list[tuple[str, Any]] = [
            ("para", content["party_a"]),
            ("para", content["party_b"]),
        ]
        blocks.extend(("para", [clause]) for clause in clause_list)
        blocks.append(("table", content["table"]))
        blocks.append(("para", content["sign"]))
        return blocks

    def total_height(blocks: Sequence[tuple[str, Any]]) -> int:
        h = 0
        for kind, data in blocks:
            if kind == "para":
                h += renderer.measure_paragraph(data, para_w)
            else:
                h += renderer.measure_table(data)
        return h

    # 放不下就丢弃最后的可选条款（保留 4 条必要条款 + 落款）
    while True:
        blocks = make_blocks(clauses)
        if (
            total_height(blocks) + 14 * len(blocks)
            <= PAGE_H - top - bottom - title_h
            or len(clauses) <= 4
        ):
            break
        clauses.pop()

    block_h = total_height(blocks)
    n_gaps = len(blocks)  # 标题后 1 个 + 每个块前 1 个
    min_gap = 14
    available = PAGE_H - top - bottom - title_h - block_h
    if available < n_gaps * min_gap:  # 极端情况：缩小间距
        min_gap = 8
        available = PAGE_H - top - bottom - title_h - block_h
    extra = available - n_gaps * min_gap
    if extra < 0:
        raise RuntimeError("内容过长，版面放不下，请检查字号/内容模板")
    weights = [rng.uniform(0.4, 1.6) for _ in range(n_gaps)]
    gaps = [int(round(min_gap + extra * w / sum(weights))) for w in weights]

    # 依次绘制
    y = top + title_h
    drawn_paras: list[Region] = []
    table_region: Region | None = None
    for i, (kind, data) in enumerate(blocks):
        y += gaps[i]
        if kind == "para":
            region, h = renderer.draw_paragraph(left, y, para_w, data)
            drawn_paras.append(region)
        else:
            table_x = rng.randint(88, 132)
            table_w = PAGE_W - 2 * table_x
            table_spec = dict(data)
            table_spec["_width"] = table_w
            table_region, h = renderer.draw_table(table_x, y, table_spec)
        y += h

    # 印章：默认盖在「落款」右侧空白处；难例则压在正文条款上造成遮挡
    short_name, seal_type = content["stamp"]
    if hard_case == "stamp_occlusion":
        target = drawn_paras[3]  # 第 1 条之后（工资条款附近）
        for cand in drawn_paras:
            if "工资" in cand.text and "甲方（盖章）" not in cand.text:
                target = cand
                break
        cx = (target.bbox[0] + target.bbox[2]) / 2.0 + rng.randint(-30, 30)
        cy = (target.bbox[1] + target.bbox[3]) / 2.0 + rng.randint(-10, 10)
        stamp_r = rng.randint(88, 104)
    else:
        sign_region = drawn_paras[-1]
        stamp_r = rng.randint(66, 78)
        cx = rng.randint(610, 830)
        cy = (sign_region.bbox[1] + sign_region.bbox[3]) / 2.0
        if table_region is not None:
            # 保证印章不压住表格底部的单元格文字（落款区只有 3 行，印章较大时要下移）
            cy = max(cy, table_region.bbox[3] + stamp_r + 6.0)
        cy = min(cy, PAGE_H - stamp_r - 6.0)
    # 保证印章完整落在页面内
    cx = min(max(cx, stamp_r + 6), PAGE_W - stamp_r - 6)
    cy = min(max(cy, stamp_r + 6), PAGE_H - stamp_r - 6)
    renderer.draw_stamp(cx, cy, stamp_r, short_name, seal_type)

    img = renderer.img
    angle = 0.0
    regions = renderer.regions

    # —— 难例后处理 ——
    if hard_case == "low_contrast":
        img, regions = add_low_contrast_noise(img, regions, rng)
    elif hard_case == "rotation":
        angle = rng.uniform(2.0, 5.0) * rng.choice([-1.0, 1.0])
        img, regions = rotate_image_and_boxes(img, regions, angle, bg=renderer.bg)

    return img, regions, content, angle


def add_low_contrast_noise(
    img: Image.Image, regions: list[Region], rng: random.Random
) -> tuple[Image.Image, list[Region]]:
    """低对比度难例：轻微失焦 + 高斯噪声（模拟扫描/拍照质量差）。"""
    blurred = img.filter(ImageFilter.GaussianBlur(0.6))
    arr = np.asarray(blurred).astype(np.float32)
    noise_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    arr = np.clip(arr + noise_rng.normal(0.0, 5.0, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB"), regions


def rotate_image_and_boxes(
    img: Image.Image,
    regions: list[Region],
    angle_deg: float,
    *,
    bg: tuple[int, int, int] = (255, 255, 255),
) -> tuple[Image.Image, list[Region]]:
    """绕图片中心旋转并同步变换 bbox（纯 PIL，不依赖 cv2）。

    返回的 bbox 是旋转后区域的外接轴对齐矩形，并裁剪到新画布内。
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    w, h = img.size
    new_w = int(math.ceil(abs(w * ca) + abs(h * sa)))
    new_h = int(math.ceil(abs(w * sa) + abs(h * ca)))
    cs_x, cs_y = w / 2.0, h / 2.0
    cd_x, cd_y = new_w / 2.0, new_h / 2.0

    def forward(x: float, y: float) -> tuple[float, float]:
        dx, dy = x - cs_x, y - cs_y
        return (ca * dx + sa * dy + cd_x, -sa * dx + ca * dy + cd_y)

    # PIL AFFINE 需要的是「目标坐标 -> 源坐标」的逆变换
    # src = (ca, -sa) @ (dst - cd) + cs
    out = img.transform(
        (new_w, new_h),
        Image.AFFINE,
        (
            ca,
            -sa,
            cs_x - ca * cd_x + sa * cd_y,
            sa,
            ca,
            cs_y - sa * cd_x - ca * cd_y,
        ),
        resample=Image.BICUBIC,
        fillcolor=bg,
    )

    new_regions: list[Region] = []
    for r in regions:
        x1, y1, x2, y2 = r.bbox
        corners = [forward(x1, y1), forward(x2, y1), forward(x1, y2), forward(x2, y2)]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        box = clamp_box((min(xs), min(ys), max(xs), max(ys)), new_w, new_h)
        new_regions.append(Region(r.cls, box, r.text))
    return out, new_regions


# ---------------------------------------------------------------------------
# 标注写出
# ---------------------------------------------------------------------------


def reading_order(regions: Sequence[Region]) -> list[Region]:
    """阅读顺序：先按 bbox 上沿，再按左沿（与 OCR 流水线的排序规则完全一致）。"""
    return sorted(regions, key=lambda r: (round(r.bbox[1], 1), round(r.bbox[0], 1)))


def write_yolo_label(
    path: Path, regions: Sequence[Region], width: int, height: int
) -> int:
    """写 YOLO 格式标签，返回写入的区域数。"""
    lines: list[str] = []
    for r in regions:
        x1, y1, x2, y2 = r.bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            continue
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        lines.append(
            f"{CLASS_TO_ID[r.cls]} {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_plan(num: int, num_train: int) -> list[dict[str, Any]]:
    """构造图片生成计划：id / split / 难例类型。"""
    plan: list[dict[str, Any]] = []
    for idx in range(num):
        split = "train" if idx < num_train else "val"
        hard = HARD_CASE_PLAN.get(idx)
        plan.append({"index": idx, "split": split, "hard_case": hard})
    return plan


def generate_dataset(out_dir: Path, num: int, num_train: int, seed: int) -> dict[str, Any]:
    """生成完整数据集并返回 annotations 字典。"""
    if num_train >= num:
        raise ValueError("--num-train 必须小于 --num")
    if num < 8:
        raise ValueError("--num 至少 8，才能切出 train/val")

    font_path = resolve_font_path()
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    plan = build_plan(num, num_train)
    images_meta: list[dict[str, Any]] = []
    class_counts = {"train": {c: 0 for c in CLASS_NAMES}, "val": {c: 0 for c in CLASS_NAMES}}
    total_chars = {"train": 0, "val": 0}

    for item in plan:
        idx = item["index"]
        split = item["split"]
        hard = item["hard_case"]
        rng = random.Random(seed * 100_000 + idx)  # 每张图独立、可复现
        img, regions, content, angle = render_document(rng, hard)
        width, height = img.size
        doc_id = f"doc_{idx:04d}"
        img_path = out_dir / "images" / split / f"{doc_id}.png"
        img.save(img_path)

        ordered = reading_order(regions)
        full_text = "\n".join(r.text for r in ordered)
        label_path = out_dir / "labels" / split / f"{doc_id}.txt"
        n_labels = write_yolo_label(label_path, regions, width, height)

        # 自检：bbox 合法、类别合法、文字非空
        for r in regions:
            x1, y1, x2, y2 = r.bbox
            assert r.cls in CLASS_TO_ID, r.cls
            assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, r.bbox
            assert r.text, f"{doc_id} 存在空文本区域"
        assert n_labels >= 5, f"{doc_id} 标注数量异常: {n_labels}"

        meta = {
            "id": doc_id,
            "file": f"images/{split}/{doc_id}.png",
            "label": f"labels/{split}/{doc_id}.txt",
            "split": split,
            "width": width,
            "height": height,
            "hard_case": hard,
            "rotation_deg": round(angle, 2),
            "num_regions": len(regions),
            "full_text": full_text,
            "region_order": [r.cls for r in ordered],
            "regions": [r.to_dict(f"r{i}") for i, r in enumerate(regions)],
            "fields": content["fields"],
        }
        images_meta.append(meta)
        for r in regions:
            class_counts[split][r.cls] += 1
        total_chars[split] += len(full_text.replace("\n", ""))

    annotations: dict[str, Any] = {
        "meta": {
            "generator": "scripts/make_dataset.py",
            "seed": seed,
            "num_images": num,
            "num_train": num_train,
            "page_size": [PAGE_W, PAGE_H],
            "font": font_path,
            "classes": CLASS_NAMES,
            "class_to_id": CLASS_TO_ID,
            "hard_cases": {str(k): v for k, v in HARD_CASE_PLAN.items() if k < num},
            "class_counts": class_counts,
            "full_text_chars_excluding_newline": total_chars,
            "note": "合成数据，bbox 由渲染模板直接给出，是精确 ground truth。",
        },
        "images": images_meta,
    }
    (out_dir / "annotations.json").write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ultralytics 数据集配置：path 写相对值，运行时由 yolo_detect.py 修正为绝对路径，
    # 这样整个项目目录可以随意移动/复制（避免把本机绝对路径写死进仓库）。
    data_yaml = (
        f"# 由 scripts/make_dataset.py 自动生成\n"
        f"path: .\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    )
    (out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return annotations


def print_summary(annotations: dict[str, Any], out_dir: Path) -> None:
    meta = annotations["meta"]
    print("=" * 68)
    print(f"合成数据集已生成: {out_dir.resolve()}")
    print(f"  图片数   : {meta['num_images']}（train {meta['num_train']} / val "
          f"{meta['num_images'] - meta['num_train']}）")
    print(f"  页面尺寸 : {meta['page_size'][0]}x{meta['page_size'][1]}")
    print(f"  字体     : {meta['font']}")
    print(f"  seed     : {meta['seed']}")
    for split in ("train", "val"):
        counts = meta["class_counts"][split]
        print(f"  [{split}] 区域数: " + ", ".join(f"{k}={v}" for k, v in counts.items())
              + f" | 全文字符数(去换行)={meta['full_text_chars_excluding_newline'][split]}")
    hard = [img["id"] + ":" + str(img["hard_case"]) for img in annotations["images"] if img["hard_case"]]
    print(f"  难例     : {len(hard)} 张 -> {', '.join(hard)}")
    print("=" * 68)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成中文法律文书合成评估集")
    parser.add_argument("--out", type=Path, default=Path("data"), help="输出目录（默认 data）")
    parser.add_argument("--num", type=int, default=32, help="图片总数（默认 32）")
    parser.add_argument("--num-train", type=int, default=24, help="训练集图片数（默认 24）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    annotations = generate_dataset(args.out, args.num, args.num_train, args.seed)
    print_summary(annotations, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
