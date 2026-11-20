"""离线评测：检索消融 + 答案质量（引用校验 / LLM-as-judge）。

用法：
    # 只跑检索消融（快，纯 CPU 约几分钟）
    python scripts/evaluate.py --ablations

    # 检索消融 + 对 full 配置做答案生成与引用校验（mock LLM，完全离线）
    python scripts/evaluate.py --ablations --answers

    # 指定某几个配置 + 真实 LLM 打分
    python scripts/evaluate.py --configs full hybrid_rerank --answers --real-llm

输出：
    data/eval/retrieval_results.json    每种配置的指标与逐题明细
    data/eval/answer_results.json       full 配置的答案、引用与打分
    data/eval/report.md                 可直接贴进 README 的 Markdown 报告
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lrs.config import ABLATION_PRESETS, EVAL_DIR, load_settings  # noqa: E402
from lrs.metrics import evaluate_rankings, format_metrics_row  # noqa: E402
from lrs.schemas import Retrieved  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evaluate")

BENCHMARK = PROJECT_ROOT / "data" / "benchmark.jsonl"
RESULT_JSON = EVAL_DIR / "retrieval_results.json"
ANSWER_JSON = EVAL_DIR / "answer_results.json"
REPORT_MD = EVAL_DIR / "report.md"


def load_benchmark(path: Path = BENCHMARK) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    if not cases:
        raise ValueError(f"{path} 为空")
    return cases


def _rank_articles(results: list[Retrieved]) -> list[str]:
    """把 Retrieved 列表转成 article_id 排名（相邻同条法条去重）。"""
    out: list[str] = []
    for r in results:
        if not out or out[-1] != r.article_id:
            out.append(r.article_id)
    return out


def run_config(service, cases: list[dict], name: str, settings) -> tuple[dict, list[list[str]]]:
    """跑一种配置的全部题目，返回 (指标, 每题排名)。

    关键：`service.retrieve()` 会读取 `service.settings` 决定是否启用分解/扩展，
    所以必须先切换 service 的 settings，否则所有配置都会走同一套逻辑（消融失效）。
    """
    from dataclasses import replace

    service = replace(service, settings=settings)
    t0 = time.perf_counter()
    rankings: list[list[str]] = []
    latencies: list[float] = []
    for i, case in enumerate(cases, 1):
        from lrs.query_processing import plan_query

        plan = plan_query(
            case["question"],
            decompose=settings.decompose_enabled,
            expand=settings.expand_enabled,
            max_sub_queries=settings.max_sub_queries,
        )
        ts = time.perf_counter()
        candidates = service.retrieve(plan, top_k=settings.top_k)
        # 与线上 answer() 完全一致的后续两个阶段
        if settings.rerank_enabled and candidates:
            from lrs.reranker import rerank as do_rerank

            candidates = do_rerank(case["question"], candidates, service.reranker,
                                   top_k=settings.rerank_candidates)
        if settings.filter_enabled and candidates:
            from lrs.filter import filter_candidates

            candidates = filter_candidates(case["question"], candidates, service.llm)
        candidates = candidates[: settings.final_k]
        rankings.append(_rank_articles(candidates))
        latencies.append((time.perf_counter() - ts) * 1000)
        if i % 10 == 0:
            logger.info("  [%s] %d/%d", name, i, len(cases))
    metrics = evaluate_rankings(cases, rankings)
    metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)
    metrics["elapsed_s"] = round(time.perf_counter() - t0, 1)
    metrics["settings"] = {
        "decompose": settings.decompose_enabled,
        "expand": settings.expand_enabled,
        "rerank": settings.rerank_enabled,
        "filter": settings.filter_enabled,
        "top_k": settings.top_k,
        "final_k": settings.final_k,
    }
    return metrics, rankings


def run_answers(service, cases: list[dict], llm_judge: bool = False) -> dict:
    """对 full 配置生成答案，做引用校验与（可选）LLM 打分。"""
    from lrs.prompts import ANSWER_PROMPT  # noqa: F401

    all_ids = {c.article_id for c in service.chunks}
    rows = []
    latencies = []
    for i, case in enumerate(cases, 1):
        ans = service.answer(case["question"])
        latencies.append(ans.latency_ms["total"])
        # 1) 程序化引用校验：引用的条号是否存在、是否在 gold 里
        cited_ids = [c.article_id for c in ans.citations]
        valid = [cid for cid in cited_ids if cid in all_ids]
        in_gold = [cid for cid in cited_ids if cid in set(case["relevant"])]
        row = {
            "id": case["id"],
            "question": case["question"],
            "answer": ans.answer,
            "cited": cited_ids,
            "citation_valid_rate": round(len(valid) / len(cited_ids), 3) if cited_ids else 0.0,
            "citation_gold_rate": round(len(in_gold) / len(cited_ids), 3) if cited_ids else 0.0,
            "gold_coverage": round(len(in_gold) / len(set(case["relevant"])), 3),
            "llm_mode": ans.llm_mode,
            "latency_ms": ans.latency_ms,
            "retrieved_final": [c.article_id for c in ans.retrieved if c.kept][:5],
        }
        # 2) LLM-as-judge（可选）
        if llm_judge:
            row["judge"] = judge_answer(service.llm, case, ans)
        rows.append(row)
        if i % 10 == 0:
            logger.info("  答案生成 %d/%d", i, len(cases))

    n = len(rows)
    summary = {
        "n": n,
        "llm_mode": service.llm.mode,
        "avg_latency_ms": round(sum(latencies) / n, 1),
        "citation_valid_rate": round(sum(r["citation_valid_rate"] for r in rows) / n, 3),
        "citation_gold_rate": round(sum(r["citation_gold_rate"] for r in rows) / n, 3),
        "gold_coverage": round(sum(r["gold_coverage"] for r in rows) / n, 3),
        "no_citation_cases": sum(1 for r in rows if not r["cited"]),
    }
    if llm_judge:
        judged = [r for r in rows if "judge" in r]
        summary["judge"] = {
            "n": len(judged),
            "legal_compliance": round(sum(r["judge"].get("legal_compliance", 0) for r in judged)
                                      / max(len(judged), 1), 2),
            "citation_accuracy": round(sum(r["judge"].get("citation_accuracy", 0) for r in judged)
                                       / max(len(judged), 1), 2),
            "judge_model": judged[0]["judge"].get("judge_model", "?") if judged else "?",
        }
    return {"summary": summary, "rows": rows}


def judge_answer(llm, case: dict, ans) -> dict:
    """用 LLM-as-judge 按 1-5 分评价：法律合规性 / 引用准确性。"""
    from lrs.prompts import JUDGE_PROMPT

    gold_text = "\n".join(f"- {c}" for c in case["relevant"])
    prompt = JUDGE_PROMPT.format(question=case["question"], gold=gold_text, answer=ans.answer)
    try:
        raw = llm.complete(prompt, task="judge")
        from lrs.filter import _parse_json_loose

        parsed = _parse_json_loose(raw) or {}
        return {
            "legal_compliance": float(parsed.get("legal_compliance", 0) or 0),
            "citation_accuracy": float(parsed.get("citation_accuracy", 0) or 0),
            "note": str(parsed.get("note", ""))[:200],
            "judge_model": llm.model,
        }
    except Exception as e:  # pragma: no cover
        return {"legal_compliance": 0.0, "citation_accuracy": 0.0,
                "note": f"judge 失败: {e}", "judge_model": llm.model}


def write_report(retrieval: dict[str, dict], answers: dict | None, meta: dict) -> None:
    order = ["dense_only", "bm25_only", "hybrid_rrf", "hybrid_rerank", "full"]
    names = [n for n in order if n in retrieval] + [n for n in retrieval if n not in order]
    rows = [format_metrics_row(n, retrieval[n], k=5) for n in names]
    header = ("| 配置 | MRR | MRR@5 | Hit@5 | Recall@5 | nDCG@5 | 平均耗时(ms) |\n"
              "|---|---|---|---|---|---|---|")
    lines = [
        "# 检索与答案评测报告",
        "",
        f"> 生成时间：{meta['timestamp']}　|　数据：`data/benchmark.jsonl`（{meta['cases']} 题）"
        f"　|　embedding：`{meta['dense_model']}`　|　rerank：`{meta['rerank_backend']}`"
        f"　|　LLM：`{meta['llm_mode']}`",
        "",
        "## 1. 检索消融（按法条 article_id 计算）",
        "",
        header,
    ]
    for r, name in zip(rows, names):
        lines.append(
            f"| {name} | {r['MRR']:.4f} | {r['MRR@5']:.4f} | {r['Hit@5']:.4f} | "
            f"{r['Recall@5']:.4f} | {r['nDCG@5']:.4f} | {retrieval[name].get('avg_latency_ms', '?')} |"
        )
    lines += [
        "",
        "指标定义：MRR=第一个相关法条排名倒数均值；Hit@5=前5条至少命中一条的题目占比；"
        "Recall@5=前5条覆盖 gold 法条的比例；nDCG@5=考虑排序位置的归一化折损累计增益。",
        "",
    ]

    if "dense_only" in retrieval and "full" in retrieval:
        # 同时对比"稠密基线"和"最强单路基线"，避免只报对自己有利的对比
        rows = []
        for base_name in ("dense_only", "bm25_only"):
            if base_name not in retrieval:
                continue
            base = retrieval[base_name]
            best = max(retrieval.values(), key=lambda m: m.get("mrr", 0))
            full = retrieval["full"]
            parts = []
            for key, label in [("mrr", "MRR"), ("hit@5", "Hit@5"), ("recall@5", "Recall@5")]:
                b, f = base.get(key, 0), full.get(key, 0)
                if b:
                    parts.append(f"{label} {b:.4f} → {f:.4f}（{100 * (f - b) / b:+.1f}%）")
            rows.append(f"- 相比 `{base_name}`：" + "；".join(parts))
        best_name = max(retrieval, key=lambda n: retrieval[n].get("mrr", 0))
        rows.append(f"- 本基准上 MRR 最高的配置是 `{best_name}`"
                    f"（MRR={retrieval[best_name].get('mrr', 0):.4f}），"
                    f"`full` 的 MRR={retrieval['full'].get('mrr', 0):.4f}。"
                    "**不隐藏 full 不是最优这一事实**，详见 README §5.1 分析。")
        lines += ["**对比（full vs 单路基线）**：", "", *rows, ""]

    if answers:
        s = answers["summary"]
        lines += [
            "## 2. 答案层指标（full 配置）",
            "",
            "| 指标 | 数值 | 说明 |",
            "|---|---|---|",
            f"| 引用有效率 | {s['citation_valid_rate']:.1%} | 引用的条号真实存在于语料的比例 |",
            f"| 引用命中率 | {s['citation_gold_rate']:.1%} | 引用中包含 gold 法条的比例 |",
            f"| gold 覆盖率 | {s['gold_coverage']:.1%} | 答案引用覆盖了 gold 法条的比例 |",
            f"| 无引用题数 | {s['no_citation_cases']} | 答案里没有任何可回溯引用 |",
            f"| 平均生成耗时 | {s['avg_latency_ms']:.0f} ms | 端到端（含检索） |",
            f"| LLM 模式 | `{s['llm_mode']}` | mock 为离线抽取式，非真实大模型 |",
            "",
        ]
        if "judge" in s:
            j = s["judge"]
            lines += [
                "### LLM-as-judge（1-5 分）",
                "",
                f"- 法律合规性：**{j['legal_compliance']:.2f} / 5.0**",
                f"- 引用准确性：**{j['citation_accuracy']:.2f} / 5.0**",
                f"- 裁判模型：`{j['judge_model']}`（mock 为启发式打分，仅用于打通流程）",
                "",
            ]

    lines += [
        "## 3. 错误分析",
        "",
        "逐题明细见 `data/eval/retrieval_results.json` 的 `per_case` 字段"
        "（含 first_relevant_rank / ranked_top10 / relevant）。",
        "",
        "建议按以下顺序排查：① 多意图题是否漏召回（对比 Recall@5）；"
        "② 第一命中排名是否靠后（对比 MRR@5）；③ 重排是否把 gold 挤出前 5；"
        "④ 零样本过滤是否误删 gold。",
        "",
        "## 4. 可复现命令",
        "",
        "```bash",
        "python scripts/evaluate.py --ablations --answers",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="*", default=None,
                        help=f"要评测的配置，可选：{list(ABLATION_PRESETS)}；默认全部")
    parser.add_argument("--ablations", action="store_true",
                        help="跑全部消融配置（默认行为；保留该参数是为了命令可读性）")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--answers", action="store_true", help="额外跑答案生成与引用校验")
    parser.add_argument("--judge", action="store_true", help="答案层启用 LLM-as-judge")
    parser.add_argument("--real-llm", action="store_true", help="使用真实 LLM（需 OPENAI_API_KEY）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    args = parser.parse_args()

    cases = load_benchmark()
    if args.limit:
        cases = cases[: args.limit]
    logger.info("评测集 %d 题", len(cases))

    from lrs.service import RAGService

    names = args.configs or list(ABLATION_PRESETS)
    unknown = [n for n in names if n not in ABLATION_PRESETS]
    if unknown:
        parser.error(f"未知配置 {unknown}，可选 {list(ABLATION_PRESETS)}")

    need_rerank = any(ABLATION_PRESETS[n].get("rerank_enabled", True) for n in names)
    settings = load_settings()
    t0 = time.perf_counter()
    service = RAGService.load_or_build(settings, need_reranker=need_rerank,
                                       prefer_real_llm=args.real_llm)
    logger.info("服务装配完成，用时 %.1fs；embedding=%s / rerank=%s / llm=%s",
                time.perf_counter() - t0, service.embedder.name, service.reranker.name
                if service.reranker else "off", service.llm.mode)

    retrieval: dict[str, dict] = {}
    if not args.skip_ablations:
        for name in names:
            preset = ABLATION_PRESETS[name]
            cfg = load_settings(**preset)
            logger.info("=== 配置 %s：%s", name, preset)
            metrics, _ = run_config(service, cases, name, cfg)
            retrieval[name] = metrics
            logger.info("  MRR=%.4f Hit@5=%.4f Recall@5=%.4f",
                        metrics["mrr"], metrics["hit@5"], metrics["recall@5"])
        RESULT_JSON.write_text(
            json.dumps({"meta": _meta(cases, service), "results": retrieval},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"检索评测已写入 {RESULT_JSON}")

    answers = None
    if args.answers:
        full_cfg = load_settings(**ABLATION_PRESETS["full"])
        # 用 full 配置重建（retrieve 时按 settings 开关）
        service.settings = full_cfg
        answers = run_answers(service, cases, llm_judge=args.judge)
        ANSWER_JSON.write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"答案评测已写入 {ANSWER_JSON}")

    if retrieval or answers:
        write_report(retrieval, answers, _meta(cases, service))
        print(f"报告已写入 {REPORT_MD}")
    return 0


def _meta(cases, service) -> dict:
    import platform

    return {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "cases": len(cases),
        "dense_model": service.embedder.name,
        "rerank_backend": getattr(service.reranker, "name", "off") if service.reranker else "off",
        "llm_mode": service.llm.mode,
        "corpus_articles": len({c.article_id for c in service.chunks}),
        "chunks": len(service.chunks),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "settings": {
            "top_k": service.settings.top_k,
            "final_k": service.settings.final_k,
            "rerank_candidates": service.settings.rerank_candidates,
            "rrf_k": service.settings.rrf_k,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
