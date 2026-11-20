"""命令行入口：构建索引 / 单题问答 / 交互式 REPL / 评测。

用法：
    # 建索引（首次运行会自动建，无需手动）
    python scripts/cli.py build

    # 单题问答
    python scripts/cli.py ask "公司违法辞退我，月薪12000、工作3年7个月，能赔多少？"

    # 交互式多轮（本地 REPL，不依赖 Chainlit）
    python scripts/cli.py chat
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _print_answer(res) -> None:
    print("\n" + "=" * 78)
    print(f"问题：{res.question}")
    print("-" * 78)
    if res.plan.is_multi_intent:
        print(f"查询理解：{res.plan.note}")
        for i, q in enumerate(res.plan.sub_queries, 1):
            print(f"  子查询{i}：{q}")
    elif res.plan.expansions:
        print(f"查询理解：{res.plan.note} → {'、'.join(res.plan.expansions)}")
    print("-" * 78)
    print(f"回答（LLM 模式：{res.llm_mode}）：\n")
    print(res.answer)
    print("-" * 78)
    print("引用法条：")
    for c in res.citations:
        print(f"  · {c.title}：{c.quote}")
    print("-" * 78)
    print("检索明细（重排后，★=通过零样本过滤）：")
    for r in res.retrieved:
        mark = "★" if r.kept else "×"
        print(f"  {mark} #{r.rank:<2} {r.title:<24} d={r.scores.get('dense', 0):>7} "
              f"b={r.scores.get('bm25', 0):>6} rrf={r.scores.get('rrf', 0):>6} "
              f"rr={r.scores.get('rerank', 0):>7} src={r.source}")
    print("耗时(ms)：", json.dumps(res.latency_ms, ensure_ascii=False))
    print("=" * 78 + "\n")


def cmd_build(args) -> int:
    from lrs.service import RAGService

    svc = RAGService.load_or_build(force_rebuild=args.force, need_reranker=False)
    print(f"索引就绪：{len(svc.chunks)} 个 chunk，collection={svc.store.collection_name}")
    return 0


def cmd_ask(args) -> int:
    from lrs.service import RAGService

    svc = RAGService.load_or_build(prefer_real_llm=args.real_llm)
    res = svc.answer(args.question)
    if args.json:
        from dataclasses import asdict

        print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
    else:
        _print_answer(res)
    return 0


def cmd_chat(args) -> int:
    from lrs.service import RAGService

    svc = RAGService.load_or_build(prefer_real_llm=args.real_llm)
    print("劳动法 RAG 咨询（输入 exit 退出）。"
          f"当前 LLM 模式：{svc.llm.mode}，检索模型：{svc.embedder.name}")
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0
        if q.lower() in {"exit", "quit", "q"}:
            return 0
        if not q:
            continue
        res = svc.answer(q)
        print(f"\n助手（{res.llm_mode}）：\n{res.answer}")
        if res.citations:
            print("\n引用：" + "；".join(c.title for c in res.citations))


def cmd_eval(args) -> int:
    """转发到 scripts/evaluate.py，保持入口统一。"""
    import runpy

    sys.argv = [str(PROJECT_ROOT / "scripts" / "evaluate.py"), *args.eval_args]
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "evaluate.py"), run_name="__main__")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="劳动法 RAG 咨询系统 CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="构建/重建索引")
    p_build.add_argument("--force", action="store_true", help="强制重建向量库与 BM25")
    p_build.set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask", help="单题问答")
    p_ask.add_argument("question", help="劳动合同法相关问题")
    p_ask.add_argument("--real-llm", action="store_true", help="使用真实 LLM（需 OPENAI_API_KEY）")
    p_ask.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="交互式 REPL")
    p_chat.add_argument("--real-llm", action="store_true")
    p_chat.set_defaults(func=cmd_chat)

    p_eval = sub.add_parser("eval", help="运行评测（转发给 scripts/evaluate.py）")
    p_eval.add_argument("eval_args", nargs=argparse.REMAINDER)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
