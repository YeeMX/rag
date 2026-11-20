# 生成结果快照（2026-11-21）

> 本目录是**交付快照**：把三个项目真实运行产生的结果复制到这里，方便一次性查看/发给别人。
> **权威来源仍是各项目目录**（脚本可重新生成），本目录只做留档，更新时以项目目录为准。

## 文件说明

| 文件 | 来源 | 说明 |
|---|---|---|
| `RAG评测报告.md` | `01-labor-law-rag/data/eval/report.md` | 50 题检索消融 + 答案层指标（由 `scripts/run_eval.py` 生成） |
| `benchmark_50题.jsonl` | `01-labor-law-rag/data/benchmark.jsonl` | 人工编写的 50 题评测基准（含 gold 法条与 hard negative） |
| `benchmark_stats.json` | `01-labor-law-rag/data/benchmark_stats.json` | 基准分布统计 |
| `视觉demo报告.md` | `03-vision-ocr-demo/outputs/report.md` | YOLO / OCR / 抽取全链路报告（由 `run_demo.py` 生成） |
| `真实数据验证报告.md` | `03-vision-ocr-demo/outputs/real_dataset/report_real.md` | **真实扫描件**（XFUND 中文）OCR CER 与 YOLO 域差距验证 |
| `metrics_real.json` | `03-vision-ocr-demo/outputs/real_dataset/metrics_real.json` | 真实数据指标原始 JSON |
| `metrics_yolo.json` | `03-vision-ocr-demo/outputs/metrics_yolo.json` | YOLO mAP / P / R / 速度原始指标 |
| `metrics_ocr.json` | `03-vision-ocr-demo/outputs/metrics_ocr.json` | 整图 vs 区域 OCR 的 CER 原始指标 |
| `images/chainlit界面.png` | 本项目 Chainlit UI 真实截图 | 打开 http://127.0.0.1:8000 看到的界面 |
| `images/真实扫描件检测对比.png` | `03-vision-ocr-demo/outputs/real_dataset/samples/` | 真实扫描件上的 GT vs 预测对比 |
| `images/yolo检测结果.png` | `03-vision-ocr-demo/outputs/images/det_doc_0024.png` | 版面检测可视化（合成数据） |
| `images/gt_vs_pred对比.png` | `03-vision-ocr-demo/outputs/images/gt_vs_pred_doc_0024.png` | GT（绿）vs 预测（红）对比 |
| `images/区域裁剪示例.png` | `03-vision-ocr-demo/outputs/images/region_crops_example.png` | 检测框 → 裁剪 → 逐区域 OCR 示例 |

## 快照里的关键数字

- RAG 50 题：MRR **0.6427** / Hit@5 **0.8400**（hybrid+rerank）；引用有效率 **100%**
- 视觉 YOLO：mAP@0.5 **0.9944** / mAP@0.5:0.95 **0.8831**
- OCR：整图 CER **0.0450** vs 区域裁剪 **0.0536**（难例传导问题，详见视觉 README §6）
- 抽取：9 字段准确率 **0.9167**

> 注意：这些数字对应「合成数据 + CPU + Mock LLM」环境，**不代表真实业务场景性能**；
> 各项目 README 均已注明局限。真实扫描件验证正在补充（见 `03-vision-ocr-demo/README.md` 第 7 节）。
