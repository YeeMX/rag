# 一键启动/运行三个 demo（Windows）
#
# 交互式：pwsh -File start.ps1
# 非交互：pwsh -File start.ps1 -Task rag-ui | rag-ask | rag-eval | agent-demo | agent-test | vision-demo | vision-test | test-all
#         可配合 -Question "公司违法辞退我，能赔多少？"
param(
    [ValidateSet("menu", "rag-ui", "rag-ask", "rag-eval", "agent-demo", "agent-test", "vision-demo", "vision-test", "test-all")]
    [string]$Task = "menu",
    [string]$Question = "公司违法辞退我，月薪12000、工作3年7个月，能赔多少？"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$env:HF_HUB_DISABLE_XET = "1"
$env:PYTHONIOENCODING = "utf-8"

$ragPy = "$root\.venv-rag\Scripts\python.exe"
$visionPy = "$root\.venv-vision\Scripts\python.exe"
$paddlePy = "$root\.venv-paddle\Scripts\python.exe"
$chainlit = "$root\.venv-rag\Scripts\chainlit.exe"

function Assert-Env {
    $missing = @()
    if (-not (Test-Path $ragPy)) { $missing += ".venv-rag" }
    if (-not (Test-Path $visionPy)) { $missing += ".venv-vision" }
    if (-not (Test-Path $paddlePy)) { $missing += ".venv-paddle" }
    if ($missing.Count -gt 0) {
        Write-Host "缺少环境：$($missing -join ', ')" -ForegroundColor Yellow
        Write-Host "请先运行：pwsh -File scripts\setup_envs.ps1" -ForegroundColor Yellow
        exit 1
    }
}

function Invoke-RagUi {
    Write-Host "启动 RAG 网页版：http://127.0.0.1:8000 （Ctrl+C 停止）" -ForegroundColor Cyan
    Push-Location "$root\01-labor-law-rag"
    try { & $chainlit run app\chainlit_app.py --host 127.0.0.1 --port 8000 }
    finally { Pop-Location }
}

function Invoke-RagAsk {
    Write-Host "问题：$Question" -ForegroundColor Cyan
    Push-Location "$root\01-labor-law-rag"
    try { & $ragPy scripts\cli.py ask $Question }
    finally { Pop-Location }
}

function Invoke-RagEval {
    Push-Location "$root\01-labor-law-rag"
    try { & $ragPy scripts\run_eval.py }
    finally { Pop-Location }
}

function Invoke-AgentDemo {
    Push-Location "$root\02-labor-law-agent"
    try { & $ragPy -m labor_agent.cli demo --no-color }
    finally { Pop-Location }
}

function Invoke-AgentTest {
    Push-Location "$root\02-labor-law-agent"
    try { & $ragPy -m pytest tests -q }
    finally { Pop-Location }
}

function Invoke-VisionDemo {
    Push-Location "$root\03-vision-ocr-demo"
    try { & $visionPy run_demo.py --skip-train }
    finally { Pop-Location }
}

function Invoke-VisionTest {
    Push-Location "$root\03-vision-ocr-demo"
    try { & $visionPy tests\run_all.py }
    finally { Pop-Location }
}

function Invoke-TestAll {
    Push-Location "$root\01-labor-law-rag"; & $ragPy -m pytest tests -q; Pop-Location
    Push-Location "$root\02-labor-law-agent"; & $ragPy -m pytest tests -q; Pop-Location
    Push-Location "$root\03-vision-ocr-demo"; & $visionPy tests\run_all.py; Pop-Location
}

function Show-Menu {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "  劳动法 AI 作品集 · 一键启动" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "  1) RAG 网页版        (Chainlit, http://127.0.0.1:8000)"
    Write-Host "  2) RAG 命令行提问    (默认问题，可改脚本里的 -Question)"
    Write-Host "  3) RAG 消融评测      (约 15 分钟)"
    Write-Host "  4) Agent demo        (3 工具 × function calling/LangGraph)"
    Write-Host "  5) Agent 测试        (106 个用例)"
    Write-Host "  6) 视觉 demo         (YOLO+OCR 全流程, 约 11 分钟)"
    Write-Host "  7) 视觉测试          (74 个用例)"
    Write-Host "  8) 全部测试"
    Write-Host "  0) 退出"
    Write-Host ""
}

function Run-Task([string]$name) {
    switch ($name) {
        "rag-ui"      { Invoke-RagUi }
        "rag-ask"     { Invoke-RagAsk }
        "rag-eval"    { Invoke-RagEval }
        "agent-demo"  { Invoke-AgentDemo }
        "agent-test"  { Invoke-AgentTest }
        "vision-demo" { Invoke-VisionDemo }
        "vision-test" { Invoke-VisionTest }
        "test-all"    { Invoke-TestAll }
        default       { throw "未知任务: $name" }
    }
}

Assert-Env
if ($Task -ne "menu") {
    Run-Task $Task
    exit 0
}

while ($true) {
    Show-Menu
    $choice = Read-Host "请选择"
    switch ($choice) {
        "1" { Run-Task "rag-ui" }
        "2" { Run-Task "rag-ask" }
        "3" { Run-Task "rag-eval" }
        "4" { Run-Task "agent-demo" }
        "5" { Run-Task "agent-test" }
        "6" { Run-Task "vision-demo" }
        "7" { Run-Task "vision-test" }
        "8" { Run-Task "test-all" }
        "0" { exit 0 }
        default { Write-Host "无效选项" -ForegroundColor Yellow }
    }
}
