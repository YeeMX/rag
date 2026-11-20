# 一键创建三个隔离的 Python 环境（Windows PowerShell 5+/pwsh 7）
#
# 用法（在仓库根目录或任意位置执行都行）：
#   pwsh -File scripts\setup_envs.ps1              # 创建全部三个环境
#   pwsh -File scripts\setup_envs.ps1 -Only rag    # 只建 RAG/Agent 环境
#   pwsh -File scripts\setup_envs.ps1 -Force       # 环境已存在也重新创建
#
# 说明：
# - 三个环境均创建在仓库根目录下：.venv-rag / .venv-vision / .venv-paddle
# - 依赖版本来自 *.lock.txt（与本机实测环境一致），首次安装会下载约 2.5GB，之后走 pip 缓存
param(
    [ValidateSet("all", "rag", "vision", "paddle")]
    [string]$Only = "all",
    [string]$PythonBase = "",
    [switch]$Force,
    [switch]$UseLocks
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent

# 默认用宽松的 requirements（跨机器兼容性最好）；-UseLocks 用本机实测锁定版本，便于精确复现。
$lockSuffix = if ($UseLocks) { ".lock" } else { "" }
$reqFileName = if ($UseLocks) { "requirements.lock.txt" } else { "requirements.txt" }

function Resolve-BasePython {
    if ($PythonBase) {
        $ver = & $PythonBase -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -ne 0) { throw "无法执行 -PythonBase 指定的解释器：$PythonBase" }
        return $PythonBase
    }
    # 必须用 Python 3.10/3.11（torch/paddle 的 wheel 兼容性最好）；不要用系统默认的 3.13/3.14
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "python",
        "python3"
    )
    foreach ($cand in $candidates) {
        try {
            $ver = (& $cand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null).Trim()
            if ($LASTEXITCODE -eq 0 -and $ver -match "^(3\.10|3\.11)$") {
                Write-Host "使用 Python $ver : $cand"
                return $cand
            }
        } catch { }
    }
    throw "未找到 Python 3.10/3.11。请用 -PythonBase 指定，例如：-PythonBase `"$env:LOCALAPPDATA\Programs\Python\Python311\python.exe`""
}

$base = Resolve-BasePython

$specs = [ordered]@{
    rag    = @{ Venv = "$root\.venv-rag";    Req = "$root\01-labor-law-rag\$reqFileName" }
    vision = @{ Venv = "$root\.venv-vision"; Req = "$root\03-vision-ocr-demo\requirements-yolo$lockSuffix.txt" }
    paddle = @{ Venv = "$root\.venv-paddle"; Req = "$root\03-vision-ocr-demo\requirements-ocr$lockSuffix.txt" }
}

foreach ($name in $specs.Keys) {
    if ($Only -ne "all" -and $Only -ne $name) { continue }
    $spec = $specs[$name]
    if ((Test-Path $spec.Venv) -and $Force) {
        Write-Host "[$name] 删除旧环境 $($spec.Venv)"
        Remove-Item -Recurse -Force $spec.Venv
    }
    if (-not (Test-Path $spec.Venv)) {
        Write-Host "[$name] 创建 venv -> $($spec.Venv)"
        & $base -m venv $spec.Venv
    } else {
        Write-Host "[$name] 已存在，跳过创建：$($spec.Venv)"
    }
    $py = "$($spec.Venv)\Scripts\python.exe"
    Write-Host "[$name] 安装依赖（$(Split-Path $spec.Req -Leaf)）..."
    & $py -m pip install --upgrade pip --quiet
    & $py -m pip install -r $spec.Req
    if ($LASTEXITCODE -ne 0) { throw "[$name] 依赖安装失败" }
    Write-Host "[$name] 完成"
}

Write-Host ""
Write-Host "全部环境就绪："
foreach ($name in $specs.Keys) {
    if ($Only -ne "all" -and $Only -ne $name) { continue }
    $py = "$($specs[$name].Venv)\Scripts\python.exe"
    if (Test-Path $py) {
        $v = & $py -c "import sys; print(sys.version.split()[0])" 2>$null
        Write-Host ("  {0,-8} {1}  Python {2}" -f $name, $specs[$name].Venv, $v)
    }
}
Write-Host ""
Write-Host "下一步：pwsh -File start.ps1"
