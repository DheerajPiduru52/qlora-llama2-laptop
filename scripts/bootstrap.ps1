# One-time Phase 0 bootstrap. Run with process-scoped policy only:
#   powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
# Prereqs: qlora.env + env.ps1 at the project root (repo's parent), uv.exe in <root>\tools\uv,
# and any existing Python 3.x on PATH (used read-only to take the C: baseline BEFORE installing anything).
# Installs nothing system-wide; every path comes from qlora.env (incl. UV_PYTHON_INSTALL_REGISTRY=0,
# UV_PYTHON_BIN_DIR on G:).
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
. (Join-Path (Split-Path -Parent $repo) 'env.ps1')
$root = $env:QLORA_ROOT

function Assert-Ok([string]$what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

$dirs = @(
  'tools\uv', 'tools\python\bin', 'tools\uv-tools\bin', 'repo',
  'cache\huggingface', 'cache\pip', 'cache\uv', 'cache\torch', 'cache\triton', 'cache\triton_home',
  'cache\nv_compute', 'cache\matplotlib', 'cache\nltk', 'cache\xdg',
  'data\raw', 'data\processed', 'runs',
  'artifacts\adapters', 'artifacts\merged', 'artifacts\reports',
  'mlflow\artifacts', 'tmp', 'logs'
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path (Join-Path $root $d) | Out-Null }

$uv = Join-Path $root 'tools\uv\uv.exe'
if (-not (Test-Path $uv)) {
  throw "uv.exe missing. Download the standalone Windows zip from https://github.com/astral-sh/uv/releases and extract uv.exe into $root\tools\uv (do NOT use the installer script; it writes to C:)."
}

# 1. C: baseline BEFORE any install. audit.py is stdlib-only, so any existing Python works.
$baseline = Join-Path $root 'logs\c_baseline.json'
if (-not (Test-Path $baseline)) {
  $sysPy = (Get-Command python -ErrorAction SilentlyContinue).Source
  if (-not $sysPy) { throw 'No Python on PATH to take the C: baseline before installing. Install nothing; ask first.' }
  $env:PYTHONPATH = Join-Path $repo 'src'
  & $sysPy -B -m qlora_pipeline.audit --baseline
  Assert-Ok 'C: baseline'
  Remove-Item Env:PYTHONPATH
}
if (-not (Test-Path $baseline)) { throw "Baseline missing at $baseline; refusing to install." }

# 2. Python 3.11 into tools\python, then the venv + stack.
& $uv python install 3.11
Assert-Ok 'uv python install'
Push-Location $repo
try {
  & $uv sync
  Assert-Ok 'uv sync'
} finally { Pop-Location }

Write-Host "Bootstrap done. Next: . $root\env.ps1 ; qlora env-check ; qlora audit-c"
