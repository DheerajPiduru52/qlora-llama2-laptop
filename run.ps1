# Thin wrapper: activates the project env for this process, then calls the CLI.
#   powershell -ExecutionPolicy Bypass -File run.ps1 env-check
#   powershell -ExecutionPolicy Bypass -File run.ps1 audit-c
$ErrorActionPreference = 'Stop'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'env.ps1')
& qlora @args
exit $LASTEXITCODE
