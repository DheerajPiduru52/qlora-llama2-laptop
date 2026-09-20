# Samples C: free space, system commit, and pagefile size until -StopFile exists (or -MaxMinutes pass).
# One CSV line per sample, so min/max over a job can be computed afterwards:
#   powershell -ExecutionPolicy Bypass -File scripts\watch_system.ps1 -Out <csv> -StopFile <flag>
param(
  [Parameter(Mandatory = $true)][string]$Out,
  [Parameter(Mandatory = $true)][string]$StopFile,
  [double]$IntervalSec = 1.0,
  [int]$MaxMinutes = 240
)
$drive = $env:SystemDrive
"time,c_free_gb,commit_gb,commit_limit_gb,pagefile_mb" | Set-Content -Path $Out -Encoding utf8
$end = (Get-Date).AddMinutes($MaxMinutes)
while (-not (Test-Path $StopFile) -and (Get-Date) -lt $end) {
  try {
    $c = (Get-Counter '\Memory\Committed Bytes', '\Memory\Commit Limit' -ErrorAction Stop).CounterSamples
    $pf = (Get-CimInstance Win32_PageFileUsage | Where-Object { $_.Name -like "$drive*" } |
           Measure-Object AllocatedBaseSize -Sum).Sum
    $free = (Get-PSDrive ($drive.TrimEnd(':'))).Free / 1GB
    "{0},{1:F3},{2:F3},{3:F3},{4}" -f (Get-Date -Format o), $free, ($c[0].CookedValue / 1GB),
      ($c[1].CookedValue / 1GB), $pf | Add-Content -Path $Out -Encoding utf8
  } catch { }
  Start-Sleep -Milliseconds ([int]($IntervalSec * 1000))
}
