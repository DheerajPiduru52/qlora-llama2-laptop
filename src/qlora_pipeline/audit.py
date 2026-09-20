"""C: drive footprint audit. Stdlib only so the baseline can be taken before anything is installed.

Usage (env active):
    python -B -m qlora_pipeline.audit --baseline     # record logs/c_baseline.json
    python -B -m qlora_pipeline.audit                # compare against baseline, exit 1 on FAIL
The `qlora audit-c` CLI subcommand calls the same functions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from qlora_pipeline.paths import ensure_dir, logs_dir

THRESHOLD_MB = 50.0
MB = 1024 * 1024
_REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT

# (label, env var, sub-path). Expanded at runtime; nothing hardcoded.
LOCATIONS: list[tuple[str, str, str]] = [
    (r"%USERPROFILE%\.cache", "USERPROFILE", ".cache"),
    (r"%LOCALAPPDATA%\pip", "LOCALAPPDATA", "pip"),
    (r"%LOCALAPPDATA%\uv", "LOCALAPPDATA", "uv"),
    (r"%APPDATA%\NVIDIA\ComputeCache", "APPDATA", r"NVIDIA\ComputeCache"),
    (r"%LOCALAPPDATA%\Temp", "LOCALAPPDATA", "Temp"),
    (r"%USERPROFILE%\.cache\huggingface", "USERPROFILE", r".cache\huggingface"),
    (r"%USERPROFILE%\.local", "USERPROFILE", ".local"),
]

# Shared with every other app on the machine (browsers, Windows, Claude), so growth there is
# reported as WARN and does not fail the audit. All other locations are hard fails.
WARN_ONLY = {r"%LOCALAPPDATA%\Temp"}

# The system-managed pagefile on C: grows when GPU jobs push Windows commit charge past its limit. It is not a
# project file, so it is WARN-only, but it is real C: consumption caused by our workload.
PAGEFILE_LABEL = r"C:\pagefile.sys (allocated)"


def pagefile_mb() -> int | None:
    """Allocated size of the pagefile(s) on the system drive, via CIM (no admin needed). None if unavailable."""
    drive = os.environ.get("SystemDrive", "C:")
    cmd = (f"(Get-CimInstance Win32_PageFileUsage | Where-Object {{ $_.Name -like '{drive}*' }} | "
           "Measure-Object AllocatedBaseSize -Sum).Sum")  # fmt: skip
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                           capture_output=True, text=True, timeout=60)  # fmt: skip
        out = r.stdout.strip()
        if r.returncode != 0 or (not out and r.stderr.strip()):
            return None
        return int(float(out)) if out else 0  # empty + clean exit = no pagefile on this drive
    except Exception:  # noqa: BLE001
        return None


def commit_stats() -> dict[str, float]:
    """System-wide commit charge (GB) from GetPerformanceInfo; empty off Windows."""
    if os.name != "nt":
        return {}
    import ctypes
    from ctypes import wintypes

    class PERFORMANCE_INFORMATION(ctypes.Structure):  # noqa: N801
        _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                    ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                    ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                    ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                    ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD),
                    ("ThreadCount", wintypes.DWORD)]  # fmt: skip

    pi = PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        return {}
    g = pi.PageSize / 1024**3
    return {"commit_total_gb": round(pi.CommitTotal * g, 2), "commit_limit_gb": round(pi.CommitLimit * g, 2),
            "commit_peak_gb": round(pi.CommitPeak * g, 2), "physical_total_gb": round(pi.PhysicalTotal * g, 2),
            "physical_available_gb": round(pi.PhysicalAvailable * g, 2)}  # fmt: skip


def process_commit_gb() -> dict[str, float]:
    """This process's private commit now and its lifetime peak (GB), via K32GetProcessMemoryInfo."""
    if os.name != "nt":
        return {}
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]  # fmt: skip

    c = PMC()
    c.cb = ctypes.sizeof(c)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return {}
    return {"private_gb": c.PrivateUsage / 1024**3, "peak_private_gb": c.PeakPagefileUsage / 1024**3}


def top_commit_processes(n: int = 5) -> list[dict]:
    """Top-n processes by private commit (what to close to free commit headroom)."""
    cmd = ("Get-Process | Sort-Object PrivateMemorySize64 -Descending | Select-Object -First "
           f"{n} Name,Id,PrivateMemorySize64 | ConvertTo-Json -Compress")  # fmt: skip
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                           capture_output=True, text=True, timeout=60)  # fmt: skip
        rows = json.loads(r.stdout) if r.stdout.strip() else []
        rows = rows if isinstance(rows, list) else [rows]
        return [{"name": x["Name"], "pid": x["Id"], "commit_gb": round(x["PrivateMemorySize64"] / 1024**3, 2)}
                for x in rows]  # fmt: skip
    except Exception:  # noqa: BLE001
        return []


def peaks_path() -> Path:
    return ensure_dir(logs_dir()) / "commit_peaks.json"


def job_peak_gb(job_key: str) -> dict | None:
    p = peaks_path()
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get(job_key)


def record_job_peak(job_key: str, peak_gb: float, note: str = "", lower_bound: bool = False) -> dict:
    """Keep the MAX observed full-run process peak per job (conservative), plus the latest measurement.

    An entry marked lower_bound (e.g. from a run stopped part-way) is REPLACED by the first full measurement
    instead of being max-ed with it. Recording another lower bound never lowers an existing value.
    """
    p = peaks_path()
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    prev = data.get(job_key, {})
    now = datetime.now().isoformat(timespec="seconds")
    replacing = bool(prev.get("lower_bound")) and not lower_bound
    peak = peak_gb if replacing else max(peak_gb, prev.get("peak_gb", 0.0))
    if replacing:
        note = f"{note} [replaced lower bound {prev.get('peak_gb')} GB]".strip()
    entry = {
        "peak_gb": round(peak, 2),
        "lower_bound": lower_bound and (not prev or bool(prev.get("lower_bound"))),
        "latest_gb": round(peak_gb, 2), "latest_at": now, "latest_note": note,
        "history": [*prev.get("history", []), {"gb": round(peak_gb, 2), "at": now, "note": note,
                                               "lower_bound": lower_bound}][-10:],
    }  # fmt: skip
    data[job_key] = entry
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return entry


class CommitHeadroomError(RuntimeError):
    pass


def pagefile_settings() -> dict[str, int | None]:
    """System-drive pagefile: currently allocated MB and configured maximum MB (None = system managed/unknown)."""
    drive = os.environ.get("SystemDrive", "C:")
    cmd = (f"$s = Get-CimInstance Win32_PageFileSetting | Where-Object {{ $_.Name -like '{drive}*' }}; "
           "if ($s) { [string]($s | Measure-Object MaximumSize -Sum).Sum } else { '' }")  # fmt: skip
    max_mb: int | None = None
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                           capture_output=True, text=True, timeout=60)  # fmt: skip
        out = r.stdout.strip()
        if r.returncode == 0 and out:
            max_mb = int(float(out)) or None  # 0 = system managed: no fixed cap
    except Exception:  # noqa: BLE001
        pass
    return {"allocated_mb": pagefile_mb(), "max_mb": max_mb}


def require_commit_headroom(job_key: str, default_gb: float = 18.0, margin_gb: float = 3.0,
                            c_growth_margin_gb: float = 10.0) -> dict:  # fmt: skip
    """Refuse to start unless both hold:
    1. commit headroom against the CAPPED limit (RAM + pagefile maximum) minus commit in use
       >= the job's measured peak (or default) + margin;
    2. C: free >= remaining pagefile growth (max - currently allocated) + c_growth_margin_gb,
       so the pagefile can grow to its cap without filling C:.
    Without a fixed pagefile cap, check 1 falls back to the current commit limit and check 2 is skipped."""
    cs = commit_stats()
    if not cs:
        return {}
    pf = pagefile_settings()
    rec = job_peak_gb(job_key)
    basis = rec["peak_gb"] if rec else default_gb
    kind = "lower-bound peak" if rec and rec.get("lower_bound") else "measured peak"
    src = f"{kind} {basis:.1f} GB" if rec else f"default {default_gb:.0f} GB (not yet measured)"
    need = basis + margin_gb
    if pf["max_mb"] and pf["allocated_mb"] is not None:
        limit = cs["physical_total_gb"] + pf["max_mb"] / 1024
        limit_src = f"RAM {cs['physical_total_gb']:.1f} + pagefile max {pf['max_mb'] / 1024:.1f}"
        growth_gb = max(pf["max_mb"] - pf["allocated_mb"], 0) / 1024
    else:
        limit, limit_src, growth_gb = cs["commit_limit_gb"], "current commit limit (no fixed pagefile cap)", None
    headroom = limit - cs["commit_total_gb"]
    free = c_free_gb()
    info = {"job": job_key, "need_gb": round(need, 2), "headroom_gb": round(headroom, 2), "basis": src,
            "limit_gb": round(limit, 2), "limit_source": limit_src, "pagefile": pf,
            "pagefile_growth_left_gb": None if growth_gb is None else round(growth_gb, 2),
            "c_free_gb": round(free, 2), **cs}  # fmt: skip
    if headroom < need:
        top = top_commit_processes(5)
        lines = "\n".join(f"    {t['commit_gb']:6.2f} GB  {t['name']} (pid {t['pid']})" for t in top)
        raise CommitHeadroomError(
            f"Refusing to start {job_key}: commit headroom {headroom:.1f} GB (limit {limit:.1f} = {limit_src}; "
            f"in use {cs['commit_total_gb']:.1f}) < needed {need:.1f} GB ({src} + {margin_gb:.0f} GB margin).\n"
            f"  Top 5 processes by commit (close some to free headroom):\n{lines}"
        )
    if growth_gb is not None and free < growth_gb + c_growth_margin_gb:
        raise LowDiskError(
            f"Refusing to start {job_key}: C: has {free:.1f} GB free, but the pagefile may still grow "
            f"{growth_gb:.1f} GB (to its {pf['max_mb']} MB cap) and {c_growth_margin_gb:.0f} GB must stay free: "
            f"need {growth_gb + c_growth_margin_gb:.1f} GB."
        )
    return info


def c_free_gb() -> float:
    return shutil.disk_usage(os.environ.get("SystemDrive", "C:") + "\\").free / 1024**3


class LowDiskError(RuntimeError):
    pass


def require_c_free(min_gb: float, what: str) -> float:
    """Pre-flight guard: refuse to start `what` if the system drive is below `min_gb` free."""
    free = c_free_gb()
    if free < min_gb:
        raise LowDiskError(
            f"Refusing to start {what}: C: has {free:.1f} GB free, below the {min_gb:.0f} GB pre-flight minimum. "
            r"Large GPU jobs can grow C:\pagefile.sys by 20+ GB. Free space on C: or move the pagefile to G:."
        )
    return free


def _dir_size(path: Path) -> int:
    """Total bytes under path. Skips symlinks/junctions and unreadable entries (Temp has locked files)."""
    if not path.exists():
        return 0
    total = 0
    stack = [path]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        st = e.stat(follow_symlinks=False)
                        # Skip symlinks AND junctions (DirEntry.is_junction is 3.12+; check the reparse bit).
                        if e.is_symlink() or getattr(st, "st_file_attributes", 0) & _REPARSE_POINT:
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        else:
                            total += st.st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def snapshot() -> dict:
    sys_drive = os.environ.get("SystemDrive", "C:") + "\\"
    usage = shutil.disk_usage(sys_drive)
    locs = {}
    for label, var, sub in LOCATIONS:
        base = os.environ.get(var)
        p = Path(base) / sub if base else None
        locs[label] = {
            "path": str(p) if p else None,
            "exists": bool(p and p.exists()),
            "bytes": _dir_size(p) if p else 0,
        }
    return {
        "taken_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "drive": sys_drive,
        "free_bytes": usage.free,
        "locations": locs,
        "pagefile_mb": pagefile_mb(),
        "commit": commit_stats(),
    }


def baseline_path() -> Path:
    return ensure_dir(logs_dir()) / "c_baseline.json"


def record_baseline(force: bool = False) -> dict:
    bp = baseline_path()
    if bp.exists() and not force:
        raise FileExistsError(f"Baseline already exists at {bp}. Pass --force to overwrite.")
    snap = snapshot()
    bp.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    return snap


def compare(threshold_mb: float = THRESHOLD_MB) -> tuple[bool, list[dict], dict]:
    bp = baseline_path()
    if not bp.exists():
        raise FileNotFoundError(f"No baseline at {bp}. Run `qlora audit-c --baseline` first.")
    base = json.loads(bp.read_text(encoding="utf-8"))
    now = snapshot()
    rows = []
    ok = True
    for label in now["locations"]:
        b = base["locations"].get(label, {"bytes": 0})["bytes"]
        n = now["locations"][label]["bytes"]
        delta_mb = (n - b) / MB
        if delta_mb <= threshold_mb:
            status = "ok"
        elif label in WARN_ONLY:
            status = "WARN"
        else:
            status = "FAIL"
        ok &= status != "FAIL"
        rows.append({"location": label, "baseline_mb": b / MB, "now_mb": n / MB, "delta_mb": delta_mb, "status": status})
    pf_base, pf_src = base.get("pagefile_mb"), "c_baseline.json"
    if pf_base is None and (logs_dir() / "c_pagefile_baseline.json").exists():
        pfb = json.loads((logs_dir() / "c_pagefile_baseline.json").read_text(encoding="utf-8"))
        pf_base, pf_src = pfb["pagefile_mb"], f"c_pagefile_baseline.json ({pfb.get('source', '')})"
    pf_now = now.get("pagefile_mb")
    if pf_base is not None and pf_now is not None:
        d = pf_now - pf_base
        rows.append({"location": PAGEFILE_LABEL, "baseline_mb": float(pf_base), "now_mb": float(pf_now),
                     "delta_mb": float(d), "status": "WARN" if d > threshold_mb else "ok"})  # fmt: skip
    report = {
        "pagefile_baseline_source": pf_src if pf_base is not None else None,
        "baseline_taken_at": base["taken_at"],
        "now": now,
        "free_delta_mb": (now["free_bytes"] - base["free_bytes"]) / MB,
        "rows": rows,
        "threshold_mb": threshold_mb,
        "passed": ok,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (ensure_dir(logs_dir()) / f"c_audit_{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return ok, rows, report


def print_report(rows: list[dict], report: dict) -> None:
    print(f"C: audit vs baseline taken {report['baseline_taken_at']} (threshold {report['threshold_mb']:.0f} MB)")
    print(f"{'location':38} {'baseline MB':>12} {'now MB':>10} {'delta MB':>10}  status")
    for r in rows:
        print(f"{r['location']:38} {r['baseline_mb']:12.1f} {r['now_mb']:10.1f} {r['delta_mb']:+10.1f}  {r['status']}")
    free_gb = report["now"]["free_bytes"] / 1024**3
    print(f"C: free now {free_gb:.2f} GB (change since baseline {report['free_delta_mb']:+.1f} MB, informational)")
    if report.get("pagefile_baseline_source"):
        print(f"pagefile baseline from {report['pagefile_baseline_source']}; pagefile rows are WARN-only")
    cm = report["now"].get("commit") or {}
    if cm:
        print(f"commit charge now {cm['commit_total_gb']} / {cm['commit_limit_gb']} GB limit "
              f"(RAM {cm['physical_total_gb']} GB), informational")
    print("RESULT:", "PASS" if report["passed"] else "FAIL")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C: drive footprint audit")
    ap.add_argument("--baseline", action="store_true", help="record the baseline")
    ap.add_argument("--force", action="store_true", help="overwrite an existing baseline")
    ap.add_argument("--threshold-mb", type=float, default=THRESHOLD_MB)
    a = ap.parse_args(argv)
    if a.baseline:
        snap = record_baseline(force=a.force)
        print(f"Baseline written to {baseline_path()}")
        for label, v in snap["locations"].items():
            print(f"  {label:38} {v['bytes'] / MB:10.1f} MB  {'(missing)' if not v['exists'] else ''}")
        print(f"  C: free {snap['free_bytes'] / 1024**3:.2f} GB; pagefile {snap['pagefile_mb']} MB")
        return 0
    ok, rows, report = compare(a.threshold_mb)
    print_report(rows, report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
