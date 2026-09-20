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
    report = {
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
        print(f"  C: free {snap['free_bytes'] / 1024**3:.2f} GB")
        return 0
    ok, rows, report = compare(a.threshold_mb)
    print_report(rows, report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
