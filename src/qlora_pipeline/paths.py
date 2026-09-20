"""Every project path resolves from QLORA_ROOT. Stdlib only (importable before the venv exists).

Nothing here hardcodes a drive letter; qlora.env is the only place that does.
"""

from __future__ import annotations

import os
from pathlib import Path


class PathOutsideRootError(ValueError):
    """Raised when a resolved output path would land outside QLORA_ROOT."""


def qlora_root() -> Path:
    raw = os.environ.get("QLORA_ROOT")
    if not raw:
        raise RuntimeError(
            "QLORA_ROOT is not set. Activate the env first: `. <root>\\env.ps1` or `source <root>/env.sh`."
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"QLORA_ROOT={root} does not exist or is not a directory.")
    return root


def ensure_under_root(path: str | os.PathLike[str], root: Path | None = None) -> Path:
    """Resolve `path` (relative paths are taken relative to the root) and require it to be inside root."""
    root = (root or qlora_root()).resolve()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if p != root and root not in p.parents:
        raise PathOutsideRootError(f"{p} is outside QLORA_ROOT ({root}).")
    return p


def _sub(*parts: str) -> Path:
    return ensure_under_root(Path(*parts))


def repo_dir() -> Path:
    return _sub("repo")


def configs_dir() -> Path:
    return _sub("repo", "configs")


def cache_dir(name: str = "") -> Path:
    return _sub("cache", name) if name else _sub("cache")


def data_raw_dir() -> Path:
    return _sub("data", "raw")


def data_processed_dir() -> Path:
    return _sub("data", "processed")


def runs_dir() -> Path:
    return _sub("runs")


def adapters_dir() -> Path:
    return _sub("artifacts", "adapters")


def merged_dir() -> Path:
    return _sub("artifacts", "merged")


def reports_dir() -> Path:
    return _sub("artifacts", "reports")


def mlflow_dir() -> Path:
    return _sub("mlflow")


def tmp_dir() -> Path:
    return _sub("tmp")


def logs_dir() -> Path:
    return _sub("logs")


def ensure_dir(path: Path) -> Path:
    p = ensure_under_root(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
