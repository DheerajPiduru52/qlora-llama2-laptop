"""Load qlora.env + repo/.env into os.environ. MUST run before torch/transformers/datasets/hf_hub import,
otherwise HF_HOME and friends are ignored. Stdlib + python-dotenv only.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False

# repo/src/qlora_pipeline/bootstrap.py -> parents[2] = repo, parents[3] = QLORA_ROOT
_REPO = Path(__file__).resolve().parents[2]


def _find_root() -> Path:
    env_root = os.environ.get("QLORA_ROOT")
    if env_root and (Path(env_root) / "qlora.env").is_file():
        return Path(env_root)
    guess = _REPO.parent
    if (guess / "qlora.env").is_file():
        return guess
    raise RuntimeError("Cannot locate qlora.env. Set QLORA_ROOT or activate env.ps1 / env.sh.")


def load_env() -> Path:
    """Idempotent. Values already set in the process win (activation scripts / explicit overrides)."""
    global _LOADED
    from dotenv import load_dotenv

    root = _find_root()
    if not _LOADED:
        load_dotenv(root / "qlora.env", override=False, encoding="utf-8")
        secrets = root / "repo" / ".env"
        if secrets.is_file():
            load_dotenv(secrets, override=False, encoding="utf-8")
        _LOADED = True
    return root
