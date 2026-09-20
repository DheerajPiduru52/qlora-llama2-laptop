from pathlib import Path

import pytest

from qlora_pipeline import paths


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "qroot"
    r.mkdir()
    monkeypatch.setenv("QLORA_ROOT", str(r))
    return r.resolve()


def test_relative_paths_resolve_under_root(root):
    assert paths.ensure_under_root("runs/x") == root / "runs" / "x"


@pytest.mark.parametrize("bad", ["..", "../elsewhere", "runs/../../escape"])
def test_escape_via_dotdot_rejected(root, bad):
    with pytest.raises(paths.PathOutsideRootError):
        paths.ensure_under_root(bad)


def test_absolute_outside_rejected(root, tmp_path):
    with pytest.raises(paths.PathOutsideRootError):
        paths.ensure_under_root(tmp_path / "sibling")


def test_named_dirs_all_under_root(root):
    for fn in (
        paths.repo_dir, paths.configs_dir, paths.data_raw_dir, paths.data_processed_dir, paths.runs_dir,
        paths.adapters_dir, paths.merged_dir, paths.reports_dir, paths.mlflow_dir, paths.tmp_dir, paths.logs_dir,
    ):  # fmt: skip
        p = fn()
        assert root in p.parents, fn.__name__


def test_missing_root_env(monkeypatch):
    monkeypatch.delenv("QLORA_ROOT", raising=False)
    with pytest.raises(RuntimeError):
        paths.qlora_root()


def test_no_hardcoded_drive_letter_in_code():
    src = Path(paths.__file__).parent
    for f in src.glob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert "G:\\" not in text and "G:/" not in text, f.name
