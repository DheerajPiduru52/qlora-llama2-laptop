import json

import pytest

from qlora_pipeline import audit


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    root = tmp_path / "qroot"
    root.mkdir()
    monkeypatch.setenv("QLORA_ROOT", str(root))
    base = {label: 0 for label, _, _ in audit.LOCATIONS}

    def make_snapshot(sizes):
        return {
            "taken_at": "t",
            "drive": "C:\\",
            "free_bytes": 0,
            "locations": {k: {"path": None, "exists": True, "bytes": v} for k, v in sizes.items()},
        }

    audit.baseline_path().write_text(json.dumps(make_snapshot(base)), encoding="utf-8")
    return base, make_snapshot, monkeypatch


def _run(fake, grow_label, mb):
    base, make_snapshot, mp = fake
    sizes = dict(base)
    sizes[grow_label] = int(mb * audit.MB)
    mp.setattr(audit, "snapshot", lambda: make_snapshot(sizes))
    ok, rows, _ = audit.compare()
    return ok, {r["location"]: r["status"] for r in rows}


def test_temp_growth_is_warn_only(fake):
    ok, st = _run(fake, r"%LOCALAPPDATA%\Temp", 500)
    assert ok and st[r"%LOCALAPPDATA%\Temp"] == "WARN"


@pytest.mark.parametrize("label", [lbl for lbl, _, _ in audit.LOCATIONS if lbl not in audit.WARN_ONLY])
def test_other_locations_hard_fail(fake, label):
    ok, st = _run(fake, label, 51)
    assert not ok and st[label] == "FAIL"


def test_under_threshold_ok(fake):
    ok, st = _run(fake, r"%LOCALAPPDATA%\pip", 49)
    assert ok and set(st.values()) == {"ok"}
