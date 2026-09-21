"""Serve API contract tests with a fake engine (no GPU, no model load)."""

import threading

import pytest
from fastapi.testclient import TestClient

from qlora_pipeline.config import load_config
from qlora_pipeline.serve import create_app


class FakeEngine:
    info = {"loaded": "fake"}

    def __init__(self):
        self.calls = []

    def generate(self, text, max_new_tokens, temperature, top_p):
        self.calls.append((text, max_new_tokens, temperature, top_p))
        if text == "too long":
            raise ValueError("input is 2000 tokens; the limit is 1024")
        return f"reply to: {text}", 7

    def stats(self):
        return {"peak_reserved_gb": 1.0, "peak_allocated_gb": 0.5, "reserved_gb": 0.5, "physical_vram_gb": 16.0}

    def reset_peak(self):
        pass


@pytest.fixture()
def cfg():
    return load_config("serve.yaml")


@pytest.fixture()
def client(cfg):
    eng = FakeEngine()
    with TestClient(create_app(cfg, engine=eng)) as c:
        c.engine = eng
        yield c


def test_ping_ok_when_loaded(client):
    r = client.get("/ping")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_ping_503_while_loading(cfg, monkeypatch):
    import qlora_pipeline.serve as srv

    gate = threading.Event()
    monkeypatch.setattr(srv, "load_engine", lambda _cfg: (gate.wait(5), FakeEngine())[1])
    with TestClient(create_app(cfg)) as c:
        assert c.get("/ping").status_code == 503
        assert c.post("/invocations", json={"inputs": "hi"}).status_code == 503
        gate.set()


def test_invocations_contract(client, cfg):
    r = client.post("/invocations", json={"inputs": "where is my order?",
                                          "parameters": {"max_new_tokens": 64, "temperature": 0.7}})  # fmt: skip
    assert r.status_code == 200
    assert r.json() == {"generated_text": "reply to: where is my order?"}
    assert r.headers["X-Generated-Tokens"] == "7" and float(r.headers["X-Inference-Ms"]) >= 0
    assert client.engine.calls[-1] == ("where is my order?", 64, 0.7, 1.0)


def test_defaults_are_greedy_and_configured(client, cfg):
    client.post("/invocations", json={"inputs": "hello"})
    assert client.engine.calls[-1] == ("hello", cfg.serve.default_max_new_tokens, 0.0, 1.0)


@pytest.mark.parametrize("body", [
    {}, {"inputs": ""}, {"inputs": "x", "parameters": {"temperature": -1}},
    {"inputs": "x", "parameters": {"max_new_tokens": 0}}, {"inputs": "x", "parameters": {"bogus": 1}},
    {"inputs": "x", "extra": 1},
])  # fmt: skip
def test_invalid_requests_rejected(client, body):
    assert client.post("/invocations", json=body).status_code == 422


def test_max_new_tokens_limit(client, cfg):
    too_many = cfg.serve.max_new_tokens_limit + 1
    r = client.post("/invocations", json={"inputs": "x", "parameters": {"max_new_tokens": too_many}})
    assert r.status_code == 400 and "limit" in r.json()["error"]


def test_engine_value_error_is_400(client):
    r = client.post("/invocations", json={"inputs": "too long"})
    assert r.status_code == 400 and "limit is 1024" in r.json()["error"]


def test_stats_and_reset(client):
    s = client.get("/stats").json()
    assert s["loaded"] and s["info"] == {"loaded": "fake"} and s["peak_reserved_gb"] == 1.0
    assert client.post("/stats/reset-peak").json() == {"ok": True}


def test_serve_config_points_at_champion(cfg):
    assert cfg.serve.port == 8080 and cfg.serve.source == "registry" and cfg.serve.fallback_to_merged
    assert cfg.merge.alias == "champion" and cfg.serve.default_temperature == 0.0
