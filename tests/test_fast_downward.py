"""Fast Downward search time limit (unsolvable R1 must not hang A*)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from planner.call_timings import reset, snapshot
from planner.fast_downward import (
    DEFAULT_SEARCH_TIME_LIMIT_S,
    FastDownwardPlanner,
    FastDownwardTimeout,
    resolve_search_time_limit_s,
    result_from_timeout,
)


def test_resolve_search_time_limit_defaults_to_30s(monkeypatch):
    monkeypatch.delenv("VLMRP_FD_SEARCH_TIME_LIMIT_S", raising=False)
    assert resolve_search_time_limit_s() == DEFAULT_SEARCH_TIME_LIMIT_S == 30


def test_resolve_search_time_limit_env(monkeypatch):
    monkeypatch.setenv("VLMRP_FD_SEARCH_TIME_LIMIT_S", "15")
    assert resolve_search_time_limit_s() == 15


def test_solve_passes_search_time_limit(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["timeout"] = kwargs.get("timeout")
        fake = MagicMock()
        fake.returncode = 1
        fake.stdout = ""
        fake.stderr = ""
        return fake

    monkeypatch.setattr(subprocess, "run", fake_run)
    domain = tmp_path / "d.pddl"
    problem = tmp_path / "p.pddl"
    domain.write_text("(define (domain d))")
    problem.write_text("(define (problem p))")
    planner = FastDownwardPlanner(search_time_limit_s=12)
    assert planner.solve(str(domain), str(problem)) is None
    assert "--search-time-limit" in captured["cmd"]
    assert "12s" in captured["cmd"]
    assert captured["timeout"] == 12 + 15


def test_solve_records_fd_span(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        fake = MagicMock()
        fake.returncode = 1
        fake.stdout = ""
        fake.stderr = ""
        return fake

    monkeypatch.setattr(subprocess, "run", fake_run)
    domain = tmp_path / "d.pddl"
    problem = tmp_path / "p.pddl"
    domain.write_text("(define (domain d))")
    problem.write_text("(define (problem p))")
    reset()
    FastDownwardPlanner(search_time_limit_s=12).solve(str(domain), str(problem))
    snap = snapshot()
    assert snap["n_fd_calls"] == 1
    assert snap["fd_s"] is not None
    assert snap["fd_s"] >= 0.0
    assert snap["fd_calls"][0]["name"] == "fast_downward"


def test_solve_raises_timeout_on_watchdog(monkeypatch, tmp_path):
    def fake_run(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="fast-downward", timeout=10)

    monkeypatch.setattr(subprocess, "run", fake_run)
    domain = tmp_path / "d.pddl"
    problem = tmp_path / "p.pddl"
    domain.write_text("d")
    problem.write_text("p")
    with pytest.raises(FastDownwardTimeout) as exc:
        FastDownwardPlanner(search_time_limit_s=10).solve(str(domain), str(problem))
    assert exc.value.limit_s == 10
    payload = result_from_timeout(exc.value.limit_s)
    assert payload["success"] is False
    assert "10s" in payload["error"]
