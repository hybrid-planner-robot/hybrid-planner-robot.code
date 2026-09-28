"""Thread-local LLM / Fast Downward timing recorder."""

from __future__ import annotations

import time

from planner.call_timings import (
    fd_span,
    invoke_llm,
    llm_span,
    llm_stage,
    reset,
    snapshot,
)


def test_nested_llm_span_records_once():
    reset()
    with llm_stage("select"):
        with llm_span():
            time.sleep(0.02)
            with llm_span():
                time.sleep(0.01)
    snap = snapshot()
    assert snap["n_llm_calls"] == 1
    assert snap["llm_calls"][0]["name"] == "select"
    assert snap["llm_s"] >= 0.02
    assert snap["by_name"]["select"] == snap["llm_s"]


def test_invoke_llm_names_and_sums():
    reset()
    invoke_llm("refuse", lambda: time.sleep(0.01) or "a")
    invoke_llm("intent", lambda: time.sleep(0.01) or "b")
    snap = snapshot()
    assert [c["name"] for c in snap["llm_calls"]] == ["refuse", "intent"]
    assert snap["n_llm_calls"] == 2
    assert snap["llm_s"] == round(
        snap["llm_calls"][0]["s"] + snap["llm_calls"][1]["s"], 3
    )
    assert snap["fd_s"] is None


def test_fd_span_nested_records_once():
    reset()
    with fd_span():
        time.sleep(0.01)
        with fd_span():
            time.sleep(0.01)
    snap = snapshot()
    assert snap["n_fd_calls"] == 1
    assert snap["fd_calls"][0]["name"] == "fast_downward"
    assert snap["fd_s"] >= 0.01
