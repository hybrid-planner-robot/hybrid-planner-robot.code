"""Thread-local LLM / Fast Downward wall-clock recorder.

Nested spans record once (outermost wins) so ``generate_fn`` → ``complete``
does not double-count. Stage names come from :func:`llm_stage` /
:func:`invoke_llm`.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

_T = TypeVar("_T")

_tls = threading.local()


def _state() -> threading.local:
    if not hasattr(_tls, "llm_calls"):
        reset()
    return _tls


def reset() -> None:
    _tls.llm_calls = []
    _tls.fd_calls = []
    _tls.stage_stack = []
    _tls.llm_depth = 0
    _tls.fd_depth = 0


def empty_timings() -> dict[str, Any]:
    return {
        "llm_calls": [],
        "llm_s": 0.0,
        "n_llm_calls": 0,
        "fd_calls": [],
        "fd_s": None,
        "n_fd_calls": 0,
        "by_name": {},
    }


def snapshot() -> dict[str, Any]:
    st = _state()
    llm = [dict(c) for c in st.llm_calls]
    fd = [dict(c) for c in st.fd_calls]
    by_name: dict[str, float] = {}
    for call in llm:
        name = str(call.get("name") or "llm")
        by_name[name] = round(float(by_name.get(name, 0.0)) + float(call["s"]), 3)
    return {
        "llm_calls": llm,
        "llm_s": round(sum(float(c["s"]) for c in llm), 3),
        "n_llm_calls": len(llm),
        "fd_calls": fd,
        "fd_s": round(sum(float(c["s"]) for c in fd), 3) if fd else None,
        "n_fd_calls": len(fd),
        "by_name": by_name,
    }


def from_payload(raw: Any) -> dict[str, Any]:
    """Rebuild a timings dict from summary.json (legacy runs may omit it)."""
    if not isinstance(raw, dict):
        return empty_timings()
    llm: list[dict[str, Any]] = []
    for item in raw.get("llm_calls") or []:
        if not isinstance(item, dict) or item.get("s") is None:
            continue
        llm.append(
            {
                "name": str(item.get("name") or "llm"),
                "s": round(float(item["s"]), 3),
            }
        )
    fd: list[dict[str, Any]] = []
    for item in raw.get("fd_calls") or []:
        if not isinstance(item, dict) or item.get("s") is None:
            continue
        fd.append(
            {
                "name": str(item.get("name") or "fast_downward"),
                "s": round(float(item["s"]), 3),
            }
        )
    by_name: dict[str, float] = {}
    for call in llm:
        by_name[call["name"]] = round(
            float(by_name.get(call["name"], 0.0)) + float(call["s"]), 3
        )
    fd_s = raw.get("fd_s")
    if fd:
        fd_s = round(sum(float(c["s"]) for c in fd), 3)
    elif fd_s is not None:
        fd_s = round(float(fd_s), 3)
    llm_s = raw.get("llm_s")
    if llm:
        llm_s = round(sum(float(c["s"]) for c in llm), 3)
    elif llm_s is not None:
        llm_s = round(float(llm_s), 3)
    else:
        llm_s = 0.0
    return {
        "llm_calls": llm,
        "llm_s": llm_s,
        "n_llm_calls": len(llm),
        "fd_calls": fd,
        "fd_s": fd_s,
        "n_fd_calls": len(fd),
        "by_name": by_name or dict(raw.get("by_name") or {}),
    }


def attach_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    payload["timings"] = snapshot()
    return payload


def last_llm_s() -> float:
    st = _state()
    if not st.llm_calls:
        return 0.0
    return float(st.llm_calls[-1]["s"])


@contextmanager
def llm_stage(name: str) -> Iterator[None]:
    st = _state()
    st.stage_stack.append(name)
    try:
        yield
    finally:
        st.stage_stack.pop()


@contextmanager
def llm_span(name: str | None = None) -> Iterator[None]:
    st = _state()
    nested = st.llm_depth > 0
    st.llm_depth += 1
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        st.llm_depth -= 1
        if not nested:
            label = name or (
                st.stage_stack[-1] if st.stage_stack else f"llm_{len(st.llm_calls) + 1}"
            )
            st.llm_calls.append({"name": str(label), "s": round(elapsed, 3)})


@contextmanager
def fd_span() -> Iterator[None]:
    st = _state()
    nested = st.fd_depth > 0
    st.fd_depth += 1
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        st.fd_depth -= 1
        if not nested:
            st.fd_calls.append(
                {"name": "fast_downward", "s": round(elapsed, 3)}
            )


def invoke_llm(name: str, fn: Callable[[], _T]) -> _T:
    with llm_stage(name):
        with llm_span():
            return fn()
