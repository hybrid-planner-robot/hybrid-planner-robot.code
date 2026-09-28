"""Split a PDDL problem into ``:init`` / ``:goal`` and persist FD artefacts.

Used by the host loop so Fast Downward failures still leave an inspectable
problem on disk, and by the planning battery when copying those files into
the eval report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _skip_ws_and_comments(text: str, i: int) -> int:
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\n\r":
            i += 1
            continue
        if c == ";":
            nl = text.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        break
    return i


def matching_paren(text: str, open_idx: int) -> int:
    """Index of the ``)`` matching ``text[open_idx] == '('``, or -1."""
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "(":
        return -1
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == ";":
            nl = text.find("\n", i)
            i = n if nl < 0 else nl
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def extract_section(problem: str, keyword: str) -> str:
    """Return the ``(:init …)`` / ``(:goal …)`` block, or ``\"\"`` if missing."""
    needle = f"({keyword}" if keyword.startswith(":") else f"(:{keyword}"
    idx = problem.find(needle)
    if idx < 0:
        return ""
    end = matching_paren(problem, idx)
    if end < 0:
        return ""
    return problem[idx : end + 1]


def _head_token(form: str) -> str:
    body = form.strip()
    if body.startswith("("):
        body = body[1:]
    body = body.split(";", 1)[0]
    return body.split()[0] if body.split() else ""


def _inner_forms(block: str) -> list[str]:
    if not block or not block.lstrip().startswith("("):
        return []
    start = block.find("(")
    end = matching_paren(block, start)
    if end < 0:
        return []
    inner = block[start + 1 : end]
    i = _skip_ws_and_comments(inner, 0)
    while i < len(inner) and inner[i] not in " \t\n\r();":
        i += 1
    forms: list[str] = []
    i = _skip_ws_and_comments(inner, i)
    while i < len(inner):
        i = _skip_ws_and_comments(inner, i)
        if i >= len(inner):
            break
        if inner[i] != "(":
            i += 1
            continue
        close = matching_paren(inner, i)
        if close < 0:
            break
        forms.append(inner[i : close + 1])
        i = close + 1
    return forms


def parse_fluents(block: str) -> list[list[str]]:
    """Parse top-level fluents inside ``(:init …)`` or ``(:goal …)``."""
    if not block:
        return []
    forms = _inner_forms(block)
    if len(forms) == 1 and _head_token(forms[0]) == "and":
        forms = _inner_forms(forms[0])
    facts: list[list[str]] = []
    for form in forms:
        if _head_token(form) == "and":
            for inner in _inner_forms(form):
                tokens = _tokenize_atom(inner)
                if tokens:
                    facts.append(tokens)
            continue
        tokens = _tokenize_atom(form)
        if tokens:
            facts.append(tokens)
    return facts


def _tokenize_atom(form: str) -> list[str]:
    body = form.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    body = body.split(";", 1)[0].strip()
    return body.split()


def split_problem(problem: str) -> dict[str, Any]:
    """Return init/goal blocks plus parsed fluent lists."""
    init_block = extract_section(problem, ":init")
    goal_block = extract_section(problem, ":goal")
    return {
        "pddl_init": init_block or None,
        "pddl_goal": goal_block or None,
        "init_facts": parse_fluents(init_block),
        "goal_facts": parse_fluents(goal_block),
    }


def write_fd_problem_artifacts(
    iter_dir: Path,
    problem: str,
    *,
    fd_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Write ``problem.pddl`` plus ``init.pddl`` / ``goal.pddl``.

    When ``fd_result`` is given (success or fail), also write ``fd_plan.json``.
    """
    iter_dir.mkdir(parents=True, exist_ok=True)
    sections = split_problem(problem)
    (iter_dir / "problem.pddl").write_text(problem, encoding="utf-8")
    if sections["pddl_init"]:
        (iter_dir / "init.pddl").write_text(
            sections["pddl_init"] + "\n", encoding="utf-8"
        )
    if sections["pddl_goal"]:
        (iter_dir / "goal.pddl").write_text(
            sections["pddl_goal"] + "\n", encoding="utf-8"
        )
    if fd_result is not None:
        write_fd_result(iter_dir, fd_result)
    return {
        "pddl_problem": problem,
        **sections,
    }


def write_fd_result(iter_dir: Path, fd_result: dict[str, Any]) -> Path:
    """Persist Fast Downward JSON (plan or error) next to the problem."""
    iter_dir.mkdir(parents=True, exist_ok=True)
    path = iter_dir / "fd_plan.json"
    path.write_text(
        json.dumps(fd_result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
