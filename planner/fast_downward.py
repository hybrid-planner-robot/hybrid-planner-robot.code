"""
Interface to the Fast Downward PDDL planner.
Assumes fast-downward is installed and available as `fast-downward` on PATH.
"""

from __future__ import annotations
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from planner.call_timings import fd_span

# Tabletop problems that have a plan finish in well under a second.
# Unsolvable A* (blind) can otherwise run for minutes (R1 bad :init).
DEFAULT_SEARCH_TIME_LIMIT_S = 30
ENV_SEARCH_TIME_LIMIT_S = "VLMRP_FD_SEARCH_TIME_LIMIT_S"
# FD: plan found / search ended without a plan / search hit the time limit.
_FD_OK_RETURNCODES = frozenset({0, 1, 11, 12, 23})


def resolve_search_time_limit_s(value: int | None = None) -> int:
    if value is not None:
        return max(1, int(value))
    raw = os.environ.get(ENV_SEARCH_TIME_LIMIT_S, "").strip()
    if raw:
        return max(1, int(raw))
    return DEFAULT_SEARCH_TIME_LIMIT_S


class FastDownwardTimeout(RuntimeError):
    """Search hit ``--search-time-limit`` or the subprocess watchdog."""

    def __init__(self, message: str, *, limit_s: int) -> None:
        super().__init__(message)
        self.limit_s = int(limit_s)


class FastDownwardPlanner:
    """
    Runs Fast Downward given a domain and problem PDDL file.
    Returns the plan as a list of action strings.
    """

    SEARCH_CONFIG = "astar(blind())"   # swap for lama-first in production

    def __init__(self, *, search_time_limit_s: int | None = None) -> None:
        self.search_time_limit_s = resolve_search_time_limit_s(search_time_limit_s)

    def solve(self, domain_path: str, problem_path: str) -> list[str] | None:
        """
        Args:
            domain_path:  Path to domain.pddl
            problem_path: Path to problem.pddl

        Returns:
            List of action strings (e.g. ["(pick red_cup table_a)", ...])
            or None if no plan file was written (unsolvable).

            An empty list means Fast Downward wrote a plan file with no
            actions: the goal already holds in ``:init``. That is a valid
            solution, not a failure.

        Raises:
            FastDownwardTimeout: search or subprocess exceeded the time limit.
        """
        with fd_span():
            return self._solve_impl(domain_path, problem_path)

    def _solve_impl(self, domain_path: str, problem_path: str) -> list[str] | None:
        limit = self.search_time_limit_s
        watchdog_s = limit + 15
        with tempfile.TemporaryDirectory() as tmpdir:
            sas_file = os.path.join(tmpdir, "output.sas")
            plan_file = os.path.join(tmpdir, "plan")

            cmd = [
                "fast-downward",
                "--sas-file", sas_file,
                "--plan-file", plan_file,
                "--search-time-limit", f"{limit}s",
                domain_path,
                problem_path,
                "--search", self.SEARCH_CONFIG,
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=watchdog_s,
                )
            except subprocess.TimeoutExpired as exc:
                raise FastDownwardTimeout(
                    f"search time limit ({limit}s) — subprocess watchdog",
                    limit_s=limit,
                ) from exc

            if result.returncode == 23:
                raise FastDownwardTimeout(
                    f"search time limit ({limit}s) — Fast Downward rc=23",
                    limit_s=limit,
                )
            if result.returncode not in _FD_OK_RETURNCODES:
                # FD often prints parse/type errors to stdout, not stderr
                detail = result.stderr or result.stdout[:2000]
                raise RuntimeError(
                    f"Fast Downward error (rc={result.returncode}):\n{detail}"
                )

            plan_path = Path(plan_file)
            if not plan_path.exists():
                return None  # unsolvable

            lines = plan_path.read_text().splitlines()
            return [l.strip() for l in lines if l.strip() and not l.startswith(";")]

    def solve_from_strings(self, domain_text: str, problem_text: str) -> list[str] | None:
        """
        Convenience wrapper: accepts domain and problem as strings instead of
        file paths. Writes temporary files and calls solve().
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pddl", delete=False
        ) as df, tempfile.NamedTemporaryFile(
            mode="w", suffix=".pddl", delete=False
        ) as pf:
            df.write(domain_text)
            pf.write(problem_text)
            df_name, pf_name = df.name, pf.name

        try:
            return self.solve(df_name, pf_name)
        finally:
            os.unlink(df_name)
            os.unlink(pf_name)


def result_from_actions(actions: list[str] | None) -> dict[str, Any]:
    """
    JSON-shaped FD outcome for the host loop.

    ``None`` → unsolvable. ``[]`` → success with an empty plan (goal already
    holds). A non-empty list → the action strings Fast Downward returned.
    """
    if actions is None:
        return {
            "success": False,
            "error": "unsolvable — no plan",
            "actions": [],
            "primitives": [],
        }
    from planner.plan_parser import normalize_to_primitives, parse_plan

    prims = normalize_to_primitives(parse_plan(actions))
    return {
        "success": True,
        "actions": list(actions),
        "primitives": [
            {"name": p.name, "args": list(p.args)} for p in prims
        ],
    }


def result_from_timeout(limit_s: int) -> dict[str, Any]:
    return {
        "success": False,
        "error": f"search time limit ({limit_s}s)",
        "actions": [],
        "primitives": [],
    }
