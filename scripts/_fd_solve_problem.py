#!/usr/bin/env python3
"""
_fd_solve_problem.py — Runs INSIDE the Docker container.

Solve a PDDL problem with Fast Downward and print a JSON plan to stdout.
Used by run_loop_host --control fd to preview the plan before execution.

Stdin JSON:
  {
    "domain_template": "manipulation_base",
    "problem": "(define (problem ...) ...)",
    "domains_dir": "/workspace/pddl/domains",  # optional
    "domain": "(define (domain ...) ...)"      # optional: online-enriched
                                               # domain text (Session 30);
                                               # overrides the template file
  }

Stdout JSON:
  {
    "success": true,
    "actions": ["(pick red_cup table)", "(place red_cup shelf_b)"],
    "primitives": [{"name": "pick", "args": ["red_cup", "table"]}, ...]
  }

  ``actions`` may be empty: the goal already holds in ``:init``. That is
  success. ``success: false`` with ``unsolvable — no plan`` means Fast
  Downward wrote no plan file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"success": False, "error": "empty stdin"}))
        sys.exit(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"success": False, "error": f"invalid JSON: {exc}"}))
        sys.exit(1)

    domain_template = data.get("domain_template") or "manipulation_base"
    problem = data.get("problem") or ""
    domains_dir = Path(data.get("domains_dir") or "/workspace/pddl/domains")
    domain_path = domains_dir / f"{domain_template}.pddl"
    domain_override = (data.get("domain") or "").strip()

    if not problem.strip():
        print(json.dumps({"success": False, "error": "missing problem"}))
        sys.exit(1)
    if not domain_override and not domain_path.exists():
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"domain file not found: {domain_path}",
                }
            )
        )
        sys.exit(1)

    # Prefer workspace planner (bind-mounted) over install.
    sys.path.insert(0, "/workspace")
    try:
        from planner.fast_downward import (
            FastDownwardPlanner,
            FastDownwardTimeout,
            result_from_actions,
            result_from_timeout,
        )
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"import failed: {exc}"}))
        sys.exit(1)

    domain_text = domain_override or domain_path.read_text()
    try:
        actions = FastDownwardPlanner().solve_from_strings(domain_text, problem)
    except FastDownwardTimeout as exc:
        print(json.dumps(result_from_timeout(exc.limit_s)))
        sys.exit(2)
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"FD error: {exc}"}))
        sys.exit(1)

    payload = result_from_actions(actions)
    print(json.dumps(payload))
    if not payload.get("success"):
        sys.exit(2)


if __name__ == "__main__":
    main()
