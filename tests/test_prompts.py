"""Central ``prompts/`` library is complete and loadable."""

from __future__ import annotations

from prompts import DIR, load_prompt, prompt_path


_REQUIRED = (
    ("inventory", "task_typed.txt"),
    ("inventory", "keyword.txt"),
    ("vlm", "system.txt"),
    ("vlm", "system_loop.txt"),
    ("vlm", "system_replanning.txt"),
    ("domain", "select.md"),
    ("domain", "enrich.md"),
    ("domain", "bind.md"),
    ("r1", "enrich.md"),
    ("r1", "assign.md"),
    ("goal", "cloud.md"),
    ("goal", "manipulation_base.md"),
    ("llm_plan", "refuse.md"),
    ("llm_plan", "intent.md"),
    ("llm_plan", "ground.md"),
    ("llm_plan", "system.md"),
    ("llm_pddl", "refuse.md"),
    ("llm_pddl", "schema.md"),
    ("llm_pddl", "actions.md"),
    ("llm_pddl", "init.md"),
    ("llm_pddl", "goal.md"),
    ("llm_pddl", "toy_domain.pddl"),
    ("llm_pddl", "toy_problem.pddl"),
)


def test_prompt_library_lives_at_repo_root():
    assert DIR.is_dir()
    assert DIR.name == "prompts"


def test_required_prompt_files_are_nonempty():
    for parts in _REQUIRED:
        path = prompt_path(*parts)
        assert path.is_file(), path
        assert load_prompt(*parts).strip(), path
