# Prompts

All system / few-shot prompts used by the planner live here. Python modules
must load them with ``prompts.load_prompt(...)`` (or ``prompts.prompt_path``)
rather than inlining the text.

| Folder | Used by |
|--------|---------|
| `inventory/` | `vlm/inventory.py` — name objects in a photo and type them from the task |
| `vlm/` | `vlm/planner.py` — optional vision action-planner |
| `goal/` | local text LLM `:goal` generator (`LocalLLMGoalGenerator`); `cloud.md` is the cloud backend |
| `llm_plan/` | `llm_plan` baseline stages |
| `llm_pddl/` | `llm_pddl` baseline stages |
| `domain/` | R0 domain select / enrich / bind (`planner/domain_llm.py`) |
| `r1/` | R1 enricher (`planner/r1/`) |

The default inventory prompt is `inventory/task_typed.txt` (item vs location
vs container from the robot command). `inventory/keyword.txt` is the older
keyword-bucket variant.
