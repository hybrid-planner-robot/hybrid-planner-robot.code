# Hybrid LLM–PDDL Manipulation Planner

A neurosymbolic planner for tabletop robot manipulation. A text LLM (and, when
needed, a vision model) interprets a natural-language command and a scene,
builds a PDDL domain/problem, and Fast Downward returns a grounded action plan.

The **V4 evaluation** is a mock-init planning bake-off across three settings
(tabletop, kitchen, workshop) and four planner arms. Scene fixtures used in
that eval were frozen from the **GroundingDINO perception stack** documented
below — that stack is part of this repository, not an optional extra.

This release also includes a later prompt change: when listing objects in an
image, the VLM is asked to type each entity from the **task** (graspable item
vs place-on location vs put-into container).

License: MIT (see `LICENSE`).

---

## What is in this repository

| Path | Role |
|------|------|
| `planner/` | Hybrid problem generator, domain enrichment (R0 / R1), LLM baselines, Fast Downward glue, live DINO scene acquisition |
| `prompts/` | All LLM / VLM system prompts; modules load them with `prompts.load_prompt` |
| `pddl/domains/` | Four PDDL templates: base, stacking, containers, navigation |
| `vlm/` | GroundingDINO perception, image inventory (task-conditioned typing), optional VLM client |
| `scripts/` | Host entry points: V4 campaign, hybrid plan loop, mock-scene dump, image hybrid |
| `tests/fixtures/` | V4 suites, frozen `oracle_mock_v1` scenes, model ladders |
| `docker/` + `ros2_ws/` + `simulation/` | Gazebo / ROS 2 stack used to **capture** overview images and regenerate mock scenes |
| `docs/EXPERIMENTS.md` | How to set up, regenerate scene fixtures, and re-run V4 |

The original closed-loop **VLM-as-controller** path (the vision model emits
pick/place JSON that is injected into the arm orchestrator) is still present
in `planner/pipeline.py` / `vlm/planner.py` for compatibility, but it is
**not** how V4 plans are scored. V4 scores Fast Downward (or an LLM action
list) on frozen scenes.

---

## Requirements

| Component | Notes |
|-----------|--------|
| OS | Ubuntu 20.04+ (host). The sim container is Ubuntu 22.04 + ROS 2 Humble |
| Python | 3.10+ |
| GPU | Needed for local text LLMs (7B/14B) and for GroundingDINO / Qwen-VL |
| Fast Downward | `fast-downward` on `PATH`, or the Docker image (see below) |
| Docker | Required to **regenerate** mock scenes (Gazebo + overview camera). Not required to **re-run V4** if you use the shipped fixtures and a host Fast Downward |

---

## Setup

### 1. Clone and Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Planning-only unit tests (no GPU, no ROS):

```bash
pip install -r requirements-planning.txt
python -m pytest tests/test_planning_eval_v4.py tests/test_planning_campaign.py -q
```

### 2. Fast Downward

Either install [Fast Downward](https://www.fast-downward.org/) so the
`fast-downward` binary is on `PATH`, or build the Docker image (it compiles FD
during the image build):

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
```

V4 mock-init prefers the host binary and falls back to `docker exec` into the
container named `vlm_ros2`.

### 3. Optional remote text LLM

Copy `.env.example` to `.env` and set:

```bash
VLMRP_TEXT_LLM_API_KEY=...
VLMRP_TEXT_LLM_BASE_URL=http://127.0.0.1:8000/v1
```

Remote ladder profiles (`lab-qwen36`, `lab-glm53`) talk to any OpenAI-compatible
`/v1/chat/completions` endpoint. Local profiles load HuggingFace weights with
`transformers` and ignore the base URL.

---

## V4 planner arms

Each suite case is scored four times:

| Arm | What it does |
|-----|----------------|
| `llm_plan` | Text LLM emits a grounded action list. No Fast Downward. |
| `llm_pddl` | Text LLM authors a PDDL domain + problem; Fast Downward searches. |
| `enrich` (R0) | Hybrid stack: LLM selects a template, optionally authors one catalog skill, Fast Downward searches. |
| `r1` | Same hybrid stack with the R1 affordance enricher (second LLM assignment onto objects). |

Frozen scenes live in `tests/fixtures/llm_plan/{tabletop,kitchen,workshop}.json`.
They are `oracle_mock_v1` dumps produced by `scripts/dump_mock_scene.py` from a
GroundingDINO overview sweep — **not** Gazebo ground-truth poses.

How to reproduce the campaign and regenerate those fixtures is in
**[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**. The published V4 traces
(`data/planning_eval/golden_v4/plans.csv`) contain only the command, family,
and extracted plan — not automatic pass/fail labels.

---

## Task-conditioned item vs location typing

PDDL `on` is typed `item × location`. A notebook or plate is often listed as a
graspable object, so a goal such as `(on pen notebook)` will not ground.

After V4, the default image-inventory prompt
(`prompts/inventory/task_typed.txt`) asks the VLM to assign each visible
name using the **robot task**:

- **objects** — items to grasp or move
- **locations** — surfaces you put things ON (`on the plate/notebook`)
- **containers** — vessels you put things INTO (`in/into the bowl/box`)

A deterministic fallback (`vlm.inventory.apply_task_buckets`) and the problem
assembler (`align_scene_to_goal_facts`) re-bucket a destination if the VLM
still listed it as an object.

```bash
python scripts/run_image_hybrid.py \
  --image photo.png \
  --task "put the pen on the notebook"
```

The older keyword-style prompt remains at `prompts/inventory/keyword.txt`
(`--inventory-prompt`).
