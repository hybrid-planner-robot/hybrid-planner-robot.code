# Setup and experiment reproduction (Golden V4)

This document is enough to (1) install the stack, (2) regenerate the frozen
scene fixtures from perception, and (3) re-run the V4 planning campaign.

V4 is **plan-only**. It does not move the arm. Each cell is one natural-language
command + one frozen scene + one planner arm → PDDL artefacts and a Fast
Downward plan (except `llm_plan`, which never calls the symbolic planner).

---

## 1. Software setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Install [Fast Downward](https://www.fast-downward.org/) so `fast-downward` is
on `PATH`. Alternatively build the Docker image (FD is compiled in the image):

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
```

Offline unit tests (no GPU, no ROS):

```bash
python -m pytest tests/test_planning_eval_v4.py tests/test_planning_campaign.py \
  tests/test_run_loop_host_mock.py tests/test_image_hybrid.py -q
```

---

## 2. Scene fixtures (perception → mock)

V4 does **not** call Gazebo at eval time. It loads frozen JSON:

| Setting | Fixture |
|---------|---------|
| tabletop | `tests/fixtures/llm_plan/tabletop.json` |
| kitchen | `tests/fixtures/llm_plan/kitchen.json` |
| workshop | `tests/fixtures/llm_plan/workshop.json` |

Each file is `oracle_mock_v1`: object names, the support they rest on, and
place locations. The workshop fixture notes that it was frozen from a
GroundingDINO overview (`min_score=0.4`), not from Gazebo ground truth.

### How those files were produced

`scripts/dump_mock_scene.py` runs the same live DINO spine used by the hybrid
loop (`planner.live_scene.acquire_live_scene`):

1. Start Gazebo with the matching world (`tabletop` / `kitchen` / `workshop`).
2. Capture the overview camera image (inside the ROS 2 container).
3. Run GroundingDINO (`vlm/perception.py` + `planner/dino_localisation.py`).
4. Convert detections to `oracle_mock_v1` JSON
   (`planner.r1.scenes.detections_to_oracle_mock`).

```bash
# Sim must be up with the target world (see README / bin/start_sim.sh).
python scripts/dump_mock_scene.py --world tabletop
python scripts/dump_mock_scene.py --world kitchen
python scripts/dump_mock_scene.py --world workshop --min-score 0.40
```

Default output path: `tests/fixtures/llm_plan/<world>.json`. Re-running this
**overwrites** the fixture; keep a copy if you need to compare against V4.

Oracle (Gazebo GT) is a separate `--scene-source oracle` path in the live loop.
Golden V4 reports `scene_source: oracle` in the suite JSON because the mock
adapter presents the frozen file as an oracle-shaped `SceneState`. The pixels
that created the dump were DINO, as recorded in the fixture `note` field.

---

## 3. V4 campaign

### Suites

| Suite | Cases | Families |
|-------|------:|----------|
| `suite_v4_tabletop_mock.json` | 20 | template-complete, ungenerable, long-sequence, underspecified |
| `suite_v4_kitchen_mock.json` | 30 | same + `needs_enrichment` (pour / stir / cut / tilt) |
| `suite_v4_workshop_mock.json` | 30 | same + `needs_enrichment` (cut / drill / paint / clamp) |

80 cases × 4 arms = **320** scored cells per model.

Phrasing is split explicit / implicit inside each suite. Kitchen and workshop
are symmetric on family sizes; only the enrichment skill catalog differs.

### Arms

Passed as `--arms llm_plan,llm_pddl,enrich,r1`:

- `llm_plan` — `scripts/run_loop_llm_plan.py`
- `llm_pddl` — `scripts/run_loop_llm_pddl.py`
- `enrich` — `scripts/run_loop_host.py --hybrid mvp --control fd --online-enrichment 1`
- `r1` — same host path with `--enrichment-profile r1`

Every arm is invoked with `--mock-scene --scene-file <fixture>` and
`--plan-only`.

### Models (ladder)

`tests/fixtures/planning_eval/model_ladder_v4.json`:

| Profile | Kind | Model id |
|---------|------|----------|
| `local-qwen7b` | local HuggingFace | `Qwen/Qwen2.5-7B-Instruct` |
| `local-qwen14b` | local HuggingFace | `Qwen/Qwen2.5-14B-Instruct` |
| `lab-qwen36` | remote OpenAI-compatible | served as `lab-qwen36` (Qwen3.6-35B-A3B-FP8) |
| `lab-glm53` | remote OpenAI-compatible | served as `lab-glm53` (GLM-5.3-Flash) |

Point remote profiles at your own vLLM / SGLang server:

```bash
cp .env.example .env
# VLMRP_TEXT_LLM_API_KEY=...
# VLMRP_TEXT_LLM_BASE_URL=http://127.0.0.1:8000/v1
```

You can also set `defaults.api_base_url` in the ladder JSON. Keys never belong
in the ladder file.

### Commands

List profiles:

```bash
python scripts/eval_planning_campaign.py \
  --ladder tests/fixtures/planning_eval/model_ladder_v4.json --list
```

Probe a remote endpoint, then one smoke case per setting, then the full 320:

```bash
python scripts/eval_planning_campaign.py \
  --ladder tests/fixtures/planning_eval/model_ladder_v4.json \
  --profile lab-qwen36 --probe

python scripts/eval_planning_campaign.py \
  --ladder tests/fixtures/planning_eval/model_ladder_v4.json \
  --profile lab-qwen36 --smoke

python scripts/eval_planning_campaign.py \
  --ladder tests/fixtures/planning_eval/model_ladder_v4.json \
  --profile local-qwen7b
```

`--dry-print` prints the environment and child battery commands without running
them. `--skip-docker` skips `docker start` when Fast Downward is already on the
host PATH.

Equivalent wrapper: `bin/eval_planning_campaign.sh` (activates `.venv` if
present).

### GPU-free CI check

Canned LLM + canned Fast Downward on the V4 tabletop suite:

```bash
python scripts/eval_planning_battery.py \
  --suite tests/fixtures/planning_eval/suite_v4_tabletop_mock.json \
  --arms llm_plan,llm_pddl,enrich,r1 \
  --mock-init --mock-llm \
  --out-dir data/planning_eval/ci_smoke
```

### Outputs

Campaign root: `data/planning_eval/v4_<profile>_run_<UTC>/`

```
all_worlds.json / all_worlds.md     # operational cross-world count, not a pooled paper stat
<tabletop|kitchen|workshop>/
  report.json / report.md
  cases/<arm>_<case_id>/            # domain.pddl, problem.pddl, fd_plan.json, debug.json
  runs/<arm>_<case_id>/summary.json
```

The shipped V4 traces are `data/planning_eval/golden_v4/plans.csv`: command,
family, and extracted plan only (1280 rows). Paper scores were assigned by
hand from that table. Re-running the campaign still writes `report.json` with
automatic flags (`plan_correct`, `false_plan`, …); those flags were **not**
used in the final evaluation.

---

## 4. Task-conditioned typing (post-V4, shipped)

V4 mock scenes already split objects vs locations in the JSON. The later change
applies when names come from an **image**:

- Prompt: `prompts/inventory/task_typed.txt` (default)
- Fallback: `vlm.inventory.apply_task_buckets`
- Assembler: `planner.problem_generator.assembler.align_scene_to_goal_facts`

Example: “put the pen on the notebook” must type `notebook` as a **location**,
otherwise `(on pen notebook)` is ill-typed.

```bash
python scripts/run_image_hybrid.py \
  --image path/to/rgb.png \
  --task "put the pen on the notebook"
```

---

## 5. Notes for a comparable re-run

- Temperature is 0 in the V4 ladder. Keep it there for comparability.
- `all_worlds.json` is a convenience sum across settings, not a paper metric.
- An empty `plan` cell means that run produced no action sequence (refusal,
  invalid PDDL, or unsolvable search). Do not treat campaign `success` /
  `plan_correct` fields as the published labels.
