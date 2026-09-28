# Golden V4 traces

One row per (model × arm × setting × command). Paper scores are assigned **by
hand** — fill `scoring.csv`, not the automatic campaign flags.

| File | Role |
|------|------|
| `plans.csv` | Compact traces (command + extracted plan), all 4 models |
| `scoring.csv` | Hand-scoring workbook: GLM-5.3 dropped; PDDL / R0 / R1 fields added; empty `judgement` and `flag` |

## `scoring.csv`

960 rows = 3 models (Qwen2.5-7B, Qwen2.5-14B, Qwen3.6) × 80 commands × 4 arms.
Open in LibreOffice / Excel (UTF-8 with BOM). Fill only:

| Column | Allowed values |
|--------|----------------|
| `judgement` | `no plan` · `uncorrect plan` · `correct plan` |
| `flag` | `fail` · `correct` |
| `notes` | free text (optional) |

Leave every other column as generated.

| Column | When it is filled |
|--------|-------------------|
| `plan` | Extracted action list from `plans.csv` (empty = no extracted plan) |
| `pddl_domain`, `pddl_goal` | **llm_pddl** only |
| `domain_template`, `completeness`, `enrichment_requested`, `needed_skills`, `proposed_action`, `action_authoring` | **R0** and **R1** |
| `properties_created`, `init_assignments` | **R1** (also R0 when the enricher declared new predicates) |

`arm` uses `R0` for the enrich arm and `R1` for the affordance enricher.
`enrichment_requested` is `yes` when select marked the template `incomplete`
or named a needed skill.

Reproduce a cell with the V4 campaign in `docs/EXPERIMENTS.md`.
