# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A spatial-cognition QA benchmark for vision-language models. Each item is a multi-image frame sequence from SPOC / AI2-THOR exports plus a question. Classes 1–3 are multiple choice. Class 4 (`route_knowledge`, `survey_based_route_planning`) asks for a free navigation-action sequence. There are 8 constructs in 4 classes, defined in `configs/taxonomy.yaml`.

## Invariants (never break)

1. Answers come only from exported simulator metadata, and every item has a non-null `answer_source`.
2. Generation, validation and scoring are deterministic code. No LLM judges answers. `paraphrase.py` may reword a question but never its answer, and it is off by default.
3. FREEZE is one-way: evaluation never edits or regenerates items. Frozen items require `vision_necessary == true`.
4. Questions never state the answer, a direction or a move. Verbose preambles must not leak the answer.
5. Confusable objects (the `obj-synset` rule) are disambiguated by proximity to a unique landmark or by a colour-dot mark, never by a direction.
6. Every quality-affecting number lives in `configs/generation.yaml`. Never hardcode one. Partial configs (e.g. `configs/ablations/*.yaml`, `configs/generation_max.yaml`) are merged onto it by `load_generation_config`, and are passed with `--config`.

## How to work (from `.cursor/rules/benchmark.mdc`)

- Make the smallest change that does the task, and don't refactor unrelated code.
- Read only the files the task touches. Run only the relevant tests, and rerun pilots or baselines only when asked.
- Keep reports short: a table or a few lines.
- If a request conflicts with these rules or the docs, ask before acting.

## Commands

The environment is Python 3.12+ (`.venv`). Install with `pip install -r requirements.txt`. pytest adds `src/` to the path through `pyproject.toml`. For `python -m` runs outside pytest, set `PYTHONPATH=src`.

```bash
pytest                                              # all tests
pytest tests/test_nav_graph.py                      # one file
pytest tests/test_nav_graph.py::test_name -q        # one test

# Stage 1: collector folder -> episode GT (JSON; add --export_db --db_path for SQLite)
python -m cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /path/to/collection --export_json \
  --output_path src/cm_benchmark/storage/ai2thor/nav_data \
  --visibility_model_path analysis/dt_tune/visibility_filter.joblib

# Stage 2: episode GT -> items/<scene>/items_<scene>.json
python -m cm_benchmark.generation.generate_items \
  --episode_json src/cm_benchmark/storage/ai2thor/nav_data \
  --output_path src/cm_benchmark/storage/ai2thor/items --max_per_construct 2
  # optional: --constructs, --config, --swm_min_delay/--swm_max_delay, --su_*_delay, --frame_budgets

python scripts/report_items.py src/cm_benchmark/storage/ai2thor/items   # counts by scene/construct

# Regression: rebuild the 3 reference scenes and diff against analysis/baseline
scripts/check_baseline.sh            # uses COLLECTION, MODEL, PYTHON env vars
scripts/run_baseline.sh OUT_DIR      # rebuild only
```

Human-review tooling: `scripts/build_review_package.py`, `scripts/analyze_review.py` and `scripts/plot_review_report.py`. The full CLI reference is in `docs/generation.md`.

## Related repository: spoc-robot-training

`spoc-robot-training` (a sibling repo, usually `../spoc-robot-training`) runs AI2-THOR and writes the collection folders that stage 1 reads. Its `collector.py` exports, per step and per visible object, `visible-pixels` (the instance-mask pixel count inside the 2D box), `bbox`, `occupancy`, `mark-x`/`mark-y` (colour-dot mark position), the expected 3D-projected box and angular size. Frames are 396×224 (`utils/constants/stretch_initialization_utils.py`). If a quality fix needs a field the export lacks, change it there, then re-collect. Don't approximate it here.

## Architecture

```
collection folder → episode GT → items → FREEZE → evaluation
```

- **`generator/`** (episode GT). `episode_paths.py` locates the CSV/JSON logs. `nav_sequence_generator.py` is the generic per-step builder: pose, `visible_objects`, `non_visible_objects` (seen before, not in view now) and agent→object edges. `ai2thor_nav_generator.py` is the AI2-THOR subclass. It applies the collection-time visibility filter, stores object poses and AABBs, and attaches displacement events, `placement_candidates`, world layout and the nav graph. `visibility_filters.py` holds both the static pixel floors and the DecisionTree bundle (only *distinguible* detections pass). The same filter is re-applied at item time. Nothing after this stage reads the CSVs.
- **`generation/`** (items). `generate_items.py` (CLI) → `pipeline.py` runs `plan_episode` → `build_items_from_facts` for each scene.
  - `planner.py` decides items. It has one `plan_<construct>` per construct, each applying gates and returning `PlannedFact`s (query and encode steps, locked answer, `answer_source`, frames, option pool, `distractor_seeds`). If nothing survives, it returns an `unsupported` fact with per-gate reject counts. `allocentric_encoding` always returns unsupported, because there is no trusted object facing.
  - `constructs.py` holds the label vocabulary, templates, bearing→label rules and visibility gates. `geometry.py` turns a pose and a world point into an ego label using yaw only, and refuses a label when the object straddles a sector edge. `disambiguation.py` handles naming, proximity and marks.
  - `templates.py` turns facts into items. It writes one item per style (concise and verbose) linked by `paired_item_id`, and shuffles options with a seed.
  - `nav_graph.py` / `route_replay.py` are the class-4 graph and the continuous-walk simulator. The same simulator scores model action sequences in `evaluation/score.py`.
  - `schema.py` defines `CandidateItem`. Verification fields (`blind_llm_correct`, `caption_llm_correct`, `vision_necessary`) start null.
- **`evaluation/`** contains `protocol.py` (the shared system prompt and item wrapping) and `score.py`. MCQ items are scored by exact option match.
- **`storage/episode_store.py`** is the SQLite `EpisodeStore`.

`docs/architecture.md` walks through each construct's planner gates step by step. Read it before changing a planner. `docs/parameters_report.md` justifies each config value, and `Evaluation_process_manual.md` defines what a well-built item is.

**Status:** the ground-truth validator, the vision-necessity check, FREEZE and model evaluation are not built yet.
