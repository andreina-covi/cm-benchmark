# cm-benchmark

A **spatial-cognition QA benchmark** for vision-language models. It tests whether
a model builds and uses an internal cognitive map — not whether it can answer from
language priors.

Items are **multi-image frame sequences** from 3D environments plus a question.
Classes 1–3 are multiple-choice. Class 4 (`route_knowledge`,
`survey_based_route_planning`) is a free navigation-action sequence. Every answer
is composed from simulator metadata, never invented by an LLM.

---

## Why

VLMs often look spatially competent when the answer is in the prompt, in a single
frame, or in language statistics. That is not the same as keeping a map of space
over time.

The capability axis is a **partially ordered pipeline** (later constructs subsume
earlier ones). Frame of reference (`egocentric` vs `allocentric`) is tagged on
every item.

| Class | Name | Constructs |
|------:|------|------------|
| 1 | Spatial encoding | egocentric · allocentric |
| 2 | Spatial memory | spatial working memory · invisible displacement |
| 3 | Operation on the cognitive map | spatial updating · perspective taking |
| 4 | Navigation / wayfinding | route knowledge · survey knowledge |

Construct definitions: [`configs/taxonomy.yaml`](configs/taxonomy.yaml).
Every number that changes which items are produced lives in
[`configs/generation.yaml`](configs/generation.yaml) (see
[`docs/parameters_report.md`](docs/parameters_report.md)).
How we decide an item is well-built: [`Evaluation_process_manual.md`](Evaluation_process_manual.md).
How to run generation, visibility calibration, and scoring:
[`docs/generation.md`](docs/generation.md).
The pipeline, one paragraph per module: [`docs/architecture.md`](docs/architecture.md).

---

## Architecture

Two pipelines share one item store, separated by an immutable **FREEZE**:

```
collection folder → Episode GT → Item Generation → FREEZE → Model Evaluation
```

Generation builds candidates from metadata. Evaluation consumes the frozen set
only — it never edits or regenerates items.

**Invariants:** generators never invent spatial facts; every item has a non-null
`answer_source`; scoring is deterministic code (no LLM judge of correctness);
frozen items require `vision_necessary == true`.

---

## Status

Working loop today: **exported SPOC / AI2-THOR folders → episode GT → generated
items**. Still ahead: ground-truth validator, vision-necessity, FREEZE, and
the model-evaluation pipeline. `allocentric_encoding` stays unsupported until
trusted object facing exists.

Upstream collector: [spoc-robot-navigation](https://github.com/andreina-covi/spoc-robot-navigation).
Field brief: [`prompts/ai2thor_collection_extension.md`](prompts/ai2thor_collection_extension.md).

---

## Layout

```
configs/          taxonomy, generation.yaml (quality knobs), per-construct excerpts
analysis/dt_tune/ shipped visibility_filter.joblib (Q&A FOV DecisionTree)
src/cm_benchmark/
  generator/      episode GT from exported CSVs
  generation/     question items from episode GT
  evaluation/     scoring protocol
  storage/        SQLite + JSON artifacts
scripts/          one-off helpers (slides, etc.)
docs/             how to run the current code
tests/
```

---

## Setup

Python 3.12. Install the libraries in [`requirements.txt`](requirements.txt):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Commands below need the package on `PYTHONPATH` (pytest already adds `src` via `pyproject.toml`):

```bash
export PYTHONPATH=src
```

---

## Visibility filter

Q&A should only query objects a person can identify in the frame. The repo
ships a trained DecisionTree at
[`analysis/dt_tune/visibility_filter.joblib`](analysis/dt_tune/visibility_filter.joblib).
It drops tiny or barely-shown FOV detections from questions; the object catalog
and displacement logs stay unfiltered.

Episode GT and `generate_items` load that path automatically when you run from
the repo root (or set `CM_VISIBILITY_FILTER_MODEL`). Pass
`--visibility_model_path` to override. One path applies to every scene under
the input folder. If no `.joblib` is found, static floors apply
(`min_bbox_area=100`, `min_side=8`, `min_visible_pixels=40`).

To label scenes and refit the tree, see [`docs/generation.md`](docs/generation.md).

---

## Common commands

Episode GT (JSON for question generation; SQLite only with `--export_db`):

```bash
python -m cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /path/to/generated/navigation \
  --export_json \
  --output_path     src/cm_benchmark/storage/ai2thor/nav_data \
  --visibility_model_path analysis/dt_tune/visibility_filter.joblib
```

`--csv_path_folder` may be one episode (`<timestamp>/` or its `annotations/`
subfolder) or a root whose children are those episode folders. `scene_id` is
read from filenames such as `navigation-house_007514.csv`. One
`--visibility_model_path` applies to every episode under that root. Omit the
flag to use the shipped `analysis/dt_tune/visibility_filter.joblib` when it
exists.

`--output_path` is a root. Each scene is written to its own subfolder.
Add `--export_db` to also write SQLite under `--db_path` (default
`src/cm_benchmark/storage/ai2thor/episodes`):

```text
nav_data/house_007514/nav_house_007514.json
episodes/house_007514/episodes.db   # only with --export_db
```

Generate items from that nav JSON root. One `--visibility_model_path` applies
to every scene. `--output_path` is a root, one subfolder per scene:

```bash
python -m cm_benchmark.generation.generate_items \
  --episode_json src/cm_benchmark/storage/ai2thor/nav_data \
  --output_path  src/cm_benchmark/storage/ai2thor/items \
  --visibility_model_path analysis/dt_tune/visibility_filter.joblib \
  --max_per_construct 2
```

```text
items/house_007514/items_house_007514.json
```

Quality knobs (visibility floors, direction wedges, delays, class-4 pair
gates, the class-4 frame budget, …) are read from
[`configs/generation.yaml`](configs/generation.yaml).
Pass `--config path/to/generation.yaml` to use another file (an ablation is
a copy of that file with different values). These CLI flags override the
loaded file when present: `--max_per_construct`, `--n_frames`,
`--swm_min_delay`, `--swm_max_delay`, `--su_min_delay`, `--su_max_delay`,
`--visibility_model_path`. The same `--config` flag is on
`ai2thor_nav_generator`. Each item stores `generation_config` (the file's
`name`) and `generation_config_hash` (SHA-256 of the values that produced it).

`--n_frames` is how many frames a route or survey item shows. Omit it to
keep `class4.n_frames` (16). The frames are an even sample of the walk and
always include the source and goal sightings; a survey also keeps one frame
per planned-path edge. The item is dropped when those required frames do not
fit. Pass a smaller or larger value when a model or machine needs a different
budget, then regenerate the items.

Spatial updating and spatial working memory show every field-of-view frame
from the sighting through the query, in time order, including a turn in
place. Perspective taking uses the one frame in which all three named
objects are visible. An invisible-displacement destination must be clearly
visible in a frame before the query.

If a construct cannot be proven from that scene's GT, the item is
`unsupported` — that is expected, not a bug to tune away.

```bash
pytest tests/ -q
```

More commands (visibility labeling, human review package, slides, CLI flags):
[`docs/generation.md`](docs/generation.md).

---

## Roadmap

- [x] Episode GT from AI2-THOR / SPOC exports
- [x] Generated items (templates, concise / verbose)
- [x] Q&A visibility filter (optional DecisionTree)
- [ ] Object facing → allocentric encoding
- [ ] LLM paraphrase, GT validator, vision-necessity
- [ ] FREEZE + model evaluation
- [ ] Matterport3D

---

## License

See [`LICENSE`](LICENSE).
