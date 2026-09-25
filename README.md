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
How we decide an item is well-built: [`Evaluation_process_manual.md`](Evaluation_process_manual.md).
How to run generation, visibility calibration, and scoring:
[`docs/generation.md`](docs/generation.md).

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
configs/          taxonomy + per-construct excerpts
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

## Common commands

Episode GT (SQLite is the system of record; JSON is optional):

```bash
python -m cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /path/to/generated/navigation \
  --db_path         src/cm_benchmark/storage/ai2thor/episodes \
  --export_json \
  --output_path     src/cm_benchmark/storage/ai2thor/nav_data
```

`--csv_path_folder` may be one episode (`<timestamp>/` or its `annotations/`
subfolder) or a root whose children are those episode folders. `scene_id` is
read from filenames such as `navigation-house_007514.csv`. One
`--visibility_model_path` applies to every episode under that root.

`--db_path` and `--output_path` are roots. Each scene is written to its own
subfolder:

```text
episodes/house_007514/episodes.db
nav_data/house_007514/nav_house_007514.json
```

Generate items from that nav JSON root. One `--visibility_model_path` applies
to every scene. `--output_path` is a root, one subfolder per scene:

```bash
python -m cm_benchmark.generation.generate_items \
  --episode_json src/cm_benchmark/storage/ai2thor/nav_data \
  --output_path  src/cm_benchmark/storage/ai2thor/items \
  --max_per_construct 2
```

```text
items/house_007514/items_house_007514.json
```

If a construct cannot be proven from that scene's GT, the item is
`unsupported` — that is expected, not a bug to tune away.

```bash
pytest tests/ -q
```

More commands (visibility labeling, slides, frame annotation, CLI flags):
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
