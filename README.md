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

Working loop today: **exported SPOC / AI2-THOR folders → episode GT → draft
questions**. Still ahead: ground-truth validator, vision-necessity, FREEZE, and
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
  generation/     first-draft questions
  evaluation/     scoring protocol
  storage/        SQLite + JSON artifacts
scripts/          one-off helpers (slides, etc.)
docs/             how to run the current code
tests/
```

---

## Setup

Python 3.12.

```bash
pip install pandas numpy seaborn scikit-learn matplotlib pytest pyyaml
pip install -e .    # optional; tests add src via pythonpath
```

---

## Common commands

Episode GT (SQLite is the system of record; JSON is optional):

```bash
python -m cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /path/to/collection_run \
  --db_path         src/cm_benchmark/storage/ai2thor/episodes.db \
  --export_json \
  --output_path     src/cm_benchmark/storage/ai2thor/nav_data \
  --output_filename nav_data_house_XXXXXX.json
```

`--csv_path_folder` may be the episode root or its `annotations/` subfolder.

Draft questions:

```bash
python -m cm_benchmark.generation.draft_items \
  --episode_json src/cm_benchmark/storage/ai2thor/nav_data/nav_data_house_XXXXXX.json \
  --output       src/cm_benchmark/storage/ai2thor/items/draft_house_XXXXXX.json \
  --max_per_construct 2
```

If a construct cannot be proven from that scene's GT, the draft is
`unsupported` — that is expected, not a bug to tune away.

```bash
pytest tests/ -q
```

More commands (visibility labeling, slides, frame annotation, CLI flags):
[`docs/generation.md`](docs/generation.md).

---

## Roadmap

- [x] Episode GT from AI2-THOR / SPOC exports
- [x] First-draft items (templates, concise / verbose)
- [x] Q&A visibility filter (optional DecisionTree)
- [ ] Object facing → allocentric encoding
- [ ] LLM paraphrase, GT validator, vision-necessity
- [ ] FREEZE + model evaluation
- [ ] Matterport3D

---

## License

See [`LICENSE`](LICENSE).
