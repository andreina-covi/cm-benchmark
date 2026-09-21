# Running generation

Operational notes for episode GT, draft items, visibility, and class-4
scoring. Construct *meaning* lives in [`configs/taxonomy.yaml`](../configs/taxonomy.yaml).
Numeric gates and algorithms live in `src/cm_benchmark/generation/planner.py`
and `nav_graph.py`.

---

## Episode GT

SPOC collection folder:

```text
<timestamp>/
  images/img_<t>.png
  annotations/
    navigation-*.csv, objects-*.csv, object_state-*.csv,
    displacement_events-*.csv, passage_state-*.csv, region_trajectory-*.csv
    episode_meta-*.json, world_layout-*.json, nav_graph-*.json
```

```bash
python -m cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /path/to/collection_run \
  --db_path         src/cm_benchmark/storage/ai2thor/episodes.db \
  --export_json \
  --output_path     src/cm_benchmark/storage/ai2thor/nav_data \
  --output_filename nav_data_house_XXXXXX.json
```

`--csv_path_folder` may be the episode root or `annotations/`.
`scene_id` / `episode_id` come from `episode_meta-*.json` when present.

SQLite (`EpisodeStore`) is the system of record. JSON is optional, for
inspection and drafting.

Sparse tracks (`object_state_track`, `region_trajectory`, `passage_state`)
store the first observation plus later change points. At step `t`, take the
latest entry with `step <= t` (`state_at_step` / `series_at_step`).

`distance_label` (how far) and `angle_relation` front/behind (local +Z) are
independent: `beyond` + `front` means ahead of the agent, but far.

---

## Visibility

Q&A drops tiny or barely-shown FOV detections. Defaults (override as needed):
`min_bbox_area=100`, `min_side=8`, `min_visible_pixels=40`. A trained
DecisionTree (`visibility_filter.joblib`) is preferred when labels exist.

Label a scene:

```bash
python -m cm_benchmark.utils.build_labeling_set \
  --nav_csv     /path/to/annotations/navigation-house_XXXXXX.csv \
  --images_dir  /path/to/episode/images \
  --scene_id    house_XXXXXX \
  --output_path src/cm_benchmark/storage/ai2thor/output/labeling
```

Fit across scenes (put `{scene}_calibration_manifest.csv` + `{scene}_labels.json`
in one folder):

```bash
python -m cm_benchmark.utils.fit_thresholds --folder path/to/calibration_scenes
python -m cm_benchmark.utils.fit_thresholds --folder path/to/calibration_scenes --tune
```

Then pass `visibility_model_path=...` into `Ai2ThorNavGenerator`.

---

## Draft items

```bash
python -m cm_benchmark.generation.draft_items \
  --episode_json src/cm_benchmark/storage/ai2thor/nav_data/nav_data_house_XXXXXX.json \
  --output       src/cm_benchmark/storage/ai2thor/items/draft_house_XXXXXX.json \
  --max_per_construct 2
```

Or `--db_path` + `--episode_id`. Optional `--constructs`, `--swm_min_delay`,
`--su_min_delay`.

- **concise** / **verbose** share the same answer. Verbose must not leak it.
- Temporal wording is `{k} steps ago` / `now` — not a bundled “time order” cue.
- `unsupported` is the correct outcome when a scene cannot prove a construct
  (e.g. a straight one-room walk for route knowledge).

Class 4 is not MCQ. The action vocabulary is stated once in
`cm_benchmark.evaluation.protocol.SYSTEM_INSTRUCTION`. Route scores a predicted
walk on `traversed_edges`; survey on `viewed_edges` (must stay viewed and use
at least one unwalked edge). Stored `answer` is a reference sequence for
analysis, not exclusive gold.

Pair selection (both class-4 constructs): geodesic in 1–30 m **and** at least
the scene’s own typical landmark-pair length; a detour ratio of at least this
scene’s typical geo/eucl (capped at 1.1 for route, 1.05 for survey); emit the
hardest surviving pairs. Route also requires real turns. Constants: `planner.py`.

---

## Review slides

```bash
python scripts/build_avance_presentation.py \
  --template /path/to/template.pptx \
  --draft-json src/cm_benchmark/storage/ai2thor/items/draft_house_XXXXXX.json \
  --output /path/to/examples.pptx
```

Class-4 stills get a letter marker on source and goal. Evaluation uses the
unmodified frames.

Annotate frames for spatial review:

```bash
python -m cm_benchmark.utils.annotate_frames \
  --episode_json src/cm_benchmark/storage/ai2thor/nav_data/nav_data_house_XXXXXX.json \
  --output_dir   src/cm_benchmark/storage/ai2thor/annotated/house_XXXXXX
```
