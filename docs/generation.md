# Running generation

Operational notes for episode GT, item generation, visibility, and class-4
scoring. Construct *meaning* lives in [`configs/taxonomy.yaml`](../configs/taxonomy.yaml).
Numeric gates live in [`configs/generation.yaml`](../configs/generation.yaml)
(see [`parameters_report.md`](parameters_report.md)). Algorithms live in
`src/cm_benchmark/generation/planner.py` and `nav_graph.py`. Pass `--config`
to `generate_items` or `ai2thor_nav_generator` to point at another file;
`--swm_min_delay` / `--su_min_delay` / `--max_per_construct` / `--n_frames`
override that file. `--n_frames` is the class-4 frame budget (route and
survey). Omit it to keep `class4.n_frames` in the yaml. Each item records `generation_config` and `generation_config_hash`.

---

## Episode GT

SPOC collection folder:

```text
<timestamp>/
  images/img_<t>.png
  annotations/
    navigation-*.csv, objects-*.csv, object_state-*.csv,
    displacement_events-*.csv, placement_candidates-*.csv,
    passage_state-*.csv, region_trajectory-*.csv
    episode_meta-*.json, world_layout-*.json, nav_graph-*.json
```

`placement_candidates-{scene}*.csv` may also live in a sibling `placement_trials/`
folder. Objects CSV now includes `obj-synset`, `assetId`, `room-id`, `room-type`,
and AABB min/max. Navigation has `action_success` and per-object
`obj-room` / `obj-room-type` (the object's room at that sighting).
Verbose questions name that object room, not the agent's
`current-room-type`. `episode_meta.fov_stride` (or
`config.fov_stride`) > 1 means only steps with `step % fov_stride == 0` have
object rows; those are the only shown frames and visibility evidence.
`world_layout.regions[].area` is the VSI room-area input. Invisible
displacement prefers successful placement candidates over `displacement_events`.
The nav generator keeps only `success==true` placement rows (if the `success`
column is absent, every row is kept — legacy CSVs). Receptacle dests also
require `parent_receptacle == anchor_id` (missing parent is dropped); floor
dests (`anchor-kind==floor` or Floor/Wall/Ceiling/Room `anchor-id`) skip that
parent check. Failed and rejected rows are not written to episode GT / nav JSON.
A per-scene kept / dropped count prints on stdout during ingest.

```bash
python -m cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /path/to/generated/navigation \
  --export_json \
  --output_path     src/cm_benchmark/storage/ai2thor/nav_data
```

`--csv_path_folder` may be one episode (`<timestamp>/` or `annotations/`) or a
root whose children are those episode folders. `scene_id` is read from
filenames such as `navigation-house_007514.csv` (and from `episode_meta` when
that file is present). One `visibility_model_path` applies to every episode.

`--output_path` is a root. One subfolder per scene. SQLite is off unless
you pass `--export_db` (then `--db_path`, default
`src/cm_benchmark/storage/ai2thor/episodes`):

```text
nav_data/house_007514/nav_house_007514.json
episodes/house_007514/episodes.db   # only with --export_db
```

Question generation reads the JSON. `--export_db` writes the same episode
through `EpisodeStore`.

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

## Generate items

The input is the nav generator's `--output_path`. One visibility model applies
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

Optional `--constructs`, `--swm_min_delay`, `--su_min_delay`, `--n_frames`. A single scene
database is `--db_path episodes/house_XXXXXX/episodes.db` plus `--episode_id`.

- **concise** / **verbose** share the same answer. Verbose must not leak it.
- Temporal wording is `{k} steps ago` / `now` — not a bundled “time order” cue.
- `unsupported` is the correct outcome when a scene cannot prove a construct
  (e.g. a straight one-room walk for route knowledge).
- Same-type objects are named across **every shown frame** (not only the
  encode step). Unique type → no extra phrase. Else proximity
  (`the Chair close to the Bread`) when the floor-plane AABB gap and the
  2D pixel-bbox gap are obvious, else a colour-dot mark on the exported
  mask point (`mark-x` / `mark-y`; `the Chair marked with a magenta dot`).
  A required mark with no point drops the item. Questions never contain
  landmark-relative directions. When marks are used, `image_paths` point
  at copies under `<output_path>/<scene>/marked/<item_id>/` — evaluate
  those frames, not the unmarked originals. `item_id` is stable across
  proximity settings (`sr3d` / `vsi` / `current` / `strict` / `all_marks`).
  VSI’s small-room margin is unused: collector regions have no area, so
  VSI always uses 0.30 m.

Class 4 is not MCQ. The action vocabulary is stated once in
`cm_benchmark.evaluation.protocol.SYSTEM_INSTRUCTION`. Scoring replays
continuous xz + heading (`x += m·sin(h)`, `z += m·cos(h)` from
`episode_meta.motion`); a move whose landing point is farther than a
calibrated radius from every nav-graph node is a no-op. Success is ObjectNav
1.0 m. Route's stored reference is the recorded actions between source and
goal; survey's is one breadth-first flood per source pose under the same
kinematics, nearest node from a per-scene cKDTree, and only for pairs that
already pass the geodesic gates. Visited cells are quantised
(`class4.search_state_quant_m`); expansions stay in action-count order so the
stored sequence is the same one a solo search would return.
Validity: route stays on `traversed_edges`; survey stays on `viewed_edges`
and uses at least one unwalked edge. SPL is unchanged. Stored `answer` is a
reference sequence for analysis, not exclusive gold.

Pair selection (both class-4 constructs): geodesic in 1–30 m **and** at least
the scene’s own typical landmark-pair length; a detour ratio of at least this
scene’s typical geo/eucl (capped at 1.1 for route, 1.05 for survey); emit the
hardest surviving pairs. Route also requires real turns. Constants: `planner.py`.

Class-4 frames: an even sample of at most `class4.n_frames` (default 16;
`--n_frames` overrides for one run). Route = FOV walk from the source
sighting minus `class4.route.context_margin_steps` through the goal sighting
plus that margin. Survey = FOV walk from the start of the episode through the
query, always including one frame per planned-path edge (both endpoints within
the view radius of that frame). Both always keep the source and goal
sightings. The item is dropped when the required set does not fit. Letter
markers on source/goal still apply when those frames need them. Evaluation
uses the item’s `image_paths`: unmarked originals when `disambiguation` is
`none` or `proximity`, and the colour-dot copies when it is `mark`.

Spatial updating and spatial working memory show every FOV frame from the
sighting through the query, in time order (including a turn in place).
Perspective taking shows the one frame where A, B, and C are all visible.
Invisible displacement shows the walk from the earlier of the object encode
and a clear destination sighting through the query.

---

## Human review package

Build zip-ready reviewer folders from generated items. Reviewers see exactly
the frames in each item’s `image_paths`. For route and survey they answer
whether the route from A to B can be worked out from those frames (yes/no),
plus optional free-text route and the usual problem flags. MCQ items use the
options (or Can't tell). Multi-frame items hide the question until **Show the
question**.

```bash
python scripts/build_review_package.py \
  --items  src/cm_benchmark/storage/ai2thor/items \
  --output /path/to/review_package \
  --per-construct 25 \
  --reviewers 10 \
  --reviews-per-item 2 \
  --seed 0
```

```bash
python scripts/analyze_review.py \
  --manifest /path/to/review_package/manifest.json \
  --csv /path/to/reviewer_csvs
```

Process and checklist: [`Evaluation_process_manual.md`](../Evaluation_process_manual.md).

---

## Review slides

Example slides do not need a PowerPoint template. Omit `--template` to write a
standalone deck. Pass `--template` only to copy an existing avance deck (intro
slides stay; the progress slide is refreshed) and append examples.

```bash
python scripts/build_avance_presentation.py \
  --items-json src/cm_benchmark/storage/ai2thor/items \
  --output /path/to/examples.pptx
```

```bash
python scripts/build_avance_presentation.py \
  --template /path/to/template.pptx \
  --items-json src/cm_benchmark/storage/ai2thor/items \
  --output /path/to/examples.pptx
```

`--items-json` is `generate_items --output_path`: a root of scene folders, one
scene folder, or one `items_<scene>.json`. Repeat the flag to combine roots.

Proximity-setting ablations (reuse existing nav JSON; do not rebuild it):

```bash
scripts/run_ablation.sh analysis/baseline/nav_data analysis/ablations
python scripts/ablation_report.py analysis/ablations
```

`configs/ablations/*.yaml` are overlays on `generation.yaml`. `item_id` is
the same across settings so the report can pair items for McNemar and
Kendall's tau.

Annotate frames for visibility calibration (not the human item-review UI):

```bash
python -m cm_benchmark.utils.annotate_frames \
  --episode_json src/cm_benchmark/storage/ai2thor/nav_data/house_XXXXXX/nav_house_XXXXXX.json \
  --output_dir   src/cm_benchmark/storage/ai2thor/annotated/house_XXXXXX
```
