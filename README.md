# cm-benchmark

A **spatial-cognition QA benchmark** for vision-language models (VLMs).  
The goal is to test whether models build and use an internal **cognitive map** — not whether they can answer from language priors alone.

Items are **multi-image frame sequences** from 3D environments plus a **multiple-choice question**. Every answer must be traceable to simulator metadata (poses, visibility, spatial relations), never invented by an LLM.

---

## Why this project exists

VLMs often look spatially competent when the answer is written in the prompt, visible in a single frame, or recoverable from language statistics. That is not the same as maintaining a map of space over time.

This benchmark asks a harder question:

> After seeing a trajectory through a scene, can the model encode relations, remember what left the view, update bearings after movement, take another entity’s perspective, and reason about routes — from **vision + geometry**, not from text shortcuts?

The capability axis is a **partially ordered pipeline** (later constructs subsume earlier ones). Frame of reference (`egocentric` vs `allocentric`) is a cross-cutting axis on every item.

| Class | Name | Constructs |
|------:|------|------------|
| 1 | Spatial encoding | egocentric encoding · allocentric encoding |
| 2 | Spatial memory | spatial working memory · invisible displacement |
| 3 | Operation on the cognitive map | spatial updating · perspective taking |
| 4 | Navigation / wayfinding | route knowledge · survey knowledge |

Full definitions live in [`configs/taxonomy.yaml`](configs/taxonomy.yaml) (must stay valid YAML).
Per-construct files under [`configs/constructs/`](configs/constructs/): classes 2–4 navigation /
memory / updating / perspective files are **synced excerpts** of the taxonomy blocks;
class-1 and `spatial_working_memory` still use longer generator-style specs. Edit taxonomy
first for shared semantics, then re-sync excerpts.

---

## Architecture (high level)

Two pipelines share one item store, separated by an immutable **FREEZE** wall:

```
CSV / simulator collection
        │
        ▼
┌───────────────────────────────┐
│  Item Generation Pipeline     │  ← build candidates from metadata
│  Task Planner → Q/A/Distractors│
│  GT Validator → Vision check  │
│  Judge / Balancer             │
└───────────────┬───────────────┘
                │ FREEZE (immutable set)
                ▼
┌───────────────────────────────┐
│  Model Evaluation Pipeline    │  ← consume frozen set ONLY
│  Model Runner → Scorer        │
│  Error analysis / reports     │
└───────────────────────────────┘
```

**Invariants that never bend:**

1. Generators never invent spatial facts — only compose from metadata.
2. Every item has a non-null `answer_source`.
3. Ground-truth validation and scoring are deterministic code (no LLM judge of correctness).
4. Frozen items require `vision_necessary == true`.
5. Evaluation never edits or regenerates items.

---

## Where we are now

Core loop: **exported collection folders → episode GT → draft taxonomy questions**.
AI2-THOR / SPOC is the first environment; new collectors plug in by implementing
the same export contract (or a new `NavSequenceGenerator` subclass).

```
collection folder (SPOC episode root or annotations/)
  images/ + annotations/
        │
        ▼
 Ai2ThorNavGenerator → EpisodeStore (SQLite) + optional JSON
        │
        ▼
 generation.draft_items → candidate items JSON (concise + verbose)
```

| In place today | Still upcoming |
|----------------|----------------|
| Episode GT (edges, memory, displacement, layout, sparse tracks) | Trusted object facing → `edges_object_frame` |
| First-draft Q&A per construct (see below) | GT Validator, vision-necessity, FREEZE |
| Frame annotator for spatial review | Model Evaluation pipeline |
| Tunable Q&A FOV visibility filter (+ optional labeling / DT tune) | Generators for other environments (e.g. Matterport3D) |

Collection upstream: [spoc-robot-navigation](https://github.com/andreina-covi/spoc-robot-navigation) (local twin: `spoc-robot-training`). Field brief: [`prompts/ai2thor_collection_extension.md`](prompts/ai2thor_collection_extension.md).

---

## Repository layout

```
cm-benchmark/
├── configs/
│   ├── taxonomy.yaml
│   └── constructs/
├── prompts/
│   └── ai2thor_collection_extension.md   # fields for the collector repo
├── src/cm_benchmark/
│   ├── generator/                        # episode GT from exported CSVs
│   │   ├── nav_sequence_generator.py
│   │   ├── ai2thor_nav_generator.py
│   │   ├── episode_paths.py              # folder / filename discovery
│   │   └── visibility_filters.py         # Q&A FOV keep/drop (tunable)
│   ├── generation/                       # first-draft taxonomy Q&A
│   │   ├── draft_items.py                # CLI
│   │   ├── pipeline.py
│   │   ├── planner.py
│   │   ├── constructs.py
│   │   ├── templates.py
│   │   └── paraphrase.py
│   ├── utils/
│   │   ├── spatial_transformer.py
│   │   ├── spatial_relations.py
│   │   ├── annotate_frames.py            # numbered points + legend
│   │   ├── build_labeling_set.py          # HTML labeling tool + manifest
│   │   └── fit_thresholds.py             # multi-scene LOSO / DT tune + plots
│   └── storage/
│       ├── episode_store.py
│       └── ai2thor/                      # episodes.db, nav_data/, items/, annotated/
├── scripts/                              # one-off helpers (not pipeline)
├── tests/
│   ├── fixtures/
│   └── test_*.py
└── README.md
```

---

## Setup

**Runtime:** Python 3.12  

```bash
# from repo root — use your cm-benchmark environment
pyenv activate cm-benchmark   # or: source path/to/venv/bin/activate

pip install pandas numpy seaborn scikit-learn matplotlib pytest
# optional: install the package editable so imports resolve cleanly
pip install -e .
```

Tests expect `src` on the path (see `pyproject.toml` → `pythonpath = ["src"]`).

---

## Visibility filtering & threshold calibration

SPOC exports **named non-structural** FOV detections with visibility metrics
(`obj-distance`, `visible-pixels`, `bbox-area`, `min-side`, `occupancy-ratio`).
Tiny or barely filled blobs may still appear. For Q&A, `Ai2ThorNavGenerator`
drops them via `question_visibility` hard thresholds that are **on by default**
(even when you do not pass a `visibility_filter.joblib`). Drafting also re-applies
the same filter as a safety net on already-exported episodes.

Default keep rules (override per episode / CLI as needed):

| Key | Default |
|-----|---------|
| `min_bbox_area` | `100` |
| `min_side` | `8` |
| `min_visible_pixels` | `40` |
| `min_occupancy_ratio` | off (`null`) |
| `max_obj_distance` | off (`null`) |

Pass `question_visibility=False` to disable hard thresholds. Prefer a trained
DecisionTree (`visibility_model_path=...`) when available — the model path wins
over hard thresholds at export time.

```text
navigation-*.csv
      │
      ├─► labeling HTML + manifest             build_labeling_set
      │         │
      │         ▼ human labels (*.json)
      │
      └─► multi-scene DecisionTree + LOSO      fit_thresholds (--tune)
                │
                ▼ visibility_filter.joblib  (model + features + low/high)
           Ai2ThorNavGenerator(visibility_model_path=...)
```

Hard AND thresholds (``question_visibility``) are the default when no model is set.

```text
navigation-*.csv
      │
      ├─► labeling HTML + manifest             build_labeling_set
      │         │
      │         ▼ human labels (*.json)
      │
      └─► multi-scene DecisionTree + LOSO      fit_thresholds (--tune)
                │
                ▼ best hyperparameters / rules
           apply as question_visibility (or tree rules)
```

### 1. Human labeling set (per scene)

Builds an HTML tool + `{scene_id}_calibration_manifest.csv` (features for fitting):

```bash
python -m cm_benchmark.utils.build_labeling_set \
  --nav_csv     /path/to/annotations/navigation-house_XXXXXX.csv \
  --images_dir  /path/to/episode/images \
  --scene_id    house_XXXXXX \
  --output_path src/cm_benchmark/storage/ai2thor/output/labeling
```

Open the HTML, label distinguishable / indistinguishable / ambiguous, download
`labels.json`, and **rename** it to match the scene, e.g.
`house_XXXXXX_labels.json`.

### 2. Fit / tune thresholds across scenes (LOSO + plots)

Put all scene pairs in **one folder**:

```text
calibration_scenes/
  house_007514_calibration_manifest.csv
  house_007514_labels.json
  house_001030_calibration_manifest.csv
  house_001030_labels.json
```

```bash
# Fit a tree + leave-one-scene-out with fixed hyperparameters
python -m cm_benchmark.utils.fit_thresholds \
  --folder path/to/calibration_scenes

# Grid-search DecisionTree hyperparameters for LOSO robustness + graphics
python -m cm_benchmark.utils.fit_thresholds \
  --folder path/to/calibration_scenes \
  --tune \
  --tune_out analysis/dt_tune \
  --primary_metric f1 \
  --lambda_std 1.0
```

**Robustness** ranking (default):

\[
\text{robustness} = \mathrm{mean}(\text{F1 or AUC}) - \lambda \cdot \mathrm{std}(\text{across held-out scenes})
\]

`--tune` writes:

| Output | Meaning |
|--------|---------|
| `dt_tune_results.csv` | All grid configs + mean/std/min AUC/F1 + robustness |
| `dt_best_params.json` | Best hyperparameters under that score |
| `plots/top_k_robustness.png` | Top configs by robustness |
| `plots/mean_vs_std_*.png` | Stability frontier (mean vs cross-scene std) |
| `plots/heatmap_depth_x_leaf_*.png` | `max_depth` × `min_samples_leaf` |
| `plots/metric_by_max_depth.png` | Metric distributions vs depth |
| `plots/best_loso_per_scene.png` | Best config’s F1/AUC per held-out scene |
| `visibility_filter.joblib` | Bundle: model + features + low/high bands |

Then filter with the trained model (preferred — no hardcoded mins):

```python
Ai2ThorNavGenerator(
    csv_path_folder=...,
    visibility_model_path="analysis/dt_tune/visibility_filter.joblib",
)
```

Or hard AND thresholds when no model is set:

```python
Ai2ThorNavGenerator(
    csv_path_folder=...,
    question_visibility={
        "min_side": 12,
        "min_occupancy_ratio": 0.3,
        "min_bbox_area": 100,
    },
)
```

Legacy single-scene fit still works:  
`python -m cm_benchmark.utils.fit_thresholds manifest.csv labels.json`

---

## Build episode ground-truth

Episode GT is stored in **SQLite** (system of record). JSON is an **optional** export for inspection and LLM drafting.

### Command (DB by default)

```bash
cd /path/to/cm-benchmark

python -m cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /path/to/collection_run_folder \
  --db_path         src/cm_benchmark/storage/ai2thor/episodes.db
```

`scene_id` / `episode_id` come from `episode_meta-*.json` or filenames when present.  
`--csv_path_folder` may be the **episode root** (`<timestamp>/`) or its **`annotations/`** subfolder.  
Optional overrides: `--scene_id`, `--episode_id`, `--file_navigation`, `--file_objects`, `--file_object_state`, `--file_displacement_events`.

### Also export JSON

```bash
python -m cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /path/to/collection_run_folder \
  --db_path         src/cm_benchmark/storage/ai2thor/episodes.db \
  --export_json \
  --output_path     src/cm_benchmark/storage/ai2thor/nav_data \
  --output_filename nav_data_house_XXXXXX.json
```

### Example (local dataset)

Either the episode root or `annotations/` works:

```bash
# episode root (auto-finds annotations/ + images/)
python -m cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /home/andreina/Documents/Programs/Dataset/Generated/navigation/07_20_2026_17_43_24_824674 \
  --db_path         src/cm_benchmark/storage/ai2thor/episodes.db \
  --episode_id      ai2thor_house_006068 \
  --export_json \
  --output_path     src/cm_benchmark/storage/ai2thor/nav_data \
  --output_filename nav_data_house_006068.json

# or annotations/ directly
python -m cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /home/andreina/Documents/Programs/Dataset/Generated/navigation/07_20_2026_17_43_24_824674/annotations \
  --db_path         src/cm_benchmark/storage/ai2thor/episodes.db \
  --episode_id      ai2thor_house_006068 \
  --export_json \
  --output_path     src/cm_benchmark/storage/ai2thor/nav_data \
  --output_filename nav_data_house_006068.json
```

### Inputs (collection folder)

SPOC layout:

```text
<timestamp>/
  images/img_<t>.png
  annotations/
    navigation-*.csv, objects-*.csv, object_state-*.csv, ...
    episode_meta-*.json, world_layout-*.json
```

| File | Role |
|------|------|
| **navigation-*.csv** | Agent/camera pose, action, image path, **non-structural** FOV dets + bboxes + `obj-distance` / `bbox-area` / `min-side` / `occupancy-ratio` / `visible-pixels` |
| **objects-*.csv** | Object catalog (type, pose, size, receptacles, optional color) |
| **object_state-*.csv** | Per-timestep pose / `visible` / `in_camera_fov` (pickupables; may include hidden rows) |
| **displacement_events-*.csv** | Hidden relocations (`hidden_during`, from/to receptacle + pose) |
| **world_layout-*.json** | Regions, landmarks, passages, connectivity (self-loop `from==to` rows dropped at ingest) |
| **nav_graph-*.json** | Reachable-position grid + edges (`GetReachablePositions`; offline pathfinding) |
| **passage_state-*.csv** | Door/passage open state over time |
| **region_trajectory-*.csv** | Agent region each step |
| **episode_meta-*.json** | `episode_id`, `scene_id`, `camera` (W/H/FOV), `agent` (step sizes), paths, counts |

Walls / floors / ceilings / rooms are excluded from nav FOV edges (room membership uses `current-room` / `region_trajectory`).

**Visibility split**

- **Navigation detections** → `visible_objects` / spatial edges (what is in the RGB frame). Collection may include tiny/occluded blobs; Q&A FOV filtering uses `question_visibility` defaults on `Ai2ThorNavGenerator` (and again at draft time) — see [Visibility filtering & threshold calibration](#visibility-filtering--threshold-calibration).
- **`object_state.in_camera_fov` + pose** → displacement tracks and true pose after moves (catalog poses can be stale).

---

## Sparse tracks (important)

Dense CSVs from collection are **compressed** in the episode GT. We keep the first observation plus later **change points** only.

| Field | Kept objects / rows | Carry-forward helper |
|-------|---------------------|----------------------|
| `object_state_track` | Objects in `displacement_events` only; entries when pose / fov / receptacle / visibility change | `state_at_step(entries, t)` |
| `region_trajectory` | First step + when `region_id` / `region_type` changes | `series_at_step(rows, t)` |
| `passage_state` | Per passage: first + when open/locked/regions change | `series_at_step(rows, t)` |

At query step `t`, use the latest entry with `step|timestep <= t`. Missing step `t` does **not** mean unknown — it means “same as previous change.”

```python
from cm_benchmark.generator.ai2thor_nav_generator import state_at_step, series_at_step

state = state_at_step(episode["object_state_track"]["Cup|1"]["entries"], t=40)
room = series_at_step(episode["region_trajectory"], step=40)
```

---

## Distance vs front/behind

These labels are **independent axes**:

| Field | Meaning |
|-------|---------|
| `distance_label` | How far: `within reach` / `nearby` / `far` / `beyond` |
| `angle_relation[2]` | Depth side of the camera: `front` (`local_z > 0`) or `behind` (`local_z < 0`) |

So `beyond` + `front` = ahead of the agent, but farther than `max_distance`. That is intentional, not a bug.  
`front`/`behind` use the sign of local **z** (camera forward).

---

## Storage model

| Store | Role |
|-------|------|
| **SQLite (`EpisodeStore`)** | System of record — query by episode / step / object / edge |
| **JSON** | Optional artifact for humans and LLM drafting |

Episode table also stores: `object_state_track`, `displacement_events`, `world_layout`, `passage_state`, `region_trajectory`, `episode_meta`.

```python
from cm_benchmark.storage import EpisodeStore

with EpisodeStore("src/cm_benchmark/storage/ai2thor/episodes.db") as store:
    episode = store.load_episode("ai2thor_house_001030")
    edges = store.get_edges("ai2thor_house_001030", step=5, edge_type="inferred")
```

### Episode schema (sketch)

```text
episode
├── scene, episode_meta, thresholds, movement_constant
├── agent_trajectory[] / agent_actions[]
├── route { landmarks[], turns[] }
├── object_state_track          # sparse; displaced objects only
├── displacement_events[]
├── world_layout
├── passage_state               # sparse
├── region_trajectory           # sparse
└── steps[]
    ├── agent, action, image_path, current_room?
    ├── visible_objects / non_visible_objects
    ├── edges_egocentric / edges_allocentric
    ├── edges_object_frame[]    # empty until facing exists
    └── edges_inferred
```

**Non-visible objects** are cumulative: once seen, an object stays in memory and appears under `non_visible_objects` on later steps where it is out of view, with `last_seen_step`. Construct filters can subset later.

---

## How episode GT will be used

1. **First-draft Q&A (code)** — deterministic templates compose items from DB/JSON; answer + `answer_source` locked to edges/tracks/layout.
2. **Optional paraphrase (LLM later)** — may rewrite *question wording only*; never invent geometry or change the answer.
3. **Ground-truth validation (code)** — recompute answers from poses / edges / tracks.
4. **VLM evaluation** — images + question only; exact-match against frozen answers.

Prefer the **DB** in pipeline code; use JSON as a portable snapshot.

---

## First-draft items (taxonomy Q&A)

Hybrid design: **[CODE]** selects eligible facts and locks answer / options / `answer_source`; wording uses templates. Optional `--paraphrase` is a no-op until a provider is wired.

```bash
# from JSON export
python -m cm_benchmark.generation.draft_items \
  --episode_json src/cm_benchmark/storage/ai2thor/nav_data/nav_data_house_001030.json \
  --output       src/cm_benchmark/storage/ai2thor/items/draft_house_001030.json \
  --max_per_construct 2 \
  --swm_min_delay 2 \
  --swm_max_delay 12 \
  --su_min_delay 2 \
  --su_max_delay 12

# from SQLite
python -m cm_benchmark.generation.draft_items \
  --db_path     src/cm_benchmark/storage/ai2thor/episodes.db \
  --episode_id  ai2thor_house_001030 \
  --output      src/cm_benchmark/storage/ai2thor/items/draft_house_001030.json \
  --constructs  egocentric_encoding,invisible_displacement,route_knowledge
```

### Question styles

| Style | Form |
|-------|------|
| `concise` | Short construct template |
| `verbose` | Optional GT-grounded preamble + same query — **must not leak the answer**. For `spatial_working_memory` / `invisible_displacement` only, verbose may also name other static scene objects **without** direction/distance/relation (taxonomy shared_rules exception). |

Paired items share `answer` / `answer_source` and link via `paired_item_id`.

### Multi-image / online sequential wording (classes 2–4)

Questions target **online sequential models** that already observe the navigation
stream. Do **not** attach a bundled multi-image “time order” cue to the question.
Temporal context is linguistic (`{k} steps ago`, `now`, `last {k} navigation steps`).
`image_paths` on a draft item remain for provenance (which stream window the item
covers); they are not the presentation protocol for the question itself.

Construct filters still decide **whether** a question is emitted at a given step
(visibility, translation, displacement, etc.).

Encode→query windows with **no net pose change** (position *and* heading under
tolerance) are rejected for spatial updating — action count alone is not enough.
SWM and invisible displacement still require floor-plane translation between
encode and query (rotate/look-in-place alone is not enough). Provenance
`image_paths` also drop stationary intermediate poses.
`--swm_min_delay` / `--su_min_delay` default to `2`; matching `--*_max_delay`
flags are optional. Generation is deterministic.

### Class 4 — route / survey (important)

| Construct | What the draft asks | Evidence |
|-----------|---------------------|----------|
| `route_knowledge` | MCQ over `derive_turns()` sequences for a walked A→B (scene-calibrated min hops) | `nav_graph` + snapped trajectory; `select_landmark_candidates` for naming |
| `survey_based_route_planning` | **direction_distance**: layout relation of B to A; **conditional_detour**: first-hop after removing edges near a *recorded* closed passage | Landmark poses + untraversed check + `passage_state`; never turn sequences |

Do **not** emit a single question that expects recall of the full egomotion list across hundreds of steps.

### Construct coverage (v0, strict)

If a discriminator cannot be proven from episode GT, the draft is `status: unsupported` (no thin approximations).

| Construct | Draft status |
|-----------|--------------|
| `egocentric_encoding` | full |
| `spatial_working_memory` | full (explicit delay `k`; ego edge at encoding step; optional count mode) |
| `invisible_displacement` | full: direct (`recall_direction` — receptacle or Floor+anchor within `FLOOR_ANCHOR_RADIUS`) and swap; ego bearing; `relation_shift_magnitude` difficulty |
| `spatial_updating` | full when **net pose** changes (position or heading), object static via `object_state_track`, not visible at final; duplicate encode/answer pairs dropped |
| `allocentric_encoding` | `unsupported` until trusted object facing / `edges_object_frame` |
| `route_knowledge` | full (MCQ over walked turn sequences; salience landmarks; calibrated min hops) |
| `survey_based_route_planning` | full for untraversed + perceptually evidenced pairs; optional `conditional_detour` from recorded closures |
| `perspective_taking` | full: A/B/C landmarks, relational A→B heading (`imagined_perspective_label`); no intrinsic-front metadata |

### Display names

Some Objaverse assets log `category: "Undefined"`. Questions fall back to the **object-id stem** (`ObjaScooter|4|5` → `ObjaScooter`), never the placeholder string.

### Item fields (draft)

Core: `item_id`, `construct`, `class`, `frame_of_reference`, `scene_id`, `image_paths`, `question`, `options`, `answer`, `answer_source`, `distractor_rationale`.  
Draft extras: `status` (`ok` \| `thin` \| `unsupported`), `question_style`, `paired_item_id`, `query_step`, `encoding_step`.  
Verification fields stay `null`.

`agent_trajectory` / `agent_actions` are **item-scoped, never an episode dump**:

| Field | Scope |
|-------|-------|
| `agent_trajectory` | One pose per frame in `image_paths`, in image order (`null` if unavailable) |
| `agent_actions` | Only actions in `(encoding_step, query_step]` — the delay / motion the item tests |

`route_knowledge` items carry **no** `agent_actions`: the answer *is* the collapsed action sequence, so the raw list would leak it. The full episode trajectory stays in episode GT, referenced via `answer_source`.

### Build example slides

The presentation builder selects strict (`status: ok`) concise examples
automatically for all eight constructs. A construct without sufficient GT gets
an explicit blocker slide instead of a fabricated example. Temporal items with
more than two images receive ordered sequence slides (six frames per slide)
before their Q&A slide.

```bash
.venv-pptx/bin/python scripts/build_avance_presentation.py \
  --template "/home/andreina/Documents/Programs/Benchmark - avance.pptx" \
  --draft-json src/cm_benchmark/storage/ai2thor/items/draft_house_007514.json \
  --draft-json src/cm_benchmark/storage/ai2thor/items/draft_house_001030.json \
  --output "/home/andreina/Documents/Programs/Benchmark - avance examples.pptx" \
  --examples-per-construct 2
```

`--draft-json` may be repeated to combine episodes. Input order is preference
order. Drafts must reference raw image files that still exist; regenerate a
draft from the current episode export if its original collection folder moved
or was deleted. The slide builder never substitutes images from another
episode because that would break GT traceability.

---

## Annotate frames (spatial review)

Mark each visible object with a **numbered colored circle** (not long text overlays). The right-side **legend** maps `N → object id` and shows **egocentric** relations (`agent → object` from `edges_egocentric`).

```bash
python -m cm_benchmark.utils.annotate_frames \
  --episode_json src/cm_benchmark/storage/ai2thor/nav_data/nav_data_house_001030.json \
  --output_dir   src/cm_benchmark/storage/ai2thor/annotated/house_001030 \
  --start 0 --end 10 \
  --navigation_csv /path/to/annotations/navigation-house_XXXXXX.csv
```

Useful flags: `--step`, `--show_local_xyz`, `--no_relations`.  
Pass `--navigation_csv` when the episode JSON lacks `visible_objects[*].bbox` (older exports).

---

## Tests

```bash
pytest tests/ -q
```

| File | Covers |
|------|--------|
| `test_spatial_transformer.py` | `world_to_local`, projection, 3D→2D |
| `test_spatial_relation.py` | Directions, distance labels, front vs far |
| `test_object_state_track.py` | Sparse tracks + carry-forward |
| `test_navigation_generation.py` | Tiny CSVs + folder episode (displacement / survey) |
| `test_episode_store.py` | SQLite save / load / query |
| `test_episode_paths.py` | Episode root vs `annotations/` discovery |
| `test_draft_items.py` | First-draft Q&A (styles, multi-frame, route segments) |
| `test_annotate_frames.py` | Numbered points + legend |
| `test_visibility_filters.py` | Q&A FOV keep/drop metrics |

---

## Extending to another environment

1. Subclass `NavSequenceGenerator`.
2. Implement `get_records_navigation`, `get_records_objects`, `get_visible_objects`, `update_memory`.
3. Reuse edge building, route summary, sparsify helpers, and JSON/DB export.

---

## Roadmap (short)

- [x] Modular nav GT generator (AI2-THOR)
- [x] Taxonomy-oriented episode GT (trajectory, route, edge splits)
- [x] SQLite EpisodeStore (JSON optional)
- [x] Folder input from SPOC collection (displacement + survey)
- [x] Sparse `object_state_track` / `region_trajectory` / `passage_state`
- [x] Front/behind from local-z; distance labels independent
- [x] First-draft Item Generation (templates + concise/verbose styles)
- [x] Multi-image temporal cues; class-4 source→goal planning (not full-traj recall)
- [x] Tunable Q&A FOV visibility filter (`question_visibility`)
- [x] Optional labeling + multi-scene DT/LOSO threshold tune
- [ ] Object facing → `edges_object_frame` (perspective taking)
- [ ] LLM paraphrase + GT Validator / vision-necessity
- [ ] FREEZE + Model Evaluation pipeline
- [ ] Matterport3D generator

---

## License

See [`LICENSE`](LICENSE).
