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

Full definitions live in [`configs/taxonomy.yaml`](configs/taxonomy.yaml). Per-construct generator specs live in [`configs/constructs/`](configs/constructs/).

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

Episode ground-truth is built from a **SPOC collection folder** (`images/` + `annotations/`). A **first-draft Item Generation** path then builds taxonomy MC candidates from that GT (deterministic templates; answers locked to metadata).

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
| | Matterport3D generator |

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
│   ├── collection/
│   ├── generator/
│   │   ├── nav_sequence_generator.py
│   │   ├── ai2thor_nav_generator.py
│   │   └── episode_paths.py              # folder / filename discovery
│   ├── generation/                       # first-draft taxonomy Q&A
│   │   ├── draft_items.py                # CLI
│   │   ├── pipeline.py                   # orchestrate plan → items
│   │   ├── planner.py                    # construct filters on GT
│   │   ├── constructs.py                 # templates + display names
│   │   ├── templates.py                  # concise / verbose wording
│   │   └── paraphrase.py                 # optional wording hook (off by default)
│   ├── utils/
│   │   ├── spatial_transformer.py
│   │   ├── spatial_relations.py
│   │   └── annotate_frames.py            # numbered points + legend
│   ├── validation/
│   └── storage/
│       ├── episode_store.py
│       └── ai2thor/                      # episodes.db, nav_data/, items/, annotated/
├── tests/
│   ├── fixtures/
│   │   ├── navigation_tiny.csv / objects_tiny.csv
│   │   └── episode_tiny/
│   ├── test_spatial_transformer.py
│   ├── test_spatial_relation.py
│   ├── test_object_state_track.py
│   ├── test_navigation_generation.py
│   ├── test_episode_store.py
│   ├── test_episode_paths.py
│   ├── test_draft_items.py
│   └── test_annotate_frames.py
└── README.md
```

---

## Setup

**Runtime:** Python 3.12  

```bash
# from repo root — use your cm-benchmark environment
pyenv activate cm-benchmark   # or: source path/to/venv/bin/activate

pip install pandas numpy seaborn pytest
# optional: install the package editable so imports resolve cleanly
pip install -e .
```

Tests expect `src` on the path (see `pyproject.toml` → `pythonpath = ["src"]`).

---

## Build episode ground-truth

Episode GT is stored in **SQLite** (system of record). JSON is an **optional** export for inspection and LLM drafting.

### Command (DB by default)

```bash
cd /path/to/cm-benchmark

python -m src.cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /path/to/collection_run_folder \
  --db_path         src/cm_benchmark/storage/ai2thor/episodes.db
```

`scene_id` / `episode_id` come from `episode_meta-*.json` or filenames when present.  
`--csv_path_folder` may be the **episode root** (`<timestamp>/`) or its **`annotations/`** subfolder.  
Optional overrides: `--scene_id`, `--episode_id`, `--file_navigation`, `--file_objects`, `--file_object_state`, `--file_displacement_events`.

### Also export JSON

```bash
python -m src.cm_benchmark.generator.ai2thor_nav_generator \
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
python -m src.cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /home/andreina/Documents/Programs/Dataset/Generated/navigation/07_16_2026_12_39_28_297796 \
  --db_path         src/cm_benchmark/storage/ai2thor/episodes.db \
  --episode_id      ai2thor_house_001030 \
  --export_json \
  --output_path     src/cm_benchmark/storage/ai2thor/nav_data \
  --output_filename nav_data_house_001030.json

# or annotations/ directly
python -m src.cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /home/andreina/Documents/Programs/Dataset/Generated/navigation/07_16_2026_12_39_28_297796/annotations \
  --db_path         src/cm_benchmark/storage/ai2thor/episodes.db \
  --episode_id      ai2thor_house_001030 \
  --export_json \
  --output_path     src/cm_benchmark/storage/ai2thor/nav_data \
  --output_filename nav_data_house_001030.json
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
| **navigation-*.csv** | Agent/camera pose, action, image path, **non-structural** FOV dets + bboxes |
| **objects-*.csv** | Object catalog (type, pose, size, receptacles, optional color) |
| **object_state-*.csv** | Per-timestep pose / `visible` / `in_camera_fov` (pickupables; may include hidden rows) |
| **displacement_events-*.csv** | Hidden relocations (`hidden_during`, from/to receptacle + pose) |
| **world_layout-*.json** | Regions, landmarks, passages, connectivity |
| **passage_state-*.csv** | Door/passage open state over time |
| **region_trajectory-*.csv** | Agent region each step |
| **episode_meta-*.json** | `episode_id`, `scene_id`, `episode_kind`, `images_dir`, `annotations_dir`, counts |

Walls / floors / ceilings / rooms are excluded from nav FOV edges (room membership uses `current-room` / `region_trajectory`).

**Visibility split**

- **Navigation detections** → `visible_objects` / spatial edges (what is in the RGB frame).
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
  --max_per_construct 2

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
| `verbose` | GT-grounded scene preamble (other objects first), then the same query — **must not leak the answer** |

Paired items share `answer` / `answer_source` and link via `paired_item_id`.

### Multi-image wording (classes 2–4)

When an item has **more than one** `image_path`, the question states **time order** and which frame the answer uses (e.g. “now = last image”). Ambiguous “now” without that cue is treated as a bug.

### Class 4 — route / survey (important)

These are **source → goal planning**, not memorizing every action from step 0 to the end of the episode.

| Construct | What the draft asks | Evidence |
|-----------|---------------------|----------|
| `route_knowledge` | Short walked segment (e.g. Kitchen → LivingRoom); compressed action plan | `region_trajectory` change points + turns between those steps; images near start/goal |
| `survey_knowledge` | Layout-based connection (which passage / next region) | `world_layout.connectivity` (BFS); views in source/goal regions when available |

Do **not** emit a single question that expects recall of the full egomotion list across hundreds of steps.

### Construct coverage (v0)

| Construct | Draft status |
|-----------|--------------|
| `egocentric_encoding` | full |
| `spatial_working_memory` | full |
| `invisible_displacement` | full |
| `spatial_updating` | thin (multi-frame “now = last image”) |
| `allocentric_encoding` | thin (object–object edges; no intrinsic facing) |
| `route_knowledge` | short source→goal segments |
| `survey_knowledge` | thin layout planning |
| `perspective_taking` | `status: unsupported` until trusted facing exists |

### Display names

Some Objaverse assets log `category: "Undefined"`. Questions fall back to the **object-id stem** (`ObjaScooter|4|5` → `ObjaScooter`), never the placeholder string.

### Item fields (draft)

Core: `item_id`, `construct`, `class`, `frame_of_reference`, `scene_id`, `image_paths`, `question`, `options`, `answer`, `answer_source`, `distractor_rationale`.  
Draft extras: `status` (`ok` \| `thin` \| `unsupported`), `question_style`, `paired_item_id`, `query_step`, `encoding_step`.  
Verification fields stay `null`.

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
- [ ] Object facing → `edges_object_frame` (perspective taking)
- [ ] LLM paraphrase + GT Validator / vision-necessity
- [ ] FREEZE + Model Evaluation pipeline
- [ ] Matterport3D generator

---

## License

See [`LICENSE`](LICENSE).
