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

Episode ground-truth is built from a **SPOC collection folder** (navigation + objects + displacement + survey logs). Downstream LLMs draft questions from that GT; a validator recomputes answers from the same geometry.

```
collection folder (SPOC / AI2-THOR)
  navigation, objects, object_state, displacement_events,
  world_layout, passage_state, region_trajectory, episode_meta
        │
        ▼
 NavSequenceGenerator  (abstract)
        │
        ├── Ai2ThorNavGenerator   ← implemented (folder input)
        └── MatterportNavGenerator ← planned
        │
        ▼
 EpisodeStore (SQLite) + optional JSON export
```

| In the GT today | Still upcoming |
|-----------------|----------------|
| Per-step visible / non-visible objects (cumulative memory) | `facing` / `edges_object_frame` (perspective taking) |
| Egocentric + allocentric edges; inferred edges | Item Generation + Evaluation pipelines |
| `agent_trajectory`, `agent_actions`, landmark-ordered `route` | Matterport3D generator |
| Sparse `object_state_track` (displaced objects only) + `displacement_events` | |
| Sparse `region_trajectory` / `passage_state` + `world_layout` | |

Collection upstream: [spoc-robot-navigation](https://github.com/andreina-covi/spoc-robot-navigation). Field brief for collectors: [`prompts/ai2thor_collection_extension.md`](prompts/ai2thor_collection_extension.md).

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
│   ├── utils/
│   │   ├── spatial_transformer.py
│   │   └── spatial_relations.py
│   ├── validation/
│   └── storage/
│       ├── episode_store.py
│       └── ai2thor/
├── tests/
│   ├── fixtures/
│   │   ├── navigation_tiny.csv / objects_tiny.csv
│   │   └── episode_tiny/                 # full folder-style episode
│   ├── test_spatial_transformer.py
│   ├── test_spatial_relation.py
│   ├── test_object_state_track.py        # sparse tracks + carry-forward
│   ├── test_navigation_generation.py
│   └── test_episode_store.py
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

```bash
python -m src.cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_folder /home/andreina/Documents/Programs/Dataset/Generated/navigation/07_13_2026_16_32_01_395596/nav_generator \
  --db_path         src/cm_benchmark/storage/ai2thor/episodes.db \
  --episode_id      ai2thor_house_001030 \
  --export_json \
  --output_path     src/cm_benchmark/storage/ai2thor/nav_data \
  --output_filename nav_data_house_001030.json
```

### Inputs (collection folder)

| File | Role |
|------|------|
| **navigation-*.csv** | Agent/camera pose, action, image path, visible dets + bboxes |
| **objects-*.csv** | Object catalog (type, pose, size, receptacles) |
| **object_state-*.csv** | Per-timestep pose / `visible` / `in_camera_fov` (dense in collection; may include hidden rows) |
| **displacement_events-*.csv** | Hidden relocations (`hidden_during`, from/to receptacle + pose) |
| **world_layout-*.json** | Regions, landmarks, passages, connectivity |
| **passage_state-*.csv** | Door/passage open state over time |
| **region_trajectory-*.csv** | Agent region each step |
| **episode_meta-*.json** | `episode_id`, `scene_id`, `episode_kind`, counts |

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

1. **Question / answer drafting (LLM)** — compose items from DB/JSON fields; never invent geometry.
2. **Ground-truth validation (code)** — recompute answers from poses / edges / tracks.
3. **VLM evaluation** — images + question only; exact-match against frozen answers.

Prefer the **DB** in pipeline code; use JSON as a portable snapshot.

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
- [ ] Object facing → `edges_object_frame` (perspective taking)
- [ ] Item Generation pipeline
- [ ] FREEZE + Model Evaluation pipeline
- [ ] Matterport3D generator

---

## License

See [`LICENSE`](LICENSE).
