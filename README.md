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

The current code focuses on the **episode ground-truth JSON**: turning navigation + object CSVs into a structured description of the agent’s trajectory and spatial relations. Downstream LLMs will use this JSON to draft questions; a separate validator will recompute answers from the same geometry.

```
navigation.csv + objects.csv
        │
        ▼
 NavSequenceGenerator  (abstract)
        │
        ├── Ai2ThorNavGenerator   ← implemented
        └── MatterportNavGenerator ← planned
        │
        ▼
 episode GT JSON  (poses, visibility, edges, route, placeholders)
```

| Already in the GT JSON | Still placeholder / upcoming |
|------------------------|------------------------------|
| Per-step visible / non-visible objects (cumulative memory) | `facing` / `edges_object_frame` (perspective taking) |
| Egocentric + allocentric edges | `object_state_track` / `displacement_events` (invisible displacement — needs manipulation CSV) |
| Inferred edges for remembered objects | Full Item Generation + Evaluation pipelines |
| `agent_trajectory`, `agent_actions`, landmark-ordered `route` | Matterport3D collector / generator |

---

## Repository layout

```
cm-benchmark/
├── configs/
│   ├── taxonomy.yaml              # capability axis (source of truth)
│   └── constructs/                # one YAML per construct
├── src/cm_benchmark/
│   ├── collection/                # environment collectors (AI2-THOR, Matterport…)
│   ├── generator/
│   │   ├── nav_sequence_generator.py   # abstract GT builder
│   │   └── ai2thor_nav_generator.py    # AI2-THOR CSV → JSON
│   ├── utils/
│   │   ├── spatial_transformer.py      # world ↔ local, 3D → 2D
│   │   └── spatial_relations.py        # directions, distance labels
│   ├── validation/                # GT validator, judge (early stubs)
│   └── storage/
│       ├── episode_store.py       # SQLite EpisodeStore (system of record)
│       └── ai2thor/               # DB + optional JSON artifacts
├── tests/
│   ├── fixtures/                  # tiny CSVs for end-to-end tests
│   ├── test_spatial_transformer.py
│   ├── test_spatial_relation.py
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

Episode GT is stored in **SQLite** (scalable system of record). JSON is an **optional** export for inspection and LLM drafting.

### Command (DB by default)

```bash
cd /path/to/cm-benchmark

python -m src.cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_navigation /path/to/navigation-Procedural.csv \
  --csv_path_objects    /path/to/objects-Procedural.csv \
  --db_path             src/cm_benchmark/storage/ai2thor/episodes.db \
  --episode_id          ai2thor_Procedural
```

### Also export JSON (optional)

```bash
python -m src.cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_navigation /path/to/navigation-Procedural.csv \
  --csv_path_objects    /path/to/objects-Procedural.csv \
  --db_path             src/cm_benchmark/storage/ai2thor/episodes.db \
  --episode_id          ai2thor_Procedural \
  --export_json \
  --output_path         src/cm_benchmark/storage/ai2thor/nav_data \
  --output_filename     nav_data1.json
```

### Example (local dataset)

```bash
python -m src.cm_benchmark.generator.ai2thor_nav_generator \
  --csv_path_navigation /home/andreina/Documents/Programs/Dataset/Generated/navigation/07_10_2026_15_37_47_286336/navigation-house_007514.csv \
  --csv_path_objects    /home/andreina/Documents/Programs/Dataset/Generated/navigation/07_10_2026_15_37_47_286336/objects-house_007514.csv \
  --db_path             src/cm_benchmark/storage/ai2thor/episodes.db \
  --episode_id          ai2thor_house_007514 \
  --export_json \
  --output_path         src/cm_benchmark/storage/ai2thor/nav_data \
  --output_filename     nav_data2.json
```

### Inputs

| CSV | Role |
|-----|------|
| **Navigation** | Per-timestep agent/camera pose, action, image path, visible `obj-id`s (and bboxes) |
| **Objects** | Static (for now) object metadata: type, world pose, rotation, size, receptacles |

### Storage model

| Store | Role |
|-------|------|
| **SQLite (`EpisodeStore`)** | System of record — query by episode / step / object / edge type without loading the full blob |
| **JSON** | Optional artifact for humans and LLM question drafting |

Tables: `episodes`, `steps`, `step_objects`, `edges`, `route_landmarks`, `route_turns`.

```python
from cm_benchmark.storage import EpisodeStore

with EpisodeStore("src/cm_benchmark/storage/ai2thor/episodes.db") as store:
    episode = store.load_episode("ai2thor_house_007514")
    edges = store.get_edges("ai2thor_house_007514", step=5, edge_type="inferred")
```

### What an episode contains

```text
episode
├── scene, thresholds, movement_constant
├── agent_trajectory[]          # pose + image per step
├── agent_actions[]             # ordered actions
├── route
│   ├── landmarks[]             # nearby objects first seen along the path
│   └── turns[]                 # executed actions/degrees
├── object_state_track          # null until manipulation CSV exists
├── displacement_events         # [] until manipulation CSV exists
└── steps[]
    ├── agent, action, image_path
    ├── visible_objects         # category, position, local_position, angles, facing=null
    ├── non_visible_objects     # cumulative memory + last_seen_step
    ├── edges_egocentric        # agent → object (Class 1 ego / updating)
    ├── edges_allocentric       # object → object in agent frame (Class 1 allo)
    ├── edges_object_frame      # [] until object facing is available
    └── edges_inferred          # agent → remembered object at current pose
```

**Non-visible objects** are cumulative: once seen, an object stays in memory and appears under `non_visible_objects` on every later step where it is out of view, with `last_seen_step` for working-memory depth. Construct filters can subset later; the GT does not throw map information away.

---

## How episode GT will be used

1. **Question / answer drafting (LLM)** — compose items from DB/JSON fields (object ids, edges, trajectory, route). The model must not invent geometry.
2. **Ground-truth validation (code)** — recompute the answer from poses / edges and reject mismatches.
3. **VLM evaluation** — models see images + question only; scores come from exact match against the frozen answer.

Prefer reading from the **DB** in pipeline code; use JSON when you need a portable snapshot.

Until Item Generation is wired, treat episode GT as the **shared spatial ledger**.

---

## Tests

```bash
# from repo root, with the project env active
pytest tests/ -q
```

| File | What it covers |
|------|----------------|
| `test_spatial_transformer.py` | `world_to_local`, projection, 3D→2D (including behind-camera / depth borderline) |
| `test_spatial_relation.py` | Direction and distance labels at threshold borders |
| `test_navigation_generation.py` | End-to-end: tiny CSVs → episode dict / JSON |
| `test_episode_store.py` | Save / load / query episode GT in SQLite |

Fixtures live in `tests/fixtures/` (`navigation_tiny.csv`, `objects_tiny.csv`).

---

## Extending to another environment

1. Subclass `NavSequenceGenerator`.
2. Implement `get_records_navigation`, `get_records_objects`, `get_visible_objects`, `update_memory` for that environment’s CSV (or API) schema.
3. Reuse the parent’s edge building, route summary, and JSON export.

Matterport3D is planned as the next subclass alongside the existing collector stub under `collection/`.

---

## Roadmap (short)

- [x] Modular nav GT generator (AI2-THOR)
- [x] Taxonomy-oriented episode JSON (trajectory, route, edge splits, placeholders)
- [x] SQLite EpisodeStore (system of record; JSON optional)
- [x] Unit + end-to-end tests for transforms and relations
- [ ] Object facing → `edges_object_frame` (perspective taking)
- [ ] Manipulation CSV → `object_state_track` / `displacement_events`
- [ ] Item Generation pipeline (planner, question/distractor agents, validator)
- [ ] FREEZE + Model Evaluation pipeline
- [ ] Matterport3D generator

---

## License

See [`LICENSE`](LICENSE).
