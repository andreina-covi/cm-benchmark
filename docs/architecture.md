# Architecture: from a collector folder to benchmark items

This document is for a reader who has not seen the code. It follows the data
from the files the SPOC collector writes to the multiple-choice and
action-sequence items that are evaluated. How to *run* each step is in
[`generation.md`](generation.md); what each construct *means* is in
[`configs/taxonomy.yaml`](../configs/taxonomy.yaml); every number that changes
which items are produced is in [`configs/generation.yaml`](../configs/generation.yaml)
and is explained in [`parameters_report.md`](parameters_report.md).

## The pipeline in one picture

```text
SPOC collection folder  (<timestamp>/images + annotations/*.csv|json)
        │
        │  generator/ai2thor_nav_generator.py   [CODE]   "episode GT"
        ▼
episode GT  (SQLite EpisodeStore, optional nav_data/<scene>/nav_<scene>.json)
        │
        │  generation/generate_items.py → pipeline.py
        │      planner.py      decide WHICH fact to ask and its answer   [CODE]
        │      templates.py    turn the fact into question + options     [CODE]
        │      paraphrase.py   optional rewording of the question only   [AGENT, off]
        ▼
items  (items/<scene>/items_<scene>.json)       ─── FREEZE ───►  evaluation/
```

Nothing after episode GT reads the CSVs again, and nothing in generation
calls a language model to decide an answer. Every answer is recomputed from
episode GT fields and the item records which fields in `answer_source`.

## Stage 1 — episode GT (`src/cm_benchmark/generator/`)

**`episode_paths.py`** finds the files of one episode. A collection folder
holds `navigation-<scene>.csv` and `objects-<scene>.csv` (required) plus
optional logs: object state, displacement events and candidates, passage
state, region trajectory, world layout, nav graph and `episode_meta`. It also
says which object types are structural (Wall, Floor, Ceiling, Room); those
never become question targets.

**`nav_sequence_generator.py`** is the generic episode builder. For each
timestep it asks the subclass for the objects in view, keeps a memory of every
object seen so far, and writes one *step* with: the agent pose, the image
path, `visible_objects` (in view now), `non_visible_objects` (seen before, not
in view now), and edges from the agent to each object (distance bucket and a
coarse direction). It also writes the full `agent_trajectory` and
`agent_actions`. `parse_args` defines the generator CLI.

**`ai2thor_nav_generator.py`** is the AI2-THOR / SPOC subclass. It reads the
CSVs, applies the *collection-time visibility filter* (DecisionTree bundle if
`--visibility_model_path` is given, otherwise static pixel floors) so tiny
detections never enter `visible_objects`, and stores per object: world pose,
the catalog axis-aligned bounding-box (AABB) centre (`bbox_center`), the 2D
pixel box, and visibility metrics (box area, shortest side, visible pixels,
angular width). Displaced objects use the live pose from the object-state log
instead of the catalog pose. It then attaches the displacement events,
successful `placement_candidates` (failed trials and receptacle rows whose
`parent_receptacle` is not the `anchor_id` are dropped at CSV read),
world layout, nav graph and camera size. JSON is written with
`--export_json`. SQLite is written only with `--export_db`.

**`visibility_filters.py`** holds both visibility filters: the static
thresholds (`DEFAULT_QUESTION_VISIBILITY_THRESHOLDS`) and the DecisionTree
bundle (`VisibilityFilterModel`), which labels a detection *distinguible*,
*ambiguo* or *no_distinguible* from its predicted probability. Only
*distinguible* detections pass. The same module re-applies the filter at item
time (`apply_question_visibility_to_episode`) and attaches the model to the
episode so later gates use the same decision.

**`utils/spatial_transformer.py`, `utils/spatial_relations.py`** are the
geometry primitives: world → agent-local coordinates, projection into the
image, and the distance / direction words used on episode edges.

**`storage/episode_store.py`** is the SQLite system of record (`EpisodeStore`).

## Stage 2 — items (`src/cm_benchmark/generation/`)

**`generate_items.py`** is the CLI. It loads every `nav_<scene>.json` under
`--episode_json` (or one episode from SQLite), calls the pipeline once per
scene and writes `items_<scene>.json`.

**`pipeline.py`** runs one episode: `plan_episode` → `build_items_from_facts`
→ optional `paraphrase_question`, and returns item dicts.

**`planner.py`** is where items are *decided*. It re-applies the visibility
filter, then runs one `plan_<construct>` function per construct. Each planner
walks the episode, applies that construct's gates, and returns `PlannedFact`
objects: the query step, the encoding step, the queried object, the locked
answer label, its `answer_source`, the frames to show, the option pool, and
the failure mode behind each distractor (`distractor_seeds`). When no
candidate survives, the planner returns one `unsupported` fact with the reason
(often with per-gate reject counts). Shared helpers answer four questions:
*is this object clearly visible here?* (`_distinguishable_encoding_sighting`),
*what do we call it?* (`_referring_display_name`, which adds a disambiguating
phrase when another object of the same type is in view), *where is it from
the agent?* (`_ego_label_at`), and *did the agent really move?*
(`_has_real_move_between`, `_net_pose_changed_between`).

**`constructs.py`** holds the vocabulary and the label rules: the four
egocentric options and their opposite / orthogonal / mirrored partners, the
question templates per construct, the bearing → label rule
(`angle_to_ego_label`) and its boundary checks (`bbox_edges_cross_sector` /
AABB corners / `angular_span_crosses_sector` fallback),
the perspective-taking rule (`imagined_perspective_label`), the pixel-size
visibility gates (`fov_metrics_ok`, `bbox_not_border_scrape`), the object
naming rule (`object_type_from_id`), and a thin referring-phrase wrapper
(`resolve_referring_disambiguator`). Whole-item duplicate detection,
AABB proximity, colour-dot marks, and the 2D bbox check live in
`disambiguation.py`. Landmark-relative directions are never used.

**`geometry.py`** turns an agent pose plus a world point into an ego label
(`ego_label_from_world_pose`). It uses the agent yaw only, so the camera's
downward pitch never changes a left/right answer, and refuses a label when the
object's visible extent straddles a sector edge (2D bbox edges when in view,
eight AABB corners when hidden).

**`nav_graph.py`** is the class-4 graph toolbox: build the navigability graph
from the exported nav graph, snap the trajectory and landmarks to nodes, derive
walked (`traversed_edges`) and seen (`viewed_edges`) edge sets, replay a
continuous xz + heading walk (`simulate_nav_action_sequence`,
`continuous_bfs_actions` — breadth-first on a per-scene cKDTree), and score a model's action sequence
(`score_route_action_sequence`, `score_survey_action_sequence`).

**`templates.py`** turns a fact into items. It fills the construct template
(`_core_question`), optionally prefixes a verbose scene description that must
not leak the answer (`build_verbose_preamble`), shuffles the options
with `options.shuffle_seed` (one permutation per item) and records why each
wrong option is there (`_shuffle_options`), and copies frame poses / actions /
displacement events
onto the item. One item is written per question style (concise, verbose); the
pair shares the answer and points at each other with `paired_item_id`.

**`schema.py`** defines the item (`CandidateItem`), the construct → class map
and the default frame of reference. Verification fields (`blind_llm_correct`,
`caption_llm_correct`, `vision_necessary`) start null; later pipeline stages
fill them.

**`episode_io.py`** loads episodes and lists / writes the per-scene JSON files.
**`slide_copy.py`** draws review slides (marker overlays, action glyphs); it does
not affect items. **`paraphrase.py`** is an opt-in hook that may reword the
question but never the answer (a no-op without a provider).

## Stage 3 — evaluation (`src/cm_benchmark/evaluation/`)

`protocol.py` holds the one system instruction shared by every model (it
states the class-4 action vocabulary once) and wraps an item for a model.
`score.py` scores class-4 action sequences through the same simulator the
generator used to verify its reference answers. MCQ items are scored by exact
option match.

## Each construct, in call order

All planners are called from `plan_episode` in the order of `--constructs`.
"Named" below always means `_referring_display_name` (or the ID variant),
then a whole-item re-resolve across every shown frame
(`resolve_named_objects`): unique type, else proximity to a unique
co-visible landmark, else a colour-dot mark. Landmark-relative
directions are never used.

### egocentric_encoding (class 1) — "Where is the X relative to you right now?"

1. `plan_egocentric_encoding` walks steps in order; for each visible object:
2. `_distinguishable_encoding_sighting` — visibility model or QUERY pixel floors, not a border scrape, has an ego edge;
3. `find_ego_edge`;
4. `_ego_label_at` → `ego_label_from_world_pose` with the 20° FOV wedge (`AHEAD_HALF_WIDTH_FOV`) at the AABB centre; refused when the 2D bbox edges straddle a sector;
5. `_referring_disambiguator` → `resolve_referring_disambiguator`;
6. `_ego_pool_with_diagnostics` — answer + opposite + orthogonal, padded to four;
7. stops after `max_per_construct` facts (first ones in walk order).

### allocentric_encoding (class 1)

`plan_allocentric_encoding` always returns `unsupported`: the data has no
trusted object facing, so object-to-object frames would be invented.

### spatial_working_memory (class 2) — "{k} steps ago, where was the X relative to you at that moment?"

Relation mode (most of the budget):

1. `plan_spatial_working_memory` walks query steps; candidates are `non_visible_objects` that were never displaced;
2. `_last_distinguishable_sighting` picks the encoding step (latest clear sighting, not the last weak scrape);
3. `_delay_is_allowed` (`--swm_min_delay` / `--swm_max_delay`);
4. `_has_real_move_between` — the delay must contain a floor-plane move;
5. `_ego_label_at` at the encoding pose (20° wedge);
6. a "current view" distractor: the label of another object visible now;
7. `_ego_pool_with_diagnostics`, then `_referring_disambiguator` at the encoding step;
8. frames = `_images_between(encode, query)` — every FOV frame from the sighting through the query, in time order, including turns in place.

Count mode (reserved `max(1, max_items // 3)` slots): for each step,
`_count_category_seen` counts distinct, distinguishable instances per
category; categories with ≥ 2 instances become "How many X have you seen?"
items with off-by-one distractors, encoded at `_first_step_with_category`.

### invisible_displacement (class 2) — "While out of view, the X was moved onto Y. Where is it now?"

1. `plan_invisible_displacement` iterates `displacement_events` with `hidden_during`;
2. `_last_distinguishable_sighting` with the soft `ID_ENCODE_FOV_*` floors (props are small);
3. `_id_displaced_object_name` → `_display_name_parts` names the object at that step;
4. `_candidates_for_event` + `_option_specs_for_event` — the true destination, the original location, a nearby receptacle and a salient decoy (from the collector's teleport trials);
5. `_pick_best_id_query_step` — among steps where the object stays hidden (`_hidden_query_candidates`), the one whose options give the most distinct bearings;
6. `_object_hidden_through`, `_has_real_move_between`;
7. swap events: the partner must also be distinguishable before and hidden after (`_partner_event`, `_object_distinguishable_in_frames`);
   direct moves: the destination must be named — `_landmark_display_in_frames`, or for floor destinations `_nearest_floor_anchor` + `humanize_receptacle` ("the floor near the Sofa");
8. `_try_ego_direction_fact` → `_ego_pool_from_specs` (45° wedge at the query pose) + `_pad_id_ego_options`.
   Shown frames run from the earlier of the object encode and the latest
   pre-query sighting where the destination passes the question-visibility
   floors through the query, in time order. A partial sliver does not count.
   No clear sighting drops the item. ``k`` stays query minus encode.

### spatial_updating (class 3) — "Considering the last k steps, where is the X now?"

1. `plan_spatial_updating` walks query steps over non-displaced `non_visible_objects`;
2. `_last_distinguishable_sighting`, `_delay_is_allowed` (`--su_*_delay`);
3. `_net_pose_changed_between` — position > 0.1 m or heading > 5°;
4. `_object_static_between` — the object did not move;
5. the object must not be visible at the query step;
6. `ego_label_from_world_pose` at the query pose with the 45° wedge (full circle);
7. the pre-move bearing (`_ego_label_at` at encode, 20° wedge) becomes the "non-updating" distractor;
8. `_referring_disambiguator` at the encoding step; repeated (object, encode, answer) triples are dropped;
9. frames = `_images_between(encode, query)` — every FOV frame in that window, in time order. A turn in place is kept, because the question counts that step.

### perspective_taking (class 3) — "Imagine standing at A, facing B. Where is C?"

1. `select_landmark_candidates` (category-unique, `top_n_per_region=4`, `max_total=18`) ← `_distinguishable_landmark_candidates` (salience ranking);
2. `_co_visible_distinguishable_step` keeps a triple only when A, B, and C are distinguishable with mask pixels in one frame; otherwise the triple is dropped;
3. `_referring_display_name` for A, B, C on that frame;
4. `imagined_perspective_label` — signed A→B vs A→C angle → left / right / behind, refused within the boundary margin;
5. distractors: the camera-frame label of C, the mirrored left/right, and the answer under the reversed facing (B behind A).

### route_knowledge (class 4, free action sequence) — "Start at your position in frame i … (the S is in view). Retrace … frame j (the G is in view)."

1. `_load_nav_graph_or_none` → `build_nav_graph`;
2. `_episode_traversed_subgraph` — snap the trajectory, keep walked edges;
3. `select_landmark_candidates(unique_category=False, floor_standing=True)` — endpoints are SPOC floor-standing landmarks, passages excluded — and `_build_landmark_node_map` snapping each landmark to a *visited* node within `ROUTE_LANDMARK_SNAP_M`;
4. `_scene_class4_floors` — median pair geodesic and detour ratio (capped) over walked `traversed_edges` landmark pairs (survey keeps the full-scene pool);
5. for every pair in walk order: `_class4_pair_reject_reason` (at least 2 m geodesic, ≤ 30 m, ≥ scene median, detour ratio), names for both endpoints, reject confusable endpoint pairs, `shortest_path` on walked edges, hop count, real move;
6. recorded `agent_actions` between source and goal sightings, replayed with continuous kinematics; must end within ObjectNav 1.0 m of the goal;
7. turn band `[ROUTE_MIN_TURN_COUNT, ROUTE_MAX_TURN_COUNT]` via `count_direction_changes`;
8. frames (`_sample_class4_frames`): within `n_frames`, lock source/goal sightings and the FOV frames just before/after every rotation between them; fill the remaining budget evenly on straight segments; enforce `max_frame_gap_steps` between consecutive shown frames (drop the item if unmet). Labels are `frame i of n, step s`;
9. scoring is continuous replay: success, SPL, nDTW (`d_th = goal_success_m`), headline SDTW = success × nDTW; edge-membership validity is diagnostic only;
10. all passing pairs are ranked by (turns, geodesic, hops) and the hardest are emitted.

### survey_based_route_planning (class 4) — "Start at your position in frame i … Using the layout, list a route that ends at frame j …" over a never-walked path

1. graph and walked subgraph as above; floor-standing landmarks snapped to any navigable node within `ROUTE_LANDMARK_SNAP_M`;
2. `calibrate_view_radius_m` + `viewed_edges_from_trajectory` — edges near the trajectory count as "seen";
3. per pair: no path on walked edges (`path_exists`), a path on viewed edges, `_class4_pair_reject_reason` with the survey ratio cap, names for both endpoints;
4. reference = one `continuous_bfs_to_goals` flood per source pose (quantised visited cells, action-count order) on viewed nodes, only after the geodesic gate;
5. `_agent_near_landmark` — the agent must be ≥ 1 m from the source at its sighting frame; the reference must contain at least `class4.survey.min_turns` real turns;
6. frames (`_sample_class4_frames`): within `n_frames`, lock source/goal sightings, frames covering the unwalked planned-path edges, and rotation neighbors; fill straight segments; enforce `max_frame_gap_steps` (drop if required covers/gap unmet). Labels are `frame i of n, step s`;
7. scoring: success (headline) and SPL; no answer-level validity gate (validity remains a diagnostic);
8. ranked hardest-first like route.

## Where an item's fields come from

| item field | produced by |
| --- | --- |
| `answer`, `answer_source` | the construct planner (locked before wording) |
| `options`, `distractor_rationale` | planner pool + `templates._shuffle_options` |
| `question`, `question_style` | `templates.build_question` |
| `image_paths`, `image_roles` | planner frame selection; `templates.infer_image_roles` |
| `agent_trajectory`, `agent_actions` | `templates._attach_schema_hooks` (frame poses, window actions) |
| `context` | planner `extra` fields used by slides and the class-4 scorer |
| verification fields | null here; set by later pipeline stages |
