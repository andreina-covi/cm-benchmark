# Parameters that affect benchmark quality

A value is listed here if changing it changes which items are generated, which
objects / landmarks / pairs are selected, the ground-truth answer, the
distractors, or what the model sees in the frames. Values that only change
review slides, thumbnails, file paths or tooling stay in code and are not
listed.

The audit covers every module-level constant, numeric default argument and
numeric literal inside a condition under `src/` (AST scan, then a manual pass
over arithmetic such as `max(grid, move) * 1.5`). Line numbers are as of this
audit, before centralisation. After phase 3 every value below lives in
[`configs/generation.yaml`](../configs/generation.yaml) under the key in the
**config key** column.

## How to read the justification column

- **literature: <source, value>** — the value is copied from a named paper or benchmark.
- **calibrated: <how>** — fitted from data in this repo.
- **derived: <reason>** — follows from another fixed quantity (step size, camera, label set).
- **none found** — no source. The commit that introduced the value (`git log -S`) is quoted, with its comment if it had one.

"Ablation range" is a suggestion for a sensitivity sweep, not a claim that a
value in that range is better.

## Collected scene facts used to judge "derived"

| quantity | value in the three collected episodes | source |
| --- | --- | --- |
| frame size | 396 × 224 px | `episode_meta.camera` |
| vertical FOV | 58.7° / 59.6° / 58.7° | `episode_meta.camera.fov_vertical_deg` |
| horizontal FOV | ≈ 90° (2·atan(tan(fov_v/2)·396/224)) | derived |
| agent step | 0.20 m | `world_layout.nav_metadata.agent_move_m` |
| rotation step | 45° | `episode_meta.agent.rotation_deg` |
| reachability grid | 0.15 m | `world_layout.nav_metadata.grid_size` |
| room polygons / areas | **not exported** (regions have only a centre and landmark ids) | `world_layout.regions` |

---

## 1. Visibility (what counts as "clearly visible")

| name | current value | file:line (all copies) | what it controls | constructs affected | effect of raising / lowering | justification | suggested ablation range | config key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DecisionTree bundle | `visibility_filter.joblib` (5 features, depth from tuning) | file loaded by `visibility_filters.py:100` (`resolve_visibility_model_path`) | primary keep/drop decision for every detection, at collection and at item time | all | a stricter tree drops more objects → fewer candidates, fewer ambiguous names | calibrated: `utils/fit_thresholds.py` on human labels (`{scene}_labels.json`), leave-one-scene-out tuning | refit on more scenes; compare with "no model" (static floors) | `visibility.model_path` |
| proba band (from bundle) | low 0.0409, high 0.8994 | stored in the joblib bundle | P(distinguishable) below low → no, above high → yes, between → ambiguous (rejected) | all | narrower band → more objects pass | calibrated: min / max P over 125 human "ambiguo" rows (`calibrate_ambiguous_band`) | — (refit, do not hand-edit) | in bundle |
| `DEFAULT_PROBA_LOW` / `HIGH` | 0.3 / 0.7 | `visibility_filters.py:63-64`, used at 430-431, 482-483, 526-532, 558-559 | fallback band when a bundle has none | all (fallback only; the shipped bundle overrides it) | as above | none found (b6fc1ca "Updating filtering visibility methods for objects", no comment) | 0.2–0.4 / 0.6–0.8 | `visibility.proba_low` / `proba_high` |
| degenerate band half-width | ±0.05 | `visibility_filters.py:590-591` | band used when all ambiguous labels share one probability | all (calibration only) | wider → more detections ambiguous | none found ("keep a small margin around it") | 0.02–0.1 | `visibility.degenerate_band_half_width` |
| `named_min_frame_fraction` | 0.01 (887 px of 396×224) | `constructs.py` `frame_fraction_ok`, called from `fov_metrics_ok` | hard floor on mask pixels for every named object (target, landmark, PT A/B/C, receptacle, displaced prop), with or without the DecisionTree | all | higher → fewer items, fewer small-object errors | calibrated: review round 1, 43 MC items with named objects; reviewers were wrong 74% of the time on items whose weakest named object was below 1%, vs 49% at or above it; 5% (ReVSI, arXiv 2604.24300) would drop 79% of items | 0.005–0.05 | `visibility.named_min_frame_fraction` |
| `named_min_in_frame_fraction` | 0.45 (target 0.50) | `constructs.py` `in_frame_ok`, called from `fov_metrics_ok` | share of a named object's unoccluded projected box inside the frame (SPOC `in-frame-fraction`), in the frame where it is named | all | higher → fewer items, fewer border-cut objects (a plant showing a few leaves) | literature: MindEdit-Bench (arXiv 2607.00491) visible > 0.50; Long Time No See (arXiv 2609.34630) out of view when under half of the footprint is inside the view; SpatialMosaic (arXiv 2512.23365) field-of-view truncation ratio. 0.45–0.50 kept as a tolerance band (user decision 2026-11-18, round-7 review) | 0.4–0.8 | `visibility.named_min_in_frame_fraction` |
| static floors `min_bbox_area` / `min_side` / `min_visible_pixels` | 100 px² / 8 px / 40 px | `visibility_filters.py:70-73`; **copy** `constructs.py:440-442` (`ID_ENCODE_FOV_*`) | keep rule when no DecisionTree is loaded; collection-time and item-time Q&A filter | all (fallback) | higher → small props vanish from `visible_objects` and memory | none found (b418bda; comment "Tuned to drop speck detections (Potato/Wrench ~10px) while keeping furniture") | area 50–400, side 4–16, pixels 20–160 | `visibility.static.*` |
| `QUERY_FOV_MIN_BBOX_AREA` / `MIN_SIDE` / `MIN_VISIBLE_PIXELS` | 1200 px² / 32 px / 300 px | `constructs.py:428-430` | stricter bar for queried objects, landmarks, PT A/B/C, class-4 endpoints when no DecisionTree is loaded | ego, SWM, SU, PT, route, survey, disambiguation landmarks | higher → fewer but more legible targets | none found (b418bda set 800/24/200, comment "Tuned above HousePlant@s5 (bbox≈432, side≈16) which was not human-distinguishable in review"; c87ae77 raised to 1200/32/300, comment "fallback only — prefer refitting visibility_filter.joblib") | area 800–2400, side 24–48, pixels 200–600 | `visibility.query.*` |
| `QUERY_FOV_EDGE_MARGIN_PX` | 2 px | `constructs.py:434` | how close to the border a bbox counts as clipped | same as QUERY_FOV | larger → more boxes treated as clipped | none found (c87ae77, "geometric rule; kept even when the DecisionTree is active") | 1–8 | `visibility.border.edge_margin_px` |
| `QUERY_FOV_MIN_EXTENT_IF_CLIPPED` | 48 px | `constructs.py:435` | minimum extent along a clipped axis (rejects thin border strips) | same as QUERY_FOV | higher → more border objects rejected | none found (c87ae77) | 24–96 | `visibility.border.min_extent_if_clipped_px` |
| `ID_ENCODE_FOV_*` | 100 / 8 / 40 | `constructs.py:440-442` (same values as the static floors) | encode-time bar for displaced props in invisible_displacement | invisible_displacement | higher → fewer displacement events usable | none found (5c7eb3c, comment "otherwise every swap/direct event is rejected for 'no distinguishable sighting'") | as static floors | `visibility.id_encode.*` |

## 2. Directions (egocentric and perspective labels)

| name | current value | file:line (all copies) | what it controls | constructs affected | effect of raising / lowering | justification | suggested ablation range | config key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `AHEAD_HALF_WIDTH_FOV` | 20° | `constructs.py:232`; **copy** `spatial_relations.py:74` (`get_direction_angle` default); used `constructs.py:380`, planner ego / SWM / SU-distractor / PT camera-frame calls | half-width of "ahead of you" when the object is in the current view | egocentric_encoding, SWM (relation), SU (non-updating distractor), PT (camera-frame distractor), episode edge labels | wider → more "ahead" answers; narrower → more left/right | none found (d56b883, comment "a 45° half-width collapses almost everything to 'ahead of you'"; the horizontal FOV is ≈ ±45°) | 10–30 | `directions.ahead_half_width_fov_deg` |
| `AHEAD_HALF_WIDTH_FULL` | 45° | `constructs.py:234`; **copy** literal default `constructs.py:202` (`angle_to_ego_label`); defaults at `constructs.py:317, 345`, `geometry.py:43`, `planner.py:376` | half-width of "ahead" when the object may be anywhere around the agent | spatial_updating (answer), invisible_displacement | 45 = four equal 90° sectors | derived: four equal 90° sectors; its behind boundary (135°) matches literature: VSI-Bench relative direction, "back = turn ≥ 135°" | 30–60 | `directions.ahead_half_width_full_deg` |
| `PT_DIRECTION_BACK_DEG` | 135° | `constructs.py:242` | \|signed angle\| ≥ this → "behind" in perspective taking | perspective_taking | higher → fewer "behind" answers | literature: VSI-Bench relative direction (medium), back = turn ≥ 135° | 120–150 | `directions.pt_back_deg` |
| `PT_DIRECTION_BOUNDARY_MARGIN_DEG` | 10° | `constructs.py:241` | rejects PT triples within this of ±45° / ±135° (four 90° sectors, ahead edge = `ahead_half_width_full_deg`) | perspective_taking | higher → fewer but less ambiguous PT items | none found (c87ae77; current comment "Not a published VSI-Bench / ViewSpatial constant … 10° is this benchmark's PT gate only") | 0–20 | `directions.pt_boundary_margin_deg` |
| sector-straddle refusal | no constant (uses bbox / AABB / optional `ang-width-deg`) | `constructs.py` (`bbox_edges_cross_sector`, `bearings_cross_sector`); `geometry.py` (`aabb_corners_cross_sector`, `direction_extent_crosses_sector`) | refuses an ego label if the object's extent spans more than one sector | ego, SWM, SU, ID | — | derived: visible → 2D bbox left/right edges via camera FOV; hidden → 8 AABB corners; ang-width fallback (VSI-Bench / ViewSpatial drop ambiguous directions) | on / off | `directions.refuse_straddling_objects` |
| `label_margin_deg` | 10° | `constructs.py` (`bearings_cross_sector`, `angular_span_crosses_sector`); `geometry.py` (`direction_extent_crosses_sector`) | every extent bearing (bbox edges, AABB corners, or the centre) must be at least this far from a sector edge, else no ego label | ego, SWM, SU, ID | higher → fewer but less ambiguous direction items | calibrated: review round 1, 36 direction items; human accuracy by distance of the object to the nearest edge: 13% (<5°), 45% (5–10°), 54% (10–15°), 51% (15–25°), 62% (≥25°). Same idea as VSI-Bench / 3DSRBench dropping ambiguous directions; matches the PT margin | 5–15 | `directions.label_margin_deg` |
| options per MCQ item | 4 | `planner.py:206, 210, 1111, 1119, 1122, 2635`, `templates.py:111` | pool size after padding | all MCQ constructs | fewer → higher chance baseline | derived: there are exactly four ego labels | 3–4 | `options.n_options` |
| options when the object is in the shown frame | 3 (no "behind you") | `planner.py` `_ego_pool_with_diagnostics(visible_only=True)` | egocentric_encoding and SWM recall_relation never offer "behind you", which a visible object cannot be | egocentric_encoding, SWM (relation) | 4 would add a give-away distractor | derived: an object in view is never behind the camera; VSI-Bench easier levels drop "back" | 3 | `options.n_options_visible` |

## 3. Disambiguation of same-type objects

| name | current value | file:line (all copies) | what it controls | constructs affected | effect of raising / lowering | justification | suggested ablation range | config key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `MIN_DISAMBIG_MARGIN_M` | 0.6 m | `constructs.py:421` (default at `find_disambiguator`) | "the X close to the Y" only if the nearest sibling is this much farther from Y | every construct that names a duplicated object | higher → fewer proximity phrases, more skipped candidates | none found (b418bda, comment "Referring expressions must be obvious in the image, not just metrically true") | 0.15–1.0 (phase 4 settings sr3d / vsi / current / strict) | `disambiguation.settings.current.margin_m` |
| `MIN_DISAMBIG_MARGIN_RATIO` | 1.5 | `constructs.py:422` | sibling distance must also be ≥ ratio × target distance | same | higher → fewer proximity phrases | none found (b418bda) | none, 1.5, 2.0 | `disambiguation.settings.current.ratio` |
| landmark-relative direction fallback | 20° wedge, landmark → target in agent yaw | removed | "the X left of the Y" when proximity fails | — | — | none found | **removed** (questions must not contain landmark-relative directions) | — |
| AABB proximity + 2D bbox check | setting-specific margin / ratio; 2D `ratio_2d=1.5`, `min_gap_2d_frac=0.10` | `disambiguation.py` | whole-item duplicate naming on AABB min/max gap; 2D fail → mark, not skip | every named object | stricter → more marks, fewer proximity phrases | literature: sr3d 0.20 m; VSI-Bench 0.15 / 0.30 m split at 40 m². Exported regions have no area, so the large margin is used | `sr3d` / `vsi` / `current` / `strict` / `all_marks` | `disambiguation.*` |
| mark ids | 7 | `disambiguation.mark_max_ids` | numbers 1..n (seeded permutation per confusable group) assigned only to instances in that item's shown frames; skip the type if those frames have more instances than this | same | lower → more skipped duplicates | none found; numbers replace colour names after round-1 reviewers could not name or tell apart cyan / lime / magenta (Set-of-Mark, Yang et al. 2023, uses numeric IDs) | 3–9 | `disambiguation.mark_max_ids` |
| mark radius | 0.02 × width, min 7 px | `disambiguation.mark_radius_frac`, `mark_min_radius_px` | badge size (white bold digit, black disc, white ring) | same | larger → more legible, hides more of small objects | none found; ≈8 px at 396 px keeps a 1%-floor (~30×30 px) object mostly visible | 0.015–0.03 | `disambiguation.mark_radius_frac` |

## 4. Per construct

| name | current value | file:line (all copies) | what it controls | constructs affected | effect of raising / lowering | justification | suggested ablation range | config key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `max_per_construct` | 3 (CLI); `run_baseline.sh` uses 5 | `generate_items.py` | whole-set target per construct (this × number of scenes), filled by a seeded round-robin after the per-scene cap | all | more → easier items also emitted | none found. `null` = no cap; `configs/generation_max.yaml` sets all three budget keys to `null` and `max_label_share: 1.0` for large runs (every item that passes the gates is kept; balance labels when sampling an evaluation split) | 2–10 or null | `budget.max_per_construct` |
| `max_per_scene` | 2 concise facts | `generate_items.py` | most concise facts one scene may contribute to one construct before the round-robin | all | higher → one scene can fill more of the target | none found | 1–4 | `budget.max_per_scene` |
| SWM delay window (also EE in_sequence, heading-change span) | min 2, max none (CLI, capped at `(max(frames.budgets) - 1) * frames.max_gap_steps` = 155); baseline run 5–10 | `generate_items.py:87`, `pipeline.py:29`, `planner.py:501, 2684` | allowed k between encoding and query | spatial_working_memory | longer → harder recall, fewer items | none found | min 2–10, max 5–30 | `spatial_working_memory.min_delay` / `max_delay` |
| SWM count budget | `max(1, max_items // 3)` when max_items ≥ 2 | `planner.py` | slots reserved for "how many X have you seen" | spatial_working_memory | more → more count items, fewer relation items | none found (5c7eb3c, comment "Relation used to fill max_items and return early, so recall_count never ran") | 0–½ of budget | `spatial_working_memory.count_divisor` |
| SWM count minimum | n ≥ 2 | `planner.py:637` | categories counted only with ≥ 2 instances | spatial_working_memory | — | derived: n = 1 has no memory load (taxonomy "load") | 2–3 | `spatial_working_memory.count_min_instances` |
| SWM count distractors | double_counted, n−1, n+1, n+2 | `planner.py` (`_count_options`) | `double_counted` = sum of sighting runs (every return counted as a new object), skipped if equal to the answer; then off-by-one, then off-by-two; one tag each | spatial_working_memory | — | derived: the re-identification error and off-by-one errors (user decision 2026-11-25) | — | (structural) |
| SWM count re-identification | ≥ 1 instance | `planner.py` (`_count_sighting_runs`) | counted instances that leave view (no pixels at all) and come back within the shown frames; stored as `n_reseen_instances` | spatial_working_memory | higher → fewer count items | derived: answer not contained in a single frame (VSI-Bench counting + re-identification; user decision 2026-11-25) | 1–2 | `spatial_working_memory.count_min_reseen_instances` |
| SWM heading turn event | net ≥ 55° | `planner.py` (`heading_turn_events`) | rotations within `class4.mc.turn_merge_m` (1.0 m) of travel form one cluster; a cluster is a turn event when its net is ≥ this (90/135/180°); 45° = correction, a glance = net 0 | spatial_working_memory | — | derived: between SPOC's 45° and 90° steps (user decision 2026-11-25) | 50–85 | `spatial_working_memory.heading_event_min_deg` |
| SWM heading turn events | ≥ 2 | `planner.py` (`_plan_heading_change`) | separate turn events required between frame i and the last frame | spatial_working_memory | higher → fewer, harder items | derived: answer not contained in a single turn (inspired by OST-Bench Agent State; user decision 2026-11-25) | 2–4 | `spatial_working_memory.heading_min_turn_events` |
| SWM heading per-house cap | 15 (null in `generation_max.yaml`) | `planner.py` (`_plan_heading_change`) | heading-change items kept per house (one episode per house); inert under the default `budget.max_per_scene` 2, used by `configs/ablations/heading_cap.yaml` | spatial_working_memory | lower → more even houses, fewer items | none found (user decision 2026-11-26: four Round-2 houses gave most heading items) | 10–20 | `spatial_working_memory.heading_max_per_house` |
| SWM heading view overlap | frames i and n share no clearly seen object | `planner.py` (`_plan_heading_change`) | rejects spans answerable by matching the two views; `displacement_m` stored | spatial_working_memory | — | derived: answer not readable from frames i and n (user decision 2026-11-26) | — | (structural) |
| SWM heading cones | ±20° | `planner.py` (`heading_label`) | net yaw change must lie within this of 0 / −90 / +90 / 180° (same / left / right / around); others rejected | spatial_working_memory | — | shared `directions.near_cardinal_max_deg` | — | `directions.near_cardinal_max_deg` |
| SU delay window | min 2, max none (CLI, capped as SWM); baseline run 5–10 | `generate_items.py:99`, `pipeline.py:31`, `planner.py:1466, 2686` | k between encoding and query | spatial_updating | as SWM | none found | as SWM | `spatial_updating.min_delay` / `max_delay` |
| net pose change | 0.1 m or 5° | `planner.py:1452`; **copy** `constructs.py:737` (`net_pose_changed`) | the agent must really have moved/turned for SU | spatial_updating | higher → only bigger pose changes count | none found (05535fa "Aligning and fixing inconsistencies", no comment) | 0.05–0.5 m, 5–45° | `spatial_updating.min_pos_change_m` / `min_heading_change_deg` |
| object static tolerance | 0.05 m | `planner.py:1425` (`_object_static_between`) | object must not move between encode and query in SU | spatial_updating | higher → tolerates small physics jitter | none found (05535fa, no comment) | 0.01–0.1 | `spatial_updating.object_static_eps_m` |
| real-move epsilon | 0.001 m | `planner.py:214` (`_POSITION_EPS_M`) | "the agent moved on the floor plane" for SWM / ID / route | SWM, ID, route | — | derived: float noise, far below the 0.20 m step | 1e-4–0.05 | `motion.position_eps_m` |
| `FLOOR_ANCHOR_RADIUS_M` | 1.2 m | `planner.py:690` | floor destination needs a named landmark this close | invisible_displacement | larger → more floor events kept, vaguer anchors | none found (05535fa, comment "Floor destinations need a nearby distinguishable landmark for the question cue") | 0.6–2.0 | `invisible_displacement.floor_anchor_radius_m` |
| ID diagnostic minimum | ≥ 2 distinct ego labels among the option specs | `planner.py:1137, 1209` | at least one diagnostic distractor besides the answer | invisible_displacement | 3 → fewer items, more diagnostic pools | derived: invariant 7 (each item has a diagnostic distractor) | 2–3 | `invisible_displacement.min_diagnostic_labels` |
| ID query delay cap | 10 | `planner.py` (`_id_earliest_query_window`) | earliest later step after encode where the dest has no mask pixels; k is that delay | invisible_displacement | smaller → fewer items, dest still on screen | none found; dest must leave the mask before the query | 5–20 | `invisible_displacement.max_delay` |
| ID destination landmark gap | 0.5 m | `planner.py` (`_id_landmark_far_from_object`) | the landmark naming the destination (receptacle, or the landmark a floor spot is near) must be this far (AABB gap, XZ) from the object's old box | invisible_displacement | higher → fewer items, clearer moves | round-7 review: "Bread moved onto the floor near the Countertop" when it was on the countertop read as no move. Long Time No See names counter areas by nearby landmarks for the same reason; AI2-THOR Rearrangement (Weihs et al. 2021) treats 3D IoU ≥ 0.5 as the same place. Same value as `min_displacement_m` | 0.3–1.0 | `invisible_displacement.min_landmark_gap_m` |
| ID label change with the move | on | `planner.py` (`_id_label_changes_with_move`) | the old and new positions get different ego labels at the query step, each clear of `directions.label_margin_deg` | invisible_displacement | off → items the old location also answers; on → fewer items, far short moves dropped | derived: the answer must depend on the move; the old-location option measures the A-not-B (perseveration) error. `min_displacement_m` alone is not enough: 0.5 m at 4 m is about 7°, under the 10° margin | on | `invisible_displacement.require_label_change` |
| ID landmark in the answer's sector | on | `planner.py` (`_landmark_ego_label`) | the landmark named in "onto / near the Y" gets the same clear label (whole floor box) as the object at the query; swaps exempt | invisible_displacement | off → items whose named cue points to another direction | step-6 review 2026-11-25, item 31 (lamp right at 105°, plant behind at 146°; 1 of 21 items) | on/off | `invisible_displacement.anchor_same_sector` |
| rotation_since_last_seen | stored, not capped | `planner.py` (`_total_turn_deg`) | total turning since the tracked cue was last seen; reported in <180 / 180–360 / >360° groups (`rotation_metrics`) | invisible_displacement, spatial_updating | none (measured) | user decision 2026-11-25: cap only if 3-reviewer humans are near chance on high-rotation items | — | — |
| displacement candidate roles | chosen / original_location / nearby_receptacle / salient_decoy_location | `planner.py` | which locations become options when an episode has no `placement_candidates` | invisible_displacement | — | derived: collector teleport trials; pilots are built from `placement_candidates` (latest hide, receptacle parent = anchor) | — | (structural) |
| landmark salience | layout 2.0 + 0.1·sightings; others 1.0 + 0.05·sightings | `planner.py:1755, 1759` | landmark ranking for PT and class 4 | perspective_taking, route, survey | changes which landmarks are tried first | none found (05535fa, comment "Layout landmarks get a salience boost; more sightings → higher weight") | layout boost 1–3 | `landmarks.salience.*` |
| landmark caps (default) | top 5 per region, 40 total | `planner.py:1768-1769` | landmarks kept for class 4 | route, survey | more → more pairs tried | none found (05535fa) | 3–10 / 20–80 | `landmarks.top_n_per_region` / `max_total` |
| landmark caps (PT) | top 4 per region, 18 total | `planner.py:2496` | landmarks kept for PT triples | perspective_taking | more → more triples (cubic) | none found (05535fa) | 3–6 / 10–30 | `perspective_taking.top_n_per_region` / `max_total` |
| PT footprint grid | 3 | `perspective_taking.footprint_grid` | points per axis over A's (grown by `a_standoff_m`), B's and C's floor boxes; the answer must not flip for any combination | perspective_taking | higher → stricter (more triples dropped near large objects); 1 → centre only (old behaviour) | none found; round 1: the 4 reviewed items whose label flips scored 1/16 correct, the 6 stable ones 5/10 | 2–5 | `perspective_taking.footprint_grid` |
| PT pair gap | 0.5–5.0 m | `planner.py` (`_pt_pairs_well_spaced`) | AABB gap (XZ) between every pair of A, B, C; centre distance when a box is missing | perspective_taking | higher minimum → fewer items (round 7: 0.25 → 88/98, 0.5 → 84, 1.0 → 62) | literature: VSI-Bench relative direction (VLM-3R reimplementation of its generator) rejects triples with any pair outside 0.30–5.0 m; round-7 review found a 0.36 m pair confusing, so the minimum is 0.5 m (user decision 2026-11-18). ReVSI reportedly uses 1 m (not verified) | 0.3–1.0 / 4–8 | `perspective_taking.min_pair_gap_m`, `perspective_taking.max_pair_gap_m` |
| PT disparity bins | edges 45 / 90 / 135° | `planner.py` (`_pt_disparity_bin`, `_pt_spread`) | bins the angle between the imagined heading A→B and the camera heading; bin index is the item's `difficulty`; triples are drawn round-robin over (answer label, bin), emptiest group first | perspective_taking | finer bins → more groups, each filled more slowly | literature: Spatial Orientation Test (Hegarty & Waller 2004; Kozhevnikov & Hegarty 2001), error grows with angular disparity; results are analysed by disparity, not by pair distance | 30–60° wide | `perspective_taking.disparity_bin_edges_deg` |
| PT control share | 0.2 | `generate_items.py` (`_trim_control_share`) | max share of the final PT set in disparity bin 0 (imagined heading within 45° of the camera's), the control for the disparity effect | perspective_taking | lower → fewer easy items, weaker baseline | none found; reviewer feedback 2026-11-20 (keep the control, report headline accuracy on ≥ 45°) | 0.1–0.3 | `perspective_taking.max_control_share` |
| PT A standoff | 0.5 m | `planner.py` (`_a_standpoints`) | ring this wide around A's box, on the half facing B only, added to A's centre as stand points for the stability check ("standing next to the A, facing the B") | perspective_taking | higher → fewer items near A | none found; reviewer feedback 2026-11-20 | 0.3–1.0 | `perspective_taking.a_standoff_m` |
| PT stand-point margin | rejects | `planner.py` (`_pt_label_stable`) | a stand point whose angle falls inside `pt_boundary_margin_deg` of a sector edge drops the triple, like a flip | perspective_taking | false → more items, some ambiguous | step-6 review 2026-11-25: removed all 5 missed PT items (63 → 25) | true/false | `perspective_taking.stand_point_margin_rejects` |
| PT min A–C distance | 2.5 m | `planner.py` (`plan_perspective_taking`) | minimum A–C centre distance; the angle spread over the stand area falls with distance (0.5 m ring: ±14° at 2 m, ±7° at 4 m) | perspective_taking | higher → narrower spread, fewer items | geometric; user decision 2026-11-25 (Round 2: 25 → 24) | 2.0–3.0 | `perspective_taking.min_dist_a_c_m` |
| near-cardinal limit | 20° | `geometry.py` (`near_cardinal`) | PT centre angle, hidden-object SU bearing, the ID object's new position and the SWM net heading change must be this close to an axis | perspective_taking, spatial_updating, invisible_displacement, spatial_working_memory | lower → fewer items (±15°: no "right" PT answers, SU −29%) | step-6 review 2026-11-25: 4 of 4 "ahead" PT items at 19–23° missed; Round 2: PT 25 → 24 with distance, SU 1081 → 918 | 15–22.5 | `directions.near_cardinal_max_deg` |
| PT multi-frame | true | `planner.py` (`_pt_best_window`) | A, B, C may be seen in different frames of one window (shown whole, ≤ largest frame budget) instead of together in one frame | perspective_taking | true → more triples and labels other than "ahead" (one-frame co-visibility keeps C near A→B) | none found; user decision 2026-11-20, kept or dropped after the 15-house check | true / false | `perspective_taking.multi_frame` |
| PT frame continuity | 45° / 0.25 m | `planner.py` (`_pt_frames_continuous`) | max heading / position change between consecutive shown frames of a multi-frame item | perspective_taking | lower → only smooth windows | derived: one SPOC turn (45°) and move (0.2 m); user decision 2026-11-20. Round 2 walks never exceed it (FOV frames are consecutive steps) | 30–90° / 0.2–0.5 m | `perspective_taking.continuity_max_turn_deg`, `continuity_max_move_m` |
| PT window frames | 32 | `planner.py` (`_pt_best_window`) | max consecutive frames shown for a multi-frame PT item; longer windows are dropped; a window with two of A/B/C in one frame is preferred | perspective_taking | lower → fewer, easier-to-follow items (Round 2: 8 → 39 items, 32 → 63) | derived: largest `frames.budgets`; user tried 8 after the Round-2 review, then chose 32 (2026-11-20) | 4–16 | `perspective_taking.max_window_frames` |
| PT uses as A | 2 | `planner.py` (`_pt_spread`) | max items per scene with the same object as A | perspective_taking | lower → more varied A, fewer items | none found; reviewer feedback 2026-11-20 | 1–3 | `perspective_taking.max_uses_as_a` |

## 5. Class-4 pairs (route_knowledge, survey_based_route_planning)

| name | current value | file:line (all copies) | what it controls | constructs affected | effect of raising / lowering | justification | suggested ablation range | config key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| class-4 per reviewer | 5 | `build_review_package.py` | max route+survey study items each human reviewer sees. Default from yaml; override per run with `--class4-per-reviewer N` (also on `build_vlm_chat_kit.py` when not using `--review-package`) | review package | higher → more class-4 judgments per person | none found | 3–10 | `review.class4_per_reviewer` |
| model-only templates | `[recall_count]` | `build_review_package.py` (`concise_ok`) | templates never sampled for human review, scored on models only | review package | — | derived: 14 of 17 Round-2 count items answer 2 (always-2 baseline 82%), too few and too uniform for a human result (user decision 2026-11-27) | — | `review.model_only_templates` |
| image budgets | 16, 24, 32 | `planner.py` (`_budget_frame_lists`) | one frame list per budget for every multi-frame item: whole span if it fits, else required frames (encode/query, rotation ±1; class 4 sightings and survey path covers; ID/PT all frames) plus an even fill. Budget not offered if the required set or max gap is unmet; item dropped if no budget fits. `--frame_budgets` overrides for one run | SWM, SU, ID, route, survey | each budget = a VLM image limit; add or remove budgets for larger or smaller models | literature: VSI-Bench / MMSI-Bench evaluate the same question at a per-model frame count | 8–64 | `frames.budgets` |
| max frame gap | 5 frames | `planner.py` | max pool frames (repeated poses dropped) between consecutive shown frames of a sampled list; that budget is not offered if unmet | route, survey, route MC stretches (SWM/SU are never sampled) | lower → denser coverage, more drops | user decision 2026-11-24 for route MC (straight legs are forward steps; thinning removes repeated views, not answer information); none found otherwise | 3–8 | `frames.max_gap_steps` |
| route context margin | 4 steps | `planner.py` | FOV steps kept before the source sighting and after the goal sighting | route | higher → more approach and departure context | none found | 0–8 | `class4.route.context_margin_steps` |
| `MIN_PAIR_GEODESIC_M` / `MAX_PAIR_GEODESIC_M` | 2 m / 30 m | `planner.py` | absolute geodesic band for a source–goal pair | route, survey | higher min → longer routes, fewer pairs | none found for the 2 m floor (literature band was 1–30 m, HM3D-OVON / GOAT-Bench / HSSD-200) | min 1–3 | `class4.min_pair_geodesic_m` / `max_pair_geodesic_m` |
| `SCENE_PAIR_GEODESIC_PERCENTILE` | 0.5 | `planner.py` | survey length floor, and the detour-ratio floor for route and survey. Route length is `min_pair_geodesic_m` only | survey (length); route and survey (ratio) | higher → only the longest survey pairs, and a higher detour bar | none found (7a2ee6a) | 0.25–0.75 | `class4.scene_geodesic_percentile` |
| answer-label cap | 0.4 | `generate_items.py` after the round-robin | at most this share of a construct's selected items, over the whole set, may share one answer label. A label that appears once is kept | every MC construct | lower → more label diversity, possibly fewer items | none found | 0.25–0.5 | `options.max_label_share` |
| candidate pool | 24 | `planner.py` | facts gathered before that cut | every construct | larger → more choice for the cap | none found | 12–48 | `budget.candidate_pool` |
| delay steps | 2, 5, 10, 20 | `planner.py` | SWM and spatial-updating `k` is taken from this set when the episode has it; stored as `k` and `difficulty` | spatial_working_memory, spatial_updating | — | none found | — | `spatial_working_memory.delay_steps` / `spatial_updating.delay_steps` |
| `SURVEY_MIN_GEODESIC_EUCLIDEAN_RATIO` | 1.05 (cap) | `planner.py:708` | detour-ratio floor for survey = min(1.05, scene p50) | survey | higher → only detouring pairs | literature: HSSD-200 (CVPR 2024) supplement, geo/eucl < 1.05 is "nearly straight-line" | 1.0–1.2 | `class4.survey.max_ratio_floor` |
| `ROUTE_MAX_GEODESIC_EUCLIDEAN_RATIO_CAP` | 1.1 (cap) | `planner.py:709` | detour-ratio floor for route = min(1.1, scene p50) | route | higher → only detouring pairs | literature: Habitat PointNav episode generator rejects geo/eucl < 1.1 (comment removed in f5aa4da) | 1.0–1.3 | `class4.route.max_ratio_floor` |
| `ROUTE_LANDMARK_SNAP_M` | 2.5 m | `planner.py:714` (passed for route and survey) | landmark → nav-graph node snap radius | route, survey | larger → more landmarks become endpoints, looser "at the X" | none found (5c7eb3c, comment "slightly larger radius than the default 1.5 m so room-scale furniture near the trajectory still becomes an endpoint") | 1.0–3.0 | `class4.landmark_snap_m` |
| default landmark snap | max(1.5 m, 10 × grid) | `nav_graph.py:194`, **copy** `nav_graph.py:222` | default when no radius is passed | none in the pipeline (callers pass 2.5) | — | none found (63130d5) | — | `class4.landmark_snap_default_*` |
| `ROUTE_MIN_HOP_COUNT` | 2 | `planner.py:715` | minimum lattice hops on the walked path | route | higher → longer routes | none found (5b4cb02, comment "at least one real edge; length gate is geodesic metres") | 2–6 | `class4.route.min_hop_count` |
| `ROUTE_MAX_SUBPATH_NODES` | 512 | `planner.py:716` | safety cap on path length | route | — | none found (274e5ff, "lattice hops, not decisions; turns are the real cap") | — | `class4.route.max_subpath_nodes` |
| `ROUTE_MIN_TURN_COUNT` / `MAX` | 1 / 20 real turns | `planner.py` | route reference must contain this many direction changes | route | higher min → harder routes, fewer items | none found (274e5ff; at least one real turn). The max is 20 so a long recorded walk is kept | min 1–4, max 12–30 | `class4.route.min_turns` / `max_turns` |
| survey min turns | 1 real turn | `planner.py` | survey reference must contain this many direction changes | survey | higher min → fewer straight shortcuts | none found | 1–4 | `class4.survey.min_turns` |
| MC option shuffle seed | 0 | `templates.py` | per-item permutation of MC letters | every MC construct | a different seed reshuffles letters; the same seed is reproducible | none found | — | `options.shuffle_seed` |
| `SURVEY_MIN_AGENT_SOURCE_DIST_M` | 1.0 m | `planner.py:695` | agent must be this far from the source at the source frame | survey | higher → fewer survey items | literature: ObjectNav success distance 1.0 m (Batra et al. 2020; GOAT-Bench), as cited in the code. 5c7eb3c started at 2.0 m | 0.5–2.0 | `class4.survey.min_agent_source_dist_m` |
| `CLASS4_GOAL_SUCCESS_M` | 1.0 m | `nav_graph.py` scorer default | final xz vs the goal-sighting pose (route and survey). ObjectNav distance, applied to that pose rather than the landmark centroid | route, survey | larger → easier success | literature: ObjectNav / GOAT success distance 1.0 m | 0.5–2.0 | `class4.goal_success_m` |
| continuous search cell | 0.05 m | `nav_graph.py` `continuous_bfs_to_goals` | visited (x, z, heading) cell for the survey BFS | survey | finer → more states, same shortest action count when several paths exist | none found | — | `class4.search_state_quant_m` |
| move-block slack | 1e-6 m | `nav_graph.py` `calibrate_move_block_tol` | added to the worst recorded nearest-node distance; a landing farther than that sum from every reachable node is a blocked no-op | route, survey | larger → fewer moves blocked | derived: float slack so a pose exactly at the calibrated distance still passes | — | `class4.move_block_slack_m` |
| turn landmark distance | 1.0 m | `planner.py` (`_class4_route_mc`) | a class-4 turn is named by a static object whose floor box is this close to where the agent turned; else "walk ahead" | route MC, survey MC | larger → more named turns, looser "at the X" | literature: ObjectNav / GOAT success distance 1.0 m (Batra et al. 2020) | 0.5–1.5 | `class4.mc.turn_landmark_max_m` |
| turn merge distance | 1.0 m | `route_mc.turn_clusters`, `planner._walk_turn_boundaries` | turn runs closer than this along the walk are summed (two 45° corrections = one 90° turn) | route MC, survey MC | larger → fewer, bigger turns; at 2 m the Round 2 yield fell to 0 | none found; same reach as the landmark distance | 0.5–2.0 | `class4.mc.turn_merge_m` |
| turn labels | 90° / 135° / 180° | `route_mc.turn_label` | exact labels: below 90° a correction, 90° turn left/right, 135° turn sharply left/right, 180° turn around; no other net turn gets a label | route MC, survey MC | — | derived: SPOC turns in 45° steps (user decision 2026-11-24) | — | `class4.mc.turn_labels_deg` |
| route turns | 2–4 | `planner._plan_route_stretches`, `_class4_route_mc` | labelled turns per MC route, start turn included | route MC, survey MC | more → longer options | user decision 2026-11-24: two are needed for the 2×2 options; past four, options measure reading | 2–5 | `class4.mc.min_turns` / `max_turns` |
| view radius percentile | 0.5 | `nav_graph.py:315` (`calibrate_view_radius_m`) | radius around the trajectory whose edges count as "viewed" | survey | higher → more viewed edges, more survey pairs | none found (5b4cb02) | 0.25–0.75 | `class4.survey.view_radius_percentile` |
| trajectory snap tolerance | √2 · max(grid, move) | `nav_graph.py:131`; same formula as follower simplify default `nav_graph.py:911` | agent poses snapped to the graph; goal tolerance when replaying references | route, survey | larger → more poses kept, looser goal check | derived: one 8-neighbour step of the coarser lattice | 1–2 × step | `class4.snap_tolerance_steps` |
| reference tolerance ladder | [1.5·s, s, grid, grid/2], s = max(grid, move) | `planner.py:1625` | polyline simplification tried before the per-hop fallback | route, survey | — | derived: multiples of the step / grid size | — | `class4.follower.ladder_multipliers` |
| follower arrive radius | 0.75 · step | `nav_graph.py:926` | waypoint reached when closer than this | route, survey | larger → shorter references | none found (274e5ff) | 0.5–1.0 · step | `class4.follower.arrive_fraction` |
| follower `max_actions` | 512 | `nav_graph.py:878` | safety cap | route, survey | — | none found (274e5ff) | — | `class4.follower.max_actions` |
| graph-metadata fallbacks | grid 0.25 m, rotation 45°, move = grid | `nav_graph.py:73, 82, 129, 170, 190, 218, 546, 577, 778-779, 899-900, 904, 911, 1148-1149, 1153`; `planner.py:1620` | used only when the export lacks them | route, survey (only on incomplete exports) | — | derived: fallback; the collected data always supplies grid 0.15, move 0.20, rotation 45 | — | `class4.fallback.*` |

## 6. Episode-GT values that do not reach items

These are in the nav generator (`resolve_hyperparams_from_episode_meta`,
`visibility_filters.py:663-679`). They shape episode edges and the `route`
summary, but no planner reads them: edge `angle_relation` is only copied into
item `context` for traceability, and `episode['route']` is never read during
generation. They stay with the nav generator and are listed for completeness.

| name | value | file:line | note |
| --- | --- | --- | --- |
| `w`, `h`, `fov_v` | 396, 224, 59 | `visibility_filters.py:664-666`; **copy** `metrics_from_nav_row` defaults `visibility_filters.py:202` | overridden by `episode_meta.camera` (58.7° / 59.6° measured) |
| `mov_constant` | 0.2 | `visibility_filters.py:676` | overridden by `episode_meta.agent.movement_constant` |
| `min/med/max_distance` | 0.5 / 1.0 / 1.5 m | `visibility_filters.py:673-675` | edge `distance_label` only |
| `angle_threshold_xz` | 15° | `visibility_filters.py:672` | **passed to `get_direction_angle` and discarded** (`spatial_relations.py:84`) |
| `ey` | 0.1 m | `visibility_filters.py` | above / below in edge labels |
| `ez` | 0.15 m | `visibility_filters.py` | projection skipped closer than this; affects `local_position` / `angles` only |
| `k_neighbors`, `radius` | 3, 1.5 m | `visibility_filters.py:667-668` | object–object edges (allocentric, unsupported) |
| state sparsify eps | 1e-4 m | `ai2thor_nav_generator.py:50, 57` | derived: float noise; drops sub-0.1 mm pose changes from object_state_track |

---

## Values defined in more than one place

| value | copies | agree? |
| --- | --- | --- |
| encode-time visibility floors 100 / 8 / 40 | `DEFAULT_QUESTION_VISIBILITY_THRESHOLDS` (`visibility_filters.py:70-73`) and `ID_ENCODE_FOV_*` (`constructs.py:440-442`) | yes, but independent: editing one does not move the other |
| ahead half-width 20° | `AHEAD_HALF_WIDTH_FOV` (`constructs.py:232`) and literal default `get_direction_angle(ahead_half_width=20.0)` (`spatial_relations.py:74`) | yes |
| ahead half-width 45° | `AHEAD_HALF_WIDTH_FULL` (`constructs.py:234`) and literal default `angle_to_ego_label(ahead_half_width=45.0)` (`constructs.py:202`) | yes |
| pose-change tolerance 0.1 m / 5° | `net_pose_changed` (`constructs.py:737`) and `_net_pose_changed_between` (`planner.py:1452`) | yes |
| `max_per_construct` 3 | `generate_items.py:83`, `pipeline.py:28`, `planner.py:2682` | yes; the per-planner defaults (3/3/3/2/1/2/2/2) disagree but are always overridden by `plan_episode` |
| delay minimum 2 | `generate_items.py:87, 99`, `pipeline.py:29, 31`, `planner.py:501, 1466, 2684, 2686` | yes |
| landmark snap default max(1.5, 10·grid) | `nav_graph.py:194, 222` | yes (unused in the pipeline) |
| snap tolerance √2·max(grid, move) | `trajectory_snap_tolerance` (`nav_graph.py:131`) and follower simplify default (`nav_graph.py:911`) | yes (same formula written twice) |
| graph fallbacks grid 0.25 / rotation 45 | 15 places in `nav_graph.py`, `planner.py:1620` | yes |
| option count 4 | 7 places in `planner.py` / `templates.py` | yes |
| frame size 396 × 224 | `visibility_filters.py:202` and `:664-665` | yes |

## Values that look inconsistent with each other

1. **Two widths for "ahead of you" inside one item.** In spatial_updating
   the answer uses the 45° wedge, but the "non-updating" distractor (the
   pre-move bearing) uses the 20° wedge. An object at 30° is "ahead" in
   one frame of reference and "to your right" in the other, so part of the
   distractor's difference comes from the wedge width rather than from the
   move.
2. **Episode edge labels use 20° for every object**, including inferred edges
   to objects out of view, whose bearing can be anywhere in the full circle
   (where the planners use 45°). The labels are metadata only, but a reader
   comparing `context.angle_relation` with the answer will see disagreements.
3. **`angle_threshold_xz` (15°) is configured but ignored.**
4. **"Distinguishable" has a 12× range.** Queried objects and landmarks need
   1200 px² when no DecisionTree is loaded, displaced props need 100 px². With
   the DecisionTree loaded both use the tree, so the bar depends on whether a
   model file is present.
5. **Two definitions of "the agent moved".** spatial_updating uses 0.1 m / 5°.
   SWM, invisible_displacement and route use any floor-plane change > 1 mm
   (`_POSITION_EPS_M`).
6. **Several "near" radii:** survey agent-to-source 1.0 m, floor anchor 1.2 m,
   default landmark snap 1.5 m, route/survey landmark snap 2.5 m. Only the
   1.0 m one has a source.
7. **Detour-ratio caps differ by construct** (route 1.1, survey 1.05) and come
   from different sources (Habitat PointNav vs HSSD-200).
8. **Unreachable route check.** `plan_route_knowledge` rejects "no real move"
   only when `len(path) < 3` (`planner.py:2163`), but `ROUTE_MIN_HOP_COUNT = 2`
   already guarantees `len(path) >= 3`, so this check never fires.
9. **SWM count encoding frame.** The count uses only distinguishable
   sightings (`_count_category_seen`), but the encoding step is the first step
   where the category appears at all (`_first_step_with_category`), possibly a
   weak sighting.
10. **Destination anchors match by exact `obj-id`.** Type-stem matching
    (`CounterTop|9` ≈ any CounterTop) is not used. Confusable siblings are
    objects with mask pixels > 0 in a shown frame and a confusable
    `obj-synset`; the target is marked (or proximity) and only
    visibility-passing siblings receive a drawn badge.
11. **`build_route` label mismatch.** `get_distance_text` returns
    `"within reach"`, but `build_route` filters on `'within_reach'`, so route
    landmarks only ever include "nearby" objects. `episode['route']` is not
    read during generation, so items are unaffected.
12. **Camera FOV fallback 59°** vs measured 58.7° / 59.6°: harmless, because the
    measured value overrides it.
13. **VSI-Bench disambiguation setting needs room area**, which exported
    `world_layout.regions` do not carry. The 0.30 m branch (`margin_m_large`)
    is used for every scene.
