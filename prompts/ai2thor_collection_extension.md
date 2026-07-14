# AI2-THOR collection — additional fields

Keep existing navigation and object-catalog CSVs unchanged. Add the logs below for manipulation and layout episodes.

Collection CSVs may be **dense** (one row per timestep). Downstream cm-benchmark **sparsifies** `object_state` tracks (displaced objects), `region_trajectory`, and `passage_state` to change-points only; query step `t` uses the latest prior entry.

---

## A. Invisible displacement

### Episode requirement
Object is visible → becomes not visible → relocated while not visible → still not visible at the final frame.

### `object_state.csv` (one row per timestep × tracked object; **include hidden objects**)

| Column | Required | Description |
|--------|----------|-------------|
| `timestep` | yes | Aligns with navigation CSV |
| `obj-id` | yes | |
| `obj-type` | yes | |
| `obj-pos-x`, `obj-pos-y`, `obj-pos-z` | yes | True world position |
| `obj-rot-x`, `obj-rot-y`, `obj-rot-z` | yes | |
| `visible` | yes | Simulator visibility |
| `in_camera_fov` | yes | In nav camera detections |
| `parent_receptacle` | yes | Canonical parent id, or null |
| `parent_receptacles` | recommended | Full list (JSON/string) |
| `receptacle_is_open` | if applicable | Open/closed of container |
| `held_obj-id` on agent rows | recommended | What agent holds, if any |

Episode identity (`episode_id`, `scene_id`) lives in `episode_meta-*.json` / the run folder — do not repeat on every CSV row.

### `displacement_events.csv` (one row per relocation)

| Column | Required | Description |
|--------|----------|-------------|
| `event_id` | yes | Unique in episode |
| `obj-id` | yes | |
| `at_timestep` | yes | When relocation applied |
| `action` | yes | e.g. `PlaceObjectAtPoint`, `PutObject` |
| `from_receptacle` | yes | Before (or null) |
| `to_receptacle` | yes | After (or null) |
| `from_pos-x/y/z` | yes | |
| `to_pos-x/y/z` | yes | |
| `hidden_during` | yes | Must be `true` for these episodes |
| `visible_just_before` / `visible_just_after` | yes | |
| `in_fov_just_before` / `in_fov_just_after` | yes | |
| `moved_via` | recommended | `direct` \| `receptacle_moved` \| `other` |
| `notes` | optional | e.g. `same_receptacle_hidden_shift` |

Also log manipulation actions in the navigation CSV (`ag-action`, `action_success` if available).

---

## B. Survey / layout

### Episode requirement
Trajectory covers ≥2 regions. Landmarks A and B exist. A feasible A→B path was **not** walked. Optional: door opens to enable that path.

### `world_layout.json`

- `regions[]`: `region_id`, `label`, `center`, `landmark_obj_ids[]`
- `landmarks[]`: `landmark_id`, `obj-type`, `position`, `region_id`
- `passages[]`: `passage_id`, `obj-id`, `from_region`, `to_region`, `passage_type`
- `connectivity[]`: `from_region`, `to_region`, `passage_id`, `bidirectional`

### `passage_state.csv`

| Column | Required | Description |
|--------|----------|-------------|
| `timestep` | yes | |
| `passage_id` / `obj-id` | yes | |
| `is_open` | yes | |
| `from_region`, `to_region` | recommended | |

### `region_trajectory.csv`

`timestep`, `region_id`, `region_type` (agent’s region each step; dense OK).

### `episode_meta-*.json`

`episode_id`, `scene_id`, `episode_kind`, `environment`, counts (e.g. `num_displacements`).

---

## C. Files per run

| File | Invisible displacement | Survey |
|------|:----------------------:|:------:|
| `navigation-*.csv` + `objects-*.csv` + images | ✓ | ✓ |
| `object_state-*.csv` | ✓ | |
| `displacement_events-*.csv` | ✓ | |
| `world_layout-*.json` | | ✓ |
| `passage_state-*.csv` | | ✓ if doors used |
| `region_trajectory-*.csv` | | ✓ |
| `episode_meta-*.json` | ✓ | ✓ |

---

## D. Do not

- Invent semantic object “front” / facing from rotation.
- Drop `object_state` rows when `visible=false` or `in_camera_fov=false`.
- End an invisible-displacement episode with the object clearly visible at its final location.
- Rename existing navigation/object catalog columns without a migration note.
