# AI2-THOR collection — additional fields

Keep existing navigation and object-catalog CSVs unchanged. Add the logs below for manipulation and layout episodes.

---

## A. Invisible displacement

### Episode requirement
Object is visible → becomes not visible → relocated while not visible → still not visible at the final frame.

### `object_state.csv` (one row per timestep × tracked object; **include hidden objects**)

| Column | Required | Description |
|--------|----------|-------------|
| `episode_id` | yes | |
| `timestep` | yes | Aligns with navigation CSV |
| `obj-id` | yes | |
| `obj-type` | yes | |
| `obj-pos-x`, `obj-pos-y`, `obj-pos-z` | yes | True world position |
| `obj-rot-x`, `obj-rot-y`, `obj-rot-z` | yes | |
| `visible` | yes | Simulator visibility |
| `parent_receptacle` | yes | Canonical parent id, or null |
| `parent_receptacles` | recommended | Full list (JSON/string) |
| `receptacle_is_open` | if applicable | Open/closed of container |
| `held_obj-id` on agent rows | recommended | What agent holds, if any |

### `displacement_events.csv` (one row per relocation)

| Column | Required | Description |
|--------|----------|-------------|
| `episode_id` | yes | |
| `event_id` | yes | Unique in episode |
| `obj-id` | yes | |
| `at_timestep` | yes | When relocation applied |
| `action` | yes | e.g. `PutObject`, `PickupObject` |
| `from_receptacle` | yes | Before (or null) |
| `to_receptacle` | yes | After (or null) |
| `from_pos-x/y/z` | yes | |
| `to_pos-x/y/z` | yes | |
| `hidden_during` | yes | Must be `true` for these episodes |
| `visible_just_before` | yes | At `at_timestep - 1` |
| `visible_just_after` | yes | At `at_timestep` |
| `moved_via` | recommended | `direct` \| `receptacle_moved` \| `other` |

Also log manipulation actions in the navigation CSV (`ag-action`, success flag if available).

---

## B. Survey / layout (novel unwalked path)

### Episode requirement
Trajectory covers ≥2 regions. Landmarks A and B exist. A feasible A→B path exists that was **not** walked. Optional: door opens to enable that path.

### `world_layout.json` (per scene or episode)

- `scene_id`
- `regions[]`: `region_id`, `label`, `center` `{x,y,z}`, `landmark_obj_ids[]`
- `landmarks[]`: `landmark_id`, `obj-type`, `position` `{x,y,z}`, `region_id`
- `passages[]`: `passage_id`, `obj-id`, `from_region`, `to_region`, `passage_type`
- `connectivity[]`: `from_region`, `to_region`, `passage_id`, `bidirectional`

### `passage_state.csv` (per timestep for doors/openings; required if door-open items)

| Column | Required | Description |
|--------|----------|-------------|
| `episode_id` | yes | |
| `timestep` | yes | |
| `obj-id` / `passage_id` | yes | |
| `is_open` | yes | |
| `from_region`, `to_region` | recommended | |

### `passage_events.csv` (optional)

`at_timestep`, `obj-id`, `from_state`, `to_state`, `action` (e.g. `OpenObject`).

### `region_trajectory.csv` (recommended)

`timestep`, `region_id` (agent’s region each step).

Keep full agent poses/actions/paths in the existing navigation CSV.

---

## C. Files per run

| File | Invisible displacement | Survey |
|------|:----------------------:|:------:|
| Existing `navigation-*.csv` + `objects-*.csv` + images | ✓ | ✓ |
| `object_state.csv` | ✓ | |
| `displacement_events.csv` | ✓ | |
| `world_layout.json` | | ✓ |
| `passage_state.csv` | | ✓ if doors used |
| `passage_events.csv` | | recommended |
| `region_trajectory.csv` | | recommended |

Share one `episode_id` across all files for the run.

---

## D. Do not

- Invent semantic object “front” / facing from rotation.
- Drop `object_state` rows when `visible=false`.
- End an invisible-displacement episode with the object clearly visible at its final location.
- Rename existing navigation/object catalog columns.
