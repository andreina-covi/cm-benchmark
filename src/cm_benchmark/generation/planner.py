"""Task planner: select construct-eligible facts from episode GT (deterministic)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

from cm_benchmark.generation.constructs import (
    AHEAD_HALF_WIDTH_FOV,
    AHEAD_HALF_WIDTH_FULL,
    EGO_DIRECTION_OPTIONS,
    ID_ENCODE_FOV_MIN_BBOX_AREA,
    ID_ENCODE_FOV_MIN_SIDE,
    ID_ENCODE_FOV_MIN_VISIBLE_PIXELS,
    MIRRORED_LR,
    OPPOSITE,
    ORTHOGONAL,
    QUERY_FOV_MIN_BBOX_AREA,
    QUERY_FOV_MIN_SIDE,
    QUERY_FOV_MIN_VISIBLE_PIXELS,
    find_ego_edge,
    fov_metrics_ok,
    humanize_receptacle,
    imagined_perspective_label,
    net_pose_changed,
    object_type_from_id,
    resolve_referring_disambiguator,
    step_by_index,
    xyz_as_dict,
)
from cm_benchmark.generation.geometry import (
    agent_pose_at_step,
    ego_label_from_world_pose,
)


@dataclass
class PlannedFact:
    """Locked spatial fact for one draft item (before wording)."""

    construct: str
    status: str  # ok | thin | unsupported
    query_step: Optional[int] = None
    encoding_step: Optional[int] = None
    queried_object_id: Optional[str] = None
    reference_object_id: Optional[str] = None
    answer_label: Optional[str] = None
    answer_source: Optional[list[str]] = None
    image_paths: list[str] = None
    options_pool: list[str] = None
    distractor_seeds: list[str] = None
    displacement_event: Optional[dict] = None
    reason: Optional[str] = None
    extra: Optional[dict] = None

    def __post_init__(self):
        if self.image_paths is None:
            self.image_paths = []
        if self.options_pool is None:
            self.options_pool = []
        if self.distractor_seeds is None:
            self.distractor_seeds = []
        if self.extra is None:
            self.extra = {}


def _img(step: Optional[dict]) -> list[str]:
    if not step:
        return []
    p = step.get('image_path')
    return [p] if p else []


def merge_role_images(
    pairs: Sequence[tuple[Sequence[str], str]],
) -> tuple[list[str], list[str]]:
    """Dedup image paths in order; merge role labels when the same path repeats.

    ``pairs`` is ``[(paths, role), ...]``. Returns ``(paths, roles)`` aligned.
    """
    paths: list[str] = []
    roles: list[str] = []
    index: dict[str, int] = {}
    for imgs, role in pairs:
        label = str(role or '').strip() or 'view'
        for p in imgs:
            if not p:
                continue
            if p in index:
                i = index[p]
                if label not in roles[i]:
                    roles[i] = f'{roles[i]} + {label}'
            else:
                index[p] = len(paths)
                paths.append(p)
                roles.append(label)
    return paths, roles


def _images_between(episode: dict, start_step: int, end_step: int) -> list[str]:
    """Image paths for the inclusive step window, in navigation order.

    Intermediate frames that share the previous kept frame's horizontal agent
    position are dropped (rotate-in-place / look-only). Encoding and query
    endpoints are always kept when present.
    """
    window: list[tuple[int, str]] = []
    for step in episode.get('steps') or []:
        step_idx = int(step['step'])
        if start_step <= step_idx <= end_step and step.get('image_path'):
            window.append((step_idx, step['image_path']))
    if not window:
        return []

    start_i, end_i = int(start_step), int(end_step)
    kept: list[str] = []
    last_pos: Optional[tuple[float, float]] = None
    for step_idx, path in window:
        pos = _agent_horizontal_pos(episode, step_idx)
        is_endpoint = step_idx == start_i or step_idx == end_i
        if is_endpoint or last_pos is None or _horizontal_pos_differs(last_pos, pos):
            kept.append(path)
            if pos is not None:
                last_pos = pos
    return kept


def _delay_is_allowed(k: int, min_delay: int, max_delay: Optional[int]) -> bool:
    return k >= min_delay and (max_delay is None or k <= max_delay)


def _displaced_ids(episode: dict) -> set[str]:
    return {e.get('obj_id') for e in (episode.get('displacement_events') or []) if e.get('obj_id')}


def _mode_seed(mode: str, label: str) -> str:
    """Encode failure-mode + concrete label for templates._shuffle_options."""
    return f'{mode}::{label}'


def _ego_pool_with_diagnostics(correct: str, *extra_labels: str) -> tuple[list[str], list[str]]:
    """Build a 4-way ego direction pool from real labels + opposite/orthogonal."""
    pool: list[str] = []
    seeds: list[str] = []
    for lab in (correct, *extra_labels):
        if lab and lab not in pool:
            pool.append(lab)
    if correct in OPPOSITE:
        opp = OPPOSITE[correct]
        if opp not in pool:
            pool.append(opp)
        seeds.append('opposite_direction')
        seeds.append(_mode_seed('opposite_direction', opp))
    if correct in ORTHOGONAL:
        orth = ORTHOGONAL[correct]
        if orth not in pool:
            pool.append(orth)
        seeds.append('orthogonal_direction')
        seeds.append(_mode_seed('orthogonal_direction', orth))
    for filler in EGO_DIRECTION_OPTIONS:
        if len(pool) >= 4:
            break
        if filler not in pool:
            pool.append(filler)
    return pool[:4], seeds


# Horizontal translation epsilon (meters). Rotate/look-only steps stay below this.
_POSITION_EPS_M = 1e-3


def _agent_horizontal_pos(
    episode: dict, step_idx: int
) -> Optional[tuple[float, float]]:
    """Agent (x, z) at step — floor-plane position; y is height."""
    pos, _rot = agent_pose_at_step(episode, step_idx)
    if pos is None or len(pos) < 3:
        return None
    try:
        return (float(pos[0]), float(pos[2]))
    except (TypeError, ValueError):
        return None


def _horizontal_pos_differs(
    a: Optional[tuple[float, float]],
    b: Optional[tuple[float, float]],
    *,
    eps: float = _POSITION_EPS_M,
) -> bool:
    if a is None or b is None:
        return False
    return abs(a[0] - b[0]) > eps or abs(a[1] - b[1]) > eps


def _has_real_move_between(
    episode: dict, t0: int, t1: int, *, eps: float = _POSITION_EPS_M
) -> bool:
    """True if the agent translates on the floor plane between t0 and t1 (inclusive end).

    Action names alone are not enough: rotate/look/turn in place leave position
    unchanged and are not treated as navigation for this benchmark.
    """
    if int(t1) <= int(t0):
        return False
    p_start = _agent_horizontal_pos(episode, int(t0))
    if p_start is None:
        return False
    p_end = _agent_horizontal_pos(episode, int(t1))
    if _horizontal_pos_differs(p_start, p_end, eps=eps):
        return True
    for step in episode.get('steps') or []:
        si = int(step['step'])
        if int(t0) < si <= int(t1):
            p = _agent_horizontal_pos(episode, si)
            if _horizontal_pos_differs(p_start, p, eps=eps):
                return True
    return False


def _object_visible_before(episode: dict, obj_id: str, at_t: int) -> bool:
    """True if obj appears in visible_objects at some step < at_t."""
    for step in episode.get('steps') or []:
        if int(step['step']) < int(at_t) and obj_id in (step.get('visible_objects') or {}):
            return True
    track = (episode.get('object_state_track') or {}).get(obj_id) or {}
    for entry in track.get('entries') or []:
        t = entry.get('step', entry.get('timestep'))
        if t is None:
            continue
        if int(t) < int(at_t) and entry.get('visible') and entry.get('in_camera_fov'):
            return True
    return False


def _referring_disambiguator(
    episode: dict, step_idx: int, obj_id: str
) -> Optional[str]:
    """'' if unique; phrase if duplicate+landmark; None to skip candidate."""
    step = step_by_index(episode, int(step_idx))
    if step is None:
        return None
    return resolve_referring_disambiguator(
        step, obj_id, _agent_pose_dict(episode, int(step_idx))
    )


def _distinguishable_encoding_sighting(
    episode: dict,
    step_idx: int,
    obj_id: str,
    *,
    min_bbox_area: float = QUERY_FOV_MIN_BBOX_AREA,
    min_side: float = QUERY_FOV_MIN_SIDE,
    min_visible_pixels: float = QUERY_FOV_MIN_VISIBLE_PIXELS,
) -> bool:
    """Object must be clearly visible at encode (metrics + ego edge), not just listed."""
    step = step_by_index(episode, int(step_idx))
    if step is None:
        return False
    odata = (step.get('visible_objects') or {}).get(obj_id)
    if not fov_metrics_ok(
        odata,
        min_bbox_area=min_bbox_area,
        min_side=min_side,
        min_visible_pixels=min_visible_pixels,
    ):
        return False
    return find_ego_edge(step, obj_id) is not None


# Soft FOV kwargs for invisible-displacement prop encoding (not landmark QUERY bar).
_ID_ENCODE_FOV = dict(
    min_bbox_area=ID_ENCODE_FOV_MIN_BBOX_AREA,
    min_side=ID_ENCODE_FOV_MIN_SIDE,
    min_visible_pixels=ID_ENCODE_FOV_MIN_VISIBLE_PIXELS,
)


def _ego_label_at(
    episode: dict,
    step_idx: int,
    obj_id: str,
    *,
    obj_pos=None,
    ahead_half_width: float = AHEAD_HALF_WIDTH_FULL,
) -> Optional[str]:
    """Ego MC label from poses; pass FOV vs full-circle half-width per construct."""
    ag_pos, ag_rot = agent_pose_at_step(episode, int(step_idx))
    pos = obj_pos
    if pos is None:
        step = step_by_index(episode, int(step_idx))
        if step is not None:
            odata = (step.get('visible_objects') or {}).get(obj_id) or {}
            pos = odata.get('position')
            if pos is None:
                mem = (step.get('non_visible_objects') or {}).get(obj_id) or {}
                pos = mem.get('position') or (mem.get('last_known') or {}).get('position')
        if pos is None:
            pos = _object_world_pos(episode, obj_id)
    return ego_label_from_world_pose(
        ag_pos, ag_rot, pos, episode, ahead_half_width=ahead_half_width
    )


def plan_egocentric_encoding(episode: dict, max_items: int = 3) -> list[PlannedFact]:
    out = []
    for step in episode.get('steps') or []:
        step_idx = int(step['step'])
        visible = step.get('visible_objects') or {}
        for obj_id in visible:
            if not fov_metrics_ok(visible.get(obj_id)):
                continue
            edge = find_ego_edge(step, obj_id)
            if not edge:
                continue
            label = _ego_label_at(
                episode,
                step_idx,
                obj_id,
                obj_pos=visible[obj_id].get('position'),
                ahead_half_width=AHEAD_HALF_WIDTH_FOV,
            )
            if not label or label not in EGO_DIRECTION_OPTIONS:
                continue
            disambiguator = _referring_disambiguator(episode, step_idx, obj_id)
            if disambiguator is None:
                continue
            pool, seeds = _ego_pool_with_diagnostics(label)
            if len(pool) < 2 or label not in pool:
                continue
            out.append(
                PlannedFact(
                    construct='egocentric_encoding',
                    status='ok',
                    query_step=step_idx,
                    encoding_step=step_idx,
                    queried_object_id=obj_id,
                    answer_label=label,
                    answer_source=[
                        f"agent_pose@[{step_idx}] + visible_objects[{obj_id}].position "
                        f"(equal-wedge bearing)"
                    ],
                    image_paths=_img(step),
                    options_pool=pool,
                    distractor_seeds=seeds,
                    extra={
                        'object_type': object_type_from_id(obj_id, visible),
                        'angle_relation': edge.get('angle_relation'),
                        'frame_of_reference': 'egocentric',
                        'disambiguator': disambiguator,
                    },
                )
            )
            if len(out) >= max_items:
                return out
    return out


def _count_category_seen(episode: dict, category: str, up_to_step: int) -> int:
    """Count distinct object ids of ``category`` visible on any step ≤ up_to_step."""
    seen: set[str] = set()
    for step in episode.get('steps') or []:
        if int(step['step']) > int(up_to_step):
            continue
        for oid, odata in (step.get('visible_objects') or {}).items():
            if object_type_from_id(oid, {oid: odata}) == category:
                seen.add(oid)
    return len(seen)


def plan_spatial_working_memory(
    episode: dict,
    max_items: int = 3,
    *,
    min_delay: int = 2,
    max_delay: Optional[int] = None,
) -> list[PlannedFact]:
    """Recall past relation under delay k; optional count/load mode.

    Encoding step is the latest *distinguishable* prior sighting (metrics + ego
    edge), not ``last_seen_step`` alone (which may be a weak FOV scrape).
    """
    if min_delay < 1:
        raise ValueError('SWM min_delay must be at least 1')
    if max_delay is not None and max_delay < min_delay:
        raise ValueError('SWM max_delay must be greater than or equal to min_delay')

    displaced = _displaced_ids(episode)
    out: list[PlannedFact] = []
    # Taxonomy lists both relation and count. Relation used to fill ``max_items``
    # and return early, so recall_count never ran. Reserve count slots when
    # the budget allows more than one item.
    count_budget = 0 if max_items < 2 else max(1, max_items // 3)
    relation_budget = max_items - count_budget

    for step in episode.get('steps') or []:
        if len(out) >= relation_budget:
            break
        step_idx = int(step['step'])
        non_vis = step.get('non_visible_objects') or {}
        for obj_id, mem in non_vis.items():
            if len(out) >= relation_budget:
                break
            if obj_id in displaced:
                continue
            # Encode at last distinguishable sighting, not last FOV scrape.
            enc_idx = _last_distinguishable_sighting(episode, obj_id, step_idx)
            if enc_idx is None or step_idx <= enc_idx:
                continue
            k = step_idx - enc_idx
            if not _delay_is_allowed(k, min_delay, max_delay):
                continue
            # Rotate-in-place delay is not navigation for this benchmark.
            if not _has_real_move_between(episode, enc_idx, step_idx):
                continue
            enc = step_by_index(episode, enc_idx)
            if enc is None:
                continue
            # Answer from equal-wedge bearing at encoding pose (not stored triple)
            edge = find_ego_edge(enc, obj_id)
            if not edge:
                continue
            enc_vis = (enc.get('visible_objects') or {}).get(obj_id) or {}
            label = _ego_label_at(
                episode,
                enc_idx,
                obj_id,
                obj_pos=enc_vis.get('position'),
                ahead_half_width=AHEAD_HALF_WIDTH_FOV,
            )
            if not label or label not in EGO_DIRECTION_OPTIONS:
                continue

            # Diagnostic: current-view answer if something else is at that bearing now
            current_view_lab = None
            for oid, odata in (step.get('visible_objects') or {}).items():
                if not find_ego_edge(step, oid):
                    continue
                lab = _ego_label_at(
                    episode,
                    step_idx,
                    oid,
                    obj_pos=odata.get('position'),
                    ahead_half_width=AHEAD_HALF_WIDTH_FOV,
                )
                if lab and lab != label:
                    current_view_lab = lab
                    break

            pool, seeds = _ego_pool_with_diagnostics(
                label, current_view_lab or ''
            )
            if current_view_lab and current_view_lab != label:
                seeds.append(_mode_seed('current_view_answer', current_view_lab))
                seeds.append('current_view_answer')

            if len(pool) < 2:
                continue

            disambiguator = _referring_disambiguator(episode, enc_idx, obj_id)
            if disambiguator is None:
                continue

            out.append(
                PlannedFact(
                    construct='spatial_working_memory',
                    status='ok',
                    query_step=step_idx,
                    encoding_step=enc_idx,
                    queried_object_id=obj_id,
                    answer_label=label,
                    answer_source=[
                        f"agent_pose@[{enc_idx}] + object_position "
                        f"(equal-wedge bearing; recalled)"
                    ],
                    image_paths=_images_between(episode, enc_idx, step_idx),
                    options_pool=pool,
                    distractor_seeds=seeds,
                    extra={
                        'object_type': object_type_from_id(obj_id, {obj_id: mem}),
                        'angle_relation': edge.get('angle_relation'),
                        'k': k,
                        'template_mode': 'recall_relation',
                        'frame_of_reference': 'egocentric',
                        'disambiguator': disambiguator,
                    },
                )
            )

    # Count / load mode (taxonomy: "How many {objects} have you seen so far?").
    # Uses leftover budget, including slots reserved above when max_items >= 2.
    if len(out) < max_items:
        seen_cats: set[str] = set()
        for step in episode.get('steps') or []:
            if len(out) >= max_items:
                break
            step_idx = int(step['step'])
            if step_idx < min_delay:
                continue
            categories = {
                object_type_from_id(oid, {oid: odata})
                for s in episode.get('steps') or []
                if int(s['step']) <= step_idx
                for oid, odata in (s.get('visible_objects') or {}).items()
            }
            # Prefer multi-instance categories (load); skip trivial n=1.
            ranked = sorted(
                (
                    (_count_category_seen(episode, cat, step_idx), cat)
                    for cat in categories
                ),
                key=lambda pair: (-pair[0], pair[1]),
            )
            for n, cat in ranked:
                if len(out) >= max_items:
                    break
                if n < 2 or cat in seen_cats:
                    continue
                answer = str(n)
                off1 = str(max(0, n - 1))
                off2 = str(n + 1)
                pool: list[str] = []
                for p in (answer, off1, off2, str(n + 2)):
                    if p not in pool:
                        pool.append(p)
                if len(pool) < 2:
                    continue
                seeds = [
                    'off_by_one_count',
                    _mode_seed('off_by_one_count', off1),
                    _mode_seed('off_by_one_count', off2),
                ]
                enc_idx = 0
                for s in episode.get('steps') or []:
                    for oid, odata in (s.get('visible_objects') or {}).items():
                        if object_type_from_id(oid, {oid: odata}) == cat:
                            enc_idx = int(s['step'])
                            break
                    else:
                        continue
                    break
                k = step_idx - enc_idx
                if not _delay_is_allowed(k, min_delay, max_delay):
                    continue
                if not _has_real_move_between(episode, enc_idx, step_idx):
                    continue
                out.append(
                    PlannedFact(
                        construct='spatial_working_memory',
                        status='ok',
                        query_step=step_idx,
                        encoding_step=enc_idx,
                        queried_object_id=None,
                        answer_label=answer,
                        answer_source=[
                            f'count[{cat}] over steps[0..{step_idx}].visible_objects'
                        ],
                        image_paths=_images_between(episode, enc_idx, step_idx),
                        options_pool=pool[:4],
                        distractor_seeds=seeds,
                        extra={
                            'object_type': cat,
                            'object_category': cat,
                            'k': k,
                            'template_mode': 'recall_count',
                            'frame_of_reference': 'allocentric',
                            'load_n_objects': n,
                        },
                    )
                )
                seen_cats.add(cat)
                break  # one count item per query step
    return out


def _is_floor_receptacle(receptacle_id: Optional[str]) -> bool:
    if not receptacle_id:
        return True
    stem = str(receptacle_id).split('|')[0].strip().lower()
    return stem in ('floor', 'wall', 'ceiling', 'room')


# Floor destinations need a nearby distinguishable landmark for the question cue.
FLOOR_ANCHOR_RADIUS_M = 1.2

# Survey direction/distance must stay allocentric: if the agent stands at the
# source landmark, "goal relative to source" collapses to "goal relative to me".
SURVEY_MIN_AGENT_SOURCE_DIST_M = 2.0

# Route-knowledge: snap landmarks onto the walked path with a slightly larger
# radius than the default 1.5 m so room-scale furniture near the trajectory
# still becomes an endpoint; MC sequence length caps stay explicit (not R2R).
ROUTE_LANDMARK_SNAP_M = 2.5
ROUTE_MAX_SUBPATH_NODES = 32
ROUTE_MAX_TURN_ARROWS = 16


def _agent_near_landmark(
    episode: dict,
    landmark_pos,
    steps: Sequence[Optional[int]],
    *,
    min_dist_m: float = SURVEY_MIN_AGENT_SOURCE_DIST_M,
) -> bool:
    """True if agent XZ is within ``min_dist_m`` of the landmark at any step."""
    sp = xyz_as_dict(landmark_pos)
    if sp is None:
        return False
    for t in steps:
        if t is None:
            continue
        pos, _ = agent_pose_at_step(episode, int(t))
        ap = xyz_as_dict(pos)
        if ap is None:
            continue
        if math.hypot(sp['x'] - ap['x'], sp['z'] - ap['z']) < float(min_dist_m):
            return True
    return False


def _xz_from_any(pos) -> Optional[tuple[float, float]]:
    d = xyz_as_dict(pos)
    if d is None:
        return None
    return (d['x'], d['z'])


def _nearest_floor_anchor(
    episode: dict,
    final_pos,
    frame_steps: list[int],
    *,
    radius_m: float = FLOOR_ANCHOR_RADIUS_M,
) -> Optional[tuple[str, str]]:
    """Nearest distinguishable landmark within radius of a Floor destination.

    Returns ``(landmark_id, display_name)`` or None (reject that Floor candidate).
    """
    target = _xz_from_any(final_pos)
    if target is None:
        return None
    best = None
    best_d = float('inf')
    for si in frame_steps:
        step = step_by_index(episode, si)
        if not step:
            continue
        for oid, odata in (step.get('visible_objects') or {}).items():
            if _is_floor_receptacle(oid):
                continue
            pos = odata.get('position')
            xz = _xz_from_any(pos)
            if xz is None:
                continue
            d = ((xz[0] - target[0]) ** 2 + (xz[1] - target[1]) ** 2) ** 0.5
            if d <= radius_m and d < best_d:
                best_d = d
                best = (oid, object_type_from_id(oid, {oid: odata}))
    return best


def _relation_shift_magnitude(
    episode: dict, from_pos, to_pos, query_step: int
) -> Optional[str]:
    """same_side | flipped_side from left/right at query pose (difficulty axis)."""
    ag_pos, ag_rot = agent_pose_at_step(episode, query_step)
    if ag_pos is None or ag_rot is None:
        return None
    pre = ego_label_from_world_pose(
        ag_pos, ag_rot, from_pos, episode, ahead_half_width=AHEAD_HALF_WIDTH_FULL
    )
    post = ego_label_from_world_pose(
        ag_pos, ag_rot, to_pos, episode, ahead_half_width=AHEAD_HALF_WIDTH_FULL
    )
    if not pre or not post:
        return None
    left_right = {'to your left', 'to your right'}
    if pre not in left_right or post not in left_right:
        return 'same_side' if pre == post else 'flipped_side'
    if pre == post:
        return 'same_side'
    if {pre, post} == left_right:
        return 'flipped_side'
    return 'same_side'


def _object_in_fov_at_step(episode: dict, obj_id: str, step_idx: int) -> bool:
    step = step_by_index(episode, step_idx)
    if step and obj_id in (step.get('visible_objects') or {}):
        return True
    track = (episode.get('object_state_track') or {}).get(obj_id) or {}
    entries = track.get('entries') or []
    chosen = None
    for entry in entries:
        t = entry.get('step', entry.get('timestep'))
        if t is None:
            continue
        if int(t) <= int(step_idx):
            chosen = entry
        else:
            break
    if chosen is None:
        return False
    return bool(chosen.get('in_camera_fov') or chosen.get('visible'))


def _object_hidden_through(
    episode: dict, obj_id: str, start_step: int, end_step: int
) -> bool:
    for step in episode.get('steps') or []:
        si = int(step['step'])
        if start_step <= si <= end_step and obj_id in (step.get('visible_objects') or {}):
            return False
    track = (episode.get('object_state_track') or {}).get(obj_id) or {}
    for entry in track.get('entries') or []:
        t = entry.get('step', entry.get('timestep'))
        if t is None:
            continue
        if start_step <= int(t) <= end_step and (
            entry.get('in_camera_fov') or entry.get('visible')
        ):
            return False
    return True


def _last_distinguishable_sighting(
    episode: dict,
    obj_id: str,
    before_step: int,
    *,
    min_bbox_area: float = QUERY_FOV_MIN_BBOX_AREA,
    min_side: float = QUERY_FOV_MIN_SIDE,
    min_visible_pixels: float = QUERY_FOV_MIN_VISIBLE_PIXELS,
) -> Optional[int]:
    """Latest step < before_step where the object is FOV-distinguishable.

    Uses ``_distinguishable_encoding_sighting`` (size metrics + ego edge), not
    merely membership in ``visible_objects`` / ``last_seen_step`` (which can be
    a weak scrape of a large mesh). Pass softer thresholds for ID props.
    """
    best = None
    for step in episode.get('steps') or []:
        si = int(step['step'])
        if si >= int(before_step):
            break
        if _distinguishable_encoding_sighting(
            episode,
            si,
            obj_id,
            min_bbox_area=min_bbox_area,
            min_side=min_side,
            min_visible_pixels=min_visible_pixels,
        ):
            best = si
    return best


def _landmark_matches(visible_id: str, landmark_id: str) -> bool:
    if visible_id == landmark_id:
        return True
    return str(visible_id).split('|')[0] == str(landmark_id).split('|')[0]


def _landmark_distinguishable_in_frames(
    episode: dict, landmark_id: Optional[str], frame_steps: list[int]
) -> bool:
    if not landmark_id or _is_floor_receptacle(landmark_id):
        return False
    for si in frame_steps:
        step = step_by_index(episode, si)
        if not step:
            continue
        for oid in (step.get('visible_objects') or {}):
            if _landmark_matches(oid, landmark_id):
                return True
    return False


def _candidates_for_event(episode: dict, event_id, obj_id: str) -> list[dict]:
    rows = []
    for row in episode.get('displacement_candidates') or []:
        if row.get('event_id') == event_id and row.get('obj_id') == obj_id:
            rows.append(row)
    return rows


def _option_specs_for_event(ev: dict, candidates: list[dict]) -> list[dict]:
    """Build diagnostic option specs: chosen + A-not-B + nearby + decoy."""
    by_role = {c.get('candidate_role'): c for c in candidates if c.get('candidate_role')}
    specs = []
    chosen = by_role.get('chosen')
    chosen_pos = (
        chosen.get('candidate_position')
        if chosen and chosen.get('candidate_position') is not None
        else ev.get('to_position')
    )
    chosen_rec = (
        (chosen.get('candidate_receptacle') if chosen else None)
        or ev.get('to_receptacle')
    )
    specs.append(
        {
            'role': 'chosen',
            'receptacle': chosen_rec,
            'position': chosen_pos,
            'is_answer': True,
        }
    )
    from_r = ev.get('from_receptacle')
    from_pos = ev.get('from_position')
    if from_pos is not None or (from_r and from_r != chosen_rec):
        specs.append(
            {
                'role': 'original_location',
                'receptacle': from_r,
                'position': from_pos,
                'is_answer': False,
            }
        )
    for role in ('nearby_receptacle', 'salient_decoy_location'):
        row = by_role.get(role)
        if not row:
            continue
        specs.append(
            {
                'role': role,
                'receptacle': row.get('candidate_receptacle'),
                'position': row.get('candidate_position'),
                'is_answer': False,
            }
        )
    return specs


def _pick_query_step(episode: dict, obj_id: str, at_t: int) -> Optional[int]:
    """Latest step ≥ at_t where object stays hidden from at_t through that step."""
    steps = _hidden_query_candidates(episode, obj_id, at_t)
    return steps[-1] if steps else None


def _hidden_query_candidates(episode: dict, obj_id: str, at_t: int) -> list[int]:
    """All steps ≥ at_t where object stays hidden from at_t through that step."""
    out: list[int] = []
    for step in episode.get('steps') or []:
        si = int(step['step'])
        if si < int(at_t):
            continue
        if _object_hidden_through(episode, obj_id, int(at_t), si):
            out.append(si)
    return out


def _ego_pool_from_specs(
    episode: dict, specs: list[dict], query_step: int
) -> tuple[Optional[str], list[str], list[str]]:
    """Map option specs to unique ego labels at query_step.

    Answer is kept first. Distractors that share a label already in the pool
    are skipped (not fatal) so nearby trial-teleports that collapse to the same
    cardinal direction do not kill the item.
    """
    ag_pos, ag_rot = agent_pose_at_step(episode, query_step)
    answer_dir: Optional[str] = None
    dir_pool: list[str] = []
    dir_seeds: list[str] = []
    labels_seen: set[str] = set()

    # Process answer first so it wins collisions with distractors.
    ordered = sorted(specs, key=lambda s: (0 if s.get('is_answer') else 1))
    for spec in ordered:
        if spec.get('position') is None:
            continue
        lab = ego_label_from_world_pose(
            ag_pos,
            ag_rot,
            spec['position'],
            episode,
            ahead_half_width=AHEAD_HALF_WIDTH_FULL,
        )
        if not lab:
            continue
        if lab in labels_seen:
            continue
        labels_seen.add(lab)
        dir_pool.append(lab)
        if spec.get('is_answer'):
            answer_dir = lab
        else:
            dir_seeds.append(spec['role'])
            dir_seeds.append(_mode_seed(spec['role'], lab))
    return answer_dir, dir_pool, dir_seeds


def _pick_best_id_query_step(
    episode: dict, obj_id: str, at_t: int, specs: list[dict]
) -> Optional[int]:
    """Prefer a hidden query step where candidate poses yield ≥2 unique ego labels.

    Score: more unique labels first, then later step (more delay) among ties.
    """
    best: Optional[int] = None
    best_key = (-1, -1)
    for si in _hidden_query_candidates(episode, obj_id, at_t):
        answer, pool, _seeds = _ego_pool_from_specs(episode, specs, si)
        if not answer or answer not in pool or len(pool) < 2:
            continue
        key = (len(pool), si)
        if key > best_key:
            best_key = key
            best = si
    return best


def _is_swap_event(ev: dict) -> bool:
    via = str(ev.get('moved_via') or '').lower()
    if via == 'swap':
        return True
    notes = str(ev.get('notes') or '').lower()
    if 'object_swap' in notes:
        return True
    return bool(ev.get('swap_partner_id'))


def _partner_event(episode: dict, ev: dict) -> Optional[dict]:
    partner_id = ev.get('swap_partner_id')
    event_id = ev.get('event_id')
    if not partner_id or not event_id:
        return None
    for row in episode.get('displacement_events') or []:
        if row.get('event_id') == event_id and row.get('obj_id') == partner_id:
            return row
    return None


def _object_distinguishable_in_frames(
    episode: dict, obj_id: str, frame_steps: list[int]
) -> bool:
    for si in frame_steps:
        step = step_by_index(episode, si)
        if step and obj_id in (step.get('visible_objects') or {}):
            return True
    return False


def _build_id_frame_steps(
    enc_idx: int, at_t: int, query_step: int, *, extra_steps: Optional[list[int]] = None
) -> list[int]:
    frame_steps = [enc_idx]
    if int(at_t) - 1 > enc_idx:
        frame_steps.append(int(at_t) - 1)
    for si in extra_steps or []:
        if si not in frame_steps:
            frame_steps.append(int(si))
    if query_step not in frame_steps:
        frame_steps.append(query_step)
    return sorted(frame_steps)


def _images_for_steps(episode: dict, frame_steps: list[int]) -> list[str]:
    images: list[str] = []
    for si in frame_steps:
        images.extend(_img(step_by_index(episode, si)))
    return [p for p in images if p]


def _try_ego_direction_fact(
    episode: dict,
    ev: dict,
    *,
    obj_id: str,
    object_type: str,
    enc_idx: int,
    query_step: int,
    at_t: int,
    images: list[str],
    specs: list[dict],
    template_mode: str,
    answer_source: list[str],
    extra_fields: dict,
) -> Optional[PlannedFact]:
    disambiguator = _referring_disambiguator(episode, enc_idx, obj_id)
    if disambiguator is None:
        return None
    answer_dir, dir_pool, dir_seeds = _ego_pool_from_specs(episode, specs, query_step)
    if not answer_dir or answer_dir not in dir_pool or len(dir_pool) < 2:
        return None
    return PlannedFact(
        construct='invisible_displacement',
        status='ok',
        query_step=query_step,
        encoding_step=enc_idx,
        queried_object_id=obj_id,
        answer_label=answer_dir,
        answer_source=answer_source,
        image_paths=images,
        options_pool=dir_pool[:4],
        distractor_seeds=dir_seeds,
        displacement_event=ev,
        extra={
            'object_type': object_type,
            'template_mode': template_mode,
            'frame_of_reference': 'egocentric',
            'k': max(1, int(query_step) - int(at_t)),
            'disambiguator': disambiguator,
            **extra_fields,
        },
    )


def plan_invisible_displacement(episode: dict, max_items: int = 3) -> list[PlannedFact]:
    """One item per displacement_events row; direct (recall_direction) or swap modes.

    Query step is chosen among hidden steps so candidate ego bearings stay unique;
    colliding distractor labels are dropped rather than rejecting the event.
    """
    events = episode.get('displacement_events') or []
    if not events:
        return []

    out: list[PlannedFact] = []
    for ev in events:
        if len(out) >= max_items:
            break
        if not ev.get('hidden_during', False):
            continue
        obj_id = ev.get('obj_id')
        to_r = ev.get('to_receptacle')
        at_t = ev.get('at_timestep')
        if not obj_id or at_t is None:
            continue
        enc_idx = _last_distinguishable_sighting(
            episode, obj_id, int(at_t), **_ID_ENCODE_FOV
        )
        if enc_idx is None:
            continue
        if not _distinguishable_encoding_sighting(
            episode, enc_idx, obj_id, **_ID_ENCODE_FOV
        ):
            continue
        if not _object_visible_before(episode, obj_id, int(at_t)):
            continue

        object_type = object_type_from_id(obj_id)
        candidates = _candidates_for_event(episode, ev.get('event_id'), obj_id)
        specs = _option_specs_for_event(ev, candidates)
        # Prefer a pose where distractor bearings stay unique (latest-only often collapses).
        query_step = _pick_best_id_query_step(episode, obj_id, int(at_t), specs)
        if query_step is None:
            continue
        if not _object_hidden_through(episode, obj_id, int(at_t), int(query_step)):
            continue
        # Encoding→query must include a floor-plane translation (not rotate-only).
        if not _has_real_move_between(episode, enc_idx, int(query_step)):
            continue

        if _is_swap_event(ev):
            partner_id = ev.get('swap_partner_id')
            if not partner_id or not _partner_event(episode, ev):
                continue
            partner_enc = _last_distinguishable_sighting(
                episode, partner_id, int(at_t), **_ID_ENCODE_FOV
            )
            if partner_enc is None:
                continue
            if not _object_visible_before(episode, partner_id, int(at_t)):
                continue
            frame_steps = _build_id_frame_steps(
                enc_idx, int(at_t), int(query_step), extra_steps=[partner_enc]
            )
            if not _object_distinguishable_in_frames(
                episode, partner_id, frame_steps
            ):
                continue
            images = _images_for_steps(episode, frame_steps)
            if not images:
                continue
            partner_type = object_type_from_id(partner_id)
            fact = _try_ego_direction_fact(
                episode,
                ev,
                obj_id=obj_id,
                object_type=object_type,
                enc_idx=enc_idx,
                query_step=query_step,
                at_t=int(at_t),
                images=images,
                specs=specs,
                template_mode='swap',
                answer_source=[
                    f"displacement_events[obj_id={obj_id}].to_position",
                    f"displacement_events[swap_partner_id={partner_id}].from_position",
                    f"agent_trajectory[{query_step}]",
                ],
                extra_fields={
                    'other_object_type': partner_type,
                    'swap_partner_id': partner_id,
                    'from_receptacle': ev.get('from_receptacle'),
                    'to_receptacle': to_r,
                },
            )
            if fact is not None:
                out.append(fact)
            continue

        # Direct hidden place onto a receptacle landmark (or Floor + nearby anchor)
        frame_steps = _build_id_frame_steps(enc_idx, int(at_t), int(query_step))
        floor_anchor_name = None
        floor_anchor_id = None
        if _is_floor_receptacle(to_r):
            anchor = _nearest_floor_anchor(
                episode, ev.get('to_position'), frame_steps
            )
            if anchor is None:
                continue
            floor_anchor_id, floor_anchor_name = anchor
            loc = humanize_receptacle(to_r, floor_anchor_landmark=floor_anchor_name)
            if loc is None:
                continue
            new_location = loc
        else:
            if not _landmark_distinguishable_in_frames(episode, to_r, frame_steps):
                continue
            loc = humanize_receptacle(to_r)
            if loc is None:
                continue
            new_location = loc.replace('on/in the ', '')
        images = _images_for_steps(episode, frame_steps)
        if not images:
            continue

        shift = _relation_shift_magnitude(
            episode, ev.get('from_position'), ev.get('to_position'), int(query_step)
        )

        # Direct moves: name the destination cue, ask ego bearing.
        fact = _try_ego_direction_fact(
            episode,
            ev,
            obj_id=obj_id,
            object_type=object_type,
            enc_idx=enc_idx,
            query_step=query_step,
            at_t=int(at_t),
            images=images,
            specs=specs,
            template_mode='recall_direction',
            answer_source=[
                f"displacement_events[obj_id={obj_id}].to_position",
                f"displacement_candidates[event_id={ev.get('event_id')}].chosen",
                f"agent_trajectory[{query_step}]",
            ],
            extra_fields={
                'new_location': new_location,
                'to_receptacle': to_r,
                'from_receptacle': ev.get('from_receptacle'),
                'floor_anchor_id': floor_anchor_id,
                'floor_anchor_landmark': floor_anchor_name,
                'relation_shift_magnitude': shift,
            },
        )
        if fact is not None:
            out.append(fact)

    if not out and events:
        return [
            PlannedFact(
                construct='invisible_displacement',
                status='unsupported',
                reason=(
                    'no_event_with_distinguishable_hidden_move_and_unique_options'
                ),
            )
        ]
    return out


def _agent_pose_dict(episode: dict, step_idx: int) -> Optional[dict]:
    pos, rot = agent_pose_at_step(episode, step_idx)
    if pos is None or rot is None:
        return None
    d = xyz_as_dict(pos)
    if d is None:
        return None
    try:
        heading = float(rot[1] if len(rot) > 1 else rot[0])
    except (TypeError, ValueError, IndexError):
        return None
    return {'x': d['x'], 'y': d['y'], 'z': d['z'], 'heading': heading}


def _object_static_between(
    episode: dict, obj_id: str, t0: int, t1: int, *, eps: float = 0.05
) -> bool:
    """Confirm object position unchanged via object_state_track (when present)."""
    track = (episode.get('object_state_track') or {}).get(obj_id) or {}
    entries = track.get('entries') or []
    if not entries:
        # No track and not in displacement_events → treat as static
        return obj_id not in _displaced_ids(episode)
    positions = []
    for entry in entries:
        t = entry.get('step', entry.get('timestep'))
        if t is None:
            continue
        if int(t0) <= int(t) <= int(t1):
            xz = _xz_from_any(entry.get('position'))
            if xz is not None:
                positions.append(xz)
    if len(positions) < 2:
        return True
    x0, z0 = positions[0]
    for x, z in positions[1:]:
        if abs(x - x0) > eps or abs(z - z0) > eps:
            return False
    return True


def _net_pose_changed_between(
    episode: dict, t0: int, t1: int, *, pos_tol: float = 0.1, heading_tol_deg: float = 5.0
) -> bool:
    """True if encode→query has real position OR heading change (not action count)."""
    a = _agent_pose_dict(episode, int(t0))
    b = _agent_pose_dict(episode, int(t1))
    if a is None or b is None:
        return False
    return net_pose_changed(a, b, pos_tol=pos_tol, heading_tol_deg=heading_tol_deg)


def plan_spatial_updating(
    episode: dict,
    max_items: int = 2,
    *,
    min_delay: int = 2,
    max_delay: Optional[int] = None,
) -> list[PlannedFact]:
    """Bearing at NEW pose after real agent motion; object static and not visible now.

    Encoding step is the latest distinguishable prior sighting (not raw
    ``last_seen_step``). Net pose change is verified from agent_trajectory
    (position OR heading), never from action count alone. Duplicate
    (object, encode) pairs with identical answers across query steps are dropped.
    """
    if min_delay < 1:
        raise ValueError('spatial_updating min_delay must be at least 1')
    if max_delay is not None and max_delay < min_delay:
        raise ValueError(
            'spatial_updating max_delay must be greater than or equal to min_delay'
        )

    displaced = _displaced_ids(episode)
    out: list[PlannedFact] = []
    # (obj_id, encode_step) -> answer_label of first emitted item
    seen_answers: dict[tuple[str, int], str] = {}
    for step in episode.get('steps') or []:
        step_idx = int(step['step'])
        for obj_id, mem in (step.get('non_visible_objects') or {}).items():
            if obj_id in displaced:
                continue
            # Encode at last distinguishable sighting, not last FOV scrape.
            enc_idx = _last_distinguishable_sighting(episode, obj_id, step_idx)
            if enc_idx is None or step_idx <= enc_idx:
                continue
            k = step_idx - enc_idx
            if not _delay_is_allowed(k, min_delay, max_delay):
                continue
            enc = step_by_index(episode, enc_idx)
            if enc is None:
                continue
            if not _net_pose_changed_between(episode, enc_idx, step_idx):
                continue
            if not _object_static_between(episode, obj_id, enc_idx, step_idx):
                continue
            # Queried object must NOT be visible at final pose
            if obj_id in (step.get('visible_objects') or {}):
                continue
            # Recompute bearing from poses (equal wedges); no stored-triple fallback
            label = None
            ag_pos, ag_rot = agent_pose_at_step(episode, step_idx)
            obj_pos = mem.get('position') or (mem.get('last_known') or {}).get('position')
            if obj_pos is None:
                obj_pos = _object_world_pos(episode, obj_id)
            if ag_pos is not None and ag_rot is not None and obj_pos is not None:
                label = ego_label_from_world_pose(
                    ag_pos,
                    ag_rot,
                    obj_pos,
                    episode,
                    ahead_half_width=AHEAD_HALF_WIDTH_FULL,
                )
            if not label or label not in EGO_DIRECTION_OPTIONS:
                continue
            key = (obj_id, enc_idx)
            if key in seen_answers and seen_answers[key] == label:
                continue  # duplicate encode/answer across query steps
            enc_vis = (enc.get('visible_objects') or {}).get(obj_id) or {}
            pre = _ego_label_at(
                episode,
                enc_idx,
                obj_id,
                obj_pos=enc_vis.get('position'),
                ahead_half_width=AHEAD_HALF_WIDTH_FOV,
            )
            if pre and pre not in EGO_DIRECTION_OPTIONS:
                pre = None
            pool, seeds = _ego_pool_with_diagnostics(label, pre or '')
            if pre and pre != label:
                seeds.append('pre_move_bearing')
                seeds.append(_mode_seed('pre_move_bearing', pre))
            if len(pool) < 2:
                continue
            disambiguator = _referring_disambiguator(episode, enc_idx, obj_id)
            if disambiguator is None:
                continue
            seen_answers[key] = label
            out.append(
                PlannedFact(
                    construct='spatial_updating',
                    status='ok',
                    query_step=step_idx,
                    encoding_step=enc_idx,
                    queried_object_id=obj_id,
                    answer_label=label,
                    answer_source=[
                        f"agent_trajectory[{enc_idx}→{step_idx}] (net pose change)",
                        f"object_state_track[{obj_id}] (static)",
                        f"agent_pose@[{step_idx}] + object_position (equal-wedge bearing)",
                    ],
                    image_paths=_images_between(episode, enc_idx, step_idx),
                    options_pool=pool,
                    distractor_seeds=seeds,
                    extra={
                        'object_type': object_type_from_id(obj_id, {obj_id: mem}),
                        'pre_move_label': pre,
                        'k': k,
                        'frame_of_reference': 'egocentric',
                        'disambiguator': disambiguator,
                    },
                )
            )
            if len(out) >= max_items:
                return out
    return out


def plan_allocentric_encoding(episode: dict, max_items: int = 1) -> list[PlannedFact]:
    """Unsupported until trusted reference facing / edges_object_frame exists."""
    return [
        PlannedFact(
            construct='allocentric_encoding',
            status='unsupported',
            reason='no_trusted_reference_facing_edges_object_frame_empty',
        )
    ][:max_items]



def _rotation_deg(episode: dict) -> float:
    meta = episode.get('episode_meta') or {}
    agent = meta.get('agent') or {}
    if agent.get('rotation_deg') is not None:
        return float(agent['rotation_deg'])
    return 45.0


def _landmark_display_name(obj_id: str, episode: dict) -> str:
    layout = episode.get('world_layout') or {}
    for lm in layout.get('landmarks') or []:
        if lm.get('landmark_id') == obj_id or lm.get('obj_id') == obj_id:
            return object_type_from_id(
                obj_id, {obj_id: {'category': lm.get('obj-type') or lm.get('category')}}
            )
    return object_type_from_id(obj_id)


def _object_world_pos(episode: dict, obj_id: str) -> Optional[tuple]:
    """Prefer layout landmark pose, else first distinguishable sighting pose."""
    layout = episode.get('world_layout') or {}
    for lm in layout.get('landmarks') or []:
        lid = lm.get('landmark_id') or lm.get('obj_id')
        if lid == obj_id:
            pos = lm.get('position')
            if isinstance(pos, dict):
                return (float(pos['x']), float(pos.get('y', 0.0)), float(pos['z']))
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                return (float(pos[0]), float(pos[1]), float(pos[2]))
    for step in episode.get('steps') or []:
        vis = step.get('visible_objects') or {}
        if obj_id in vis:
            pos = vis[obj_id].get('position')
            if pos is not None:
                return tuple(pos)
    return None


def _distinguishable_landmark_candidates(episode: dict) -> list[dict]:
    """Objects that passed Q&A visibility (in some step's visible_objects).

    Distinguishing filter already applied at GT build time for visible_objects.
    Prefer world_layout landmarks when they are also visible.
    """
    seen: dict[str, dict] = {}
    sighting_count: dict[str, int] = {}
    for step in episode.get('steps') or []:
        si = int(step['step'])
        for oid, odata in (step.get('visible_objects') or {}).items():
            sighting_count[oid] = sighting_count.get(oid, 0) + 1
            if oid in seen:
                continue
            pos = odata.get('position')
            if pos is None:
                continue
            seen[oid] = {
                'obj_id': oid,
                'name': object_type_from_id(oid, {oid: odata}),
                'position': tuple(pos) if not isinstance(pos, tuple) else pos,
                'first_seen_step': si,
                'from_layout': False,
                'region_id': None,
                'salience': 1.0,
            }
    layout_ids = set()
    for lm in (episode.get('world_layout') or {}).get('landmarks') or []:
        lid = lm.get('landmark_id') or lm.get('obj_id')
        if not lid:
            continue
        layout_ids.add(lid)
        if lid in seen:
            seen[lid]['from_layout'] = True
            seen[lid]['name'] = _landmark_display_name(lid, episode)
            seen[lid]['region_id'] = lm.get('region_id')
            # Prefer catalog/layout pose for snapping (may be on receptacle)
            wp = _object_world_pos(episode, lid)
            if wp is not None:
                seen[lid]['position'] = wp
            # Layout landmarks get a salience boost; more sightings → higher weight
            seen[lid]['salience'] = 2.0 + 0.1 * sighting_count.get(lid, 1)
        else:
            # Layout-only landmarks are not distinguishable — skip (filter first)
            continue
    for oid, row in seen.items():
        if not row['from_layout']:
            row['salience'] = 1.0 + 0.05 * sighting_count.get(oid, 1)
    items = list(seen.values())
    items.sort(key=lambda r: (-r['salience'], r['first_seen_step'], r['obj_id']))
    return items


def select_landmark_candidates(
    episode: dict, *, top_n_per_region: int = 5, max_total: int = 40
) -> list[dict]:
    """Salience-filtered landmarks: visibility first, then top-N weighted per region.

    Never nearest-distance-only and never uniform-random — matches taxonomy
    ``select_landmark_candidates()`` for class-4 endpoints.
    """
    candidates = _distinguishable_landmark_candidates(episode)
    if not candidates:
        return []
    by_region: dict[str, list[dict]] = {}
    for lm in candidates:
        rid = lm.get('region_id') or '_unknown'
        by_region.setdefault(rid, []).append(lm)
    selected: list[dict] = []
    for _rid, rows in by_region.items():
        rows = sorted(rows, key=lambda r: (-r['salience'], r['first_seen_step']))
        selected.extend(rows[:top_n_per_region])
    selected.sort(key=lambda r: (-r['salience'], r['first_seen_step'], r['obj_id']))
    return selected[:max_total]


def _scene_min_hop_count(
    graph, landmark_nodes: list[str], *, floor: int = 2, default: int = 4, cap: int = 8
) -> int:
    """Calibrate min hop count from this scene's landmark-pair shortest paths.

    Uses ~25th percentile of pairwise lengths, capped so MC routes stay short
    (not R2R's fixed 4–6, and not the house-wide diameter). Floor is 2 so
    short but real walks in compact houses are not rejected by a hard 3–4.
    """
    import networkx as nx

    nodes = list(dict.fromkeys(landmark_nodes))
    if len(nodes) < 2:
        return default
    lengths: list[int] = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if a not in graph or b not in graph:
                continue
            try:
                lengths.append(len(nx.shortest_path(graph, a, b)) - 1)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    if not lengths:
        return default
    lengths.sort()
    idx = max(0, int(0.25 * (len(lengths) - 1)))
    return max(floor, min(int(lengths[idx]), cap))


def _nearest_landmark_name(
    graph, node_id: str, landmarks_by_node: dict[str, str]
) -> Optional[str]:
    if node_id in landmarks_by_node:
        return landmarks_by_node[node_id]
    return None


def _build_landmark_node_map(
    graph,
    landmarks: list[dict],
    *,
    candidate_nodes: Optional[Sequence[str]] = None,
    max_distance_m: Optional[float] = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict]]:
    """Map landmark_id -> snapped graph node (naming vs math stay separate).

    If ``candidate_nodes`` is set (e.g. traversed walk for route_knowledge),
    snap each landmark to the nearest *visited* node within landmark radius.
    Otherwise snap to the nearest navigable graph node (survey endpoints).
    """
    from cm_benchmark.generation.nav_graph import (
        snap_landmark_to_graph,
        snap_to_nearest_of,
    )

    id_to_node: dict[str, str] = {}
    node_to_name: dict[str, str] = {}
    meta: dict[str, dict] = {}
    for lm in landmarks:
        oid = lm['obj_id']
        if candidate_nodes is not None:
            nid = snap_to_nearest_of(
                graph, lm['position'], candidate_nodes, max_distance_m=max_distance_m
            )
        else:
            nid = snap_landmark_to_graph(graph, lm['position'])
        if nid is None:
            continue
        id_to_node[oid] = nid
        # Prefer first landmark name if several share a node
        node_to_name.setdefault(nid, lm['name'])
        meta[oid] = {**lm, 'node_id': nid}
    return id_to_node, node_to_name, meta


def _load_nav_graph_or_none(episode: dict):
    from cm_benchmark.generation.nav_graph import build_nav_graph

    raw = episode.get('nav_graph')
    if not raw:
        return None
    try:
        return build_nav_graph(raw, snapshot='episode_start')
    except Exception:
        return None


def _traversed_subpath(traversed: list[str], start_node: str, end_node: str) -> Optional[list[str]]:
    """First contiguous walk from start_node to a later end_node."""
    try:
        i0 = traversed.index(start_node)
    except ValueError:
        return None
    for j in range(i0 + 1, len(traversed)):
        if traversed[j] == end_node:
            return traversed[i0 : j + 1]
    return None


def plan_route_knowledge(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Retrace an EXPERIENCED path as derive_turns() sequence (graph-backed)."""
    from cm_benchmark.generation.nav_graph import (
        derive_turns,
        format_turn_sequence,
        perturb_turn_sequence,
        snap_trajectory_to_graph,
        traversed_node_ids,
        was_traversed,
    )

    graph = _load_nav_graph_or_none(episode)
    if graph is None:
        return [
            PlannedFact(
                construct='route_knowledge',
                status='unsupported',
                reason='missing_or_invalid_nav_graph',
            )
        ]

    snapped = snap_trajectory_to_graph(graph, episode.get('agent_trajectory') or [])
    traversed = traversed_node_ids(snapped)
    if len(traversed) < 3:
        return [
            PlannedFact(
                construct='route_knowledge',
                status='unsupported',
                reason='trajectory_too_short_after_snap',
            )
        ]

    landmarks = select_landmark_candidates(episode)
    id_to_node, node_to_name, meta = _build_landmark_node_map(
        graph,
        landmarks,
        candidate_nodes=traversed,
        max_distance_m=ROUTE_LANDMARK_SNAP_M,
    )
    if len(id_to_node) < 2:
        return [
            PlannedFact(
                construct='route_knowledge',
                status='unsupported',
                reason='need_ge2_distinguishable_landmarks_snapped',
            )
        ]

    rot = _rotation_deg(episode)
    min_hops = _scene_min_hop_count(graph, list(id_to_node.values()))
    # Order pairs by walk appearance so source→goal follows the experienced path
    # (salience order alone often tries the reverse and rejects every pair).
    def _walk_index(oid: str) -> int:
        try:
            return traversed.index(id_to_node[oid])
        except ValueError:
            return 10**9

    ordered_ids = sorted(
        [lm['obj_id'] for lm in landmarks if lm['obj_id'] in id_to_node],
        key=lambda oid: (
            _walk_index(oid),
            int(meta[oid].get('first_seen_step') or 0),
            oid,
        ),
    )
    out: list[PlannedFact] = []
    for i, src_id in enumerate(ordered_ids):
        for goal_id in ordered_ids[i + 1 :]:
            if src_id == goal_id:
                continue
            n0, n1 = id_to_node[src_id], id_to_node[goal_id]
            if n0 == n1:
                continue
            source = meta[src_id]['name']
            goal = meta[goal_id]['name']
            if source == goal:
                continue
            sub = _traversed_subpath(traversed, n0, n1)
            if sub is None or len(sub) < 2:
                continue
            hops = len(sub) - 1
            if hops < min_hops:
                continue
            # Keep MC routes short: decision-point sequences, not full-episode dumps
            if len(sub) > ROUTE_MAX_SUBPATH_NODES:
                continue
            if not was_traversed(sub, traversed):
                continue
            # Require some real translation along the walk
            if not _has_real_move_between(
                episode,
                int(meta[src_id]['first_seen_step']),
                max(int(meta[goal_id]['first_seen_step']), int(meta[src_id]['first_seen_step']) + 1),
            ):
                if len(sub) < 3:
                    continue

            turns = derive_turns(
                sub, graph, rotation_deg=rot, landmark_at_node=node_to_name
            )
            if not turns:
                continue
            if not any((t.get('label') or 'straight') != 'straight' for t in turns):
                continue
            answer = format_turn_sequence(turns)
            if not answer or answer.count('→') > ROUTE_MAX_TURN_ARROWS:
                continue
            pool = [answer]
            seeds: list[str] = []
            for mode in (
                'reversed_sequence',
                'swapped_two_turns',
                'plausible_but_unwalked_route',
            ):
                pert = perturb_turn_sequence(turns, mode)
                if not pert:
                    continue
                lab = format_turn_sequence(pert)
                if lab and lab not in pool:
                    pool.append(lab)
                    seeds.append(mode)
                    seeds.append(_mode_seed(mode, lab))
                if len(pool) >= 4:
                    break
            if len(pool) < 2:
                continue

            t0 = int(meta[src_id]['first_seen_step'])
            t1 = int(meta[goal_id]['first_seen_step'])
            if t1 < t0:
                t0, t1 = t1, t0
            images, image_roles = merge_role_images(
                [
                    (
                        _img(step_by_index(episode, int(meta[src_id]['first_seen_step']))),
                        f'source · {source}',
                    ),
                    (
                        _img(step_by_index(episode, int(meta[goal_id]['first_seen_step']))),
                        f'goal · {goal}',
                    ),
                ]
            )
            out.append(
                PlannedFact(
                    construct='route_knowledge',
                    status='ok',
                    query_step=t1,
                    encoding_step=t0,
                    answer_label=answer,
                    answer_source=[
                        f'nav_graph.snap_trajectory[{n0}→{n1}]',
                        'derive_turns(traversed_subpath)',
                    ],
                    image_paths=images,
                    options_pool=pool[:4],
                    distractor_seeds=seeds,
                    extra={
                        'source': source,
                        'goal': goal,
                        'A': source,
                        'B': goal,
                        'source_landmark_id': src_id,
                        'goal_landmark_id': goal_id,
                        'source_node': n0,
                        'goal_node': n1,
                        'path_nodes': sub,
                        'min_hop_count': min_hops,
                        'hop_count': hops,
                        'turn_labels': [t.get('label') for t in turns],
                        'object_type': goal,
                        'frame_of_reference': 'egocentric',
                        'image_roles': image_roles,
                    },
                )
            )
            if len(out) >= max_items:
                return out

    if not out:
        return [
            PlannedFact(
                construct='route_knowledge',
                status='unsupported',
                reason='no_experienced_landmark_to_landmark_walk',
            )
        ]
    return out


def _passage_was_visible(episode: dict, passage_id: Optional[str]) -> bool:
    if not passage_id:
        return False
    for step in episode.get('steps') or []:
        for oid in (step.get('visible_objects') or {}):
            if _landmark_matches(oid, passage_id):
                return True
    return False


def _connection_perceptually_evidenced(
    episode: dict, src_id: str, goal_id: str, meta: dict
) -> bool:
    """Taxonomy: connection must be seen (doorway/passage or both regions), not layout-only."""
    layout = episode.get('world_layout') or {}
    src_region = meta.get(src_id, {}).get('region_id')
    goal_region = meta.get(goal_id, {}).get('region_id')
    # Passage connecting the two regions visible in some frame
    for row in layout.get('connectivity') or []:
        a, b = row.get('from_region'), row.get('to_region')
        pid = row.get('passage_id')
        if not pid or a is None or b is None or a == b:
            continue
        pair = {a, b}
        if src_region and goal_region and pair == {src_region, goal_region}:
            if _passage_was_visible(episode, pid):
                return True
    # Both landmarks distinguishable in frames (already true) + any doorway seen
    for step in episode.get('steps') or []:
        for oid in (step.get('visible_objects') or {}):
            if str(oid).lower().startswith('door'):
                return True
    # Same-region pairs: sightline via shared region observation
    if src_region and goal_region and src_region == goal_region:
        return True
    return False


def _recorded_passage_closures(episode: dict) -> list[dict]:
    """Passages observed closed at some timestep (real passage_state, never invented)."""
    layout = episode.get('world_layout') or {}
    passage_meta = {
        p.get('passage_id'): p for p in (layout.get('passages') or []) if p.get('passage_id')
    }
    closed: list[dict] = []
    seen = set()
    for row in episode.get('passage_state') or []:
        if row.get('is_open') is not False:
            continue
        pid = row.get('passage_id')
        if not pid or pid in seen:
            continue
        meta = passage_meta.get(pid) or {}
        fr = row.get('from_region') or meta.get('from_region')
        tr = row.get('to_region') or meta.get('to_region')
        if fr is None or tr is None or fr == tr:
            continue
        # Need a pose for the door to remove nearby edges
        pos = None
        for step in episode.get('steps') or []:
            vis = step.get('visible_objects') or {}
            for oid, odata in vis.items():
                if _landmark_matches(oid, pid):
                    pos = odata.get('position')
                    break
            if pos is not None:
                break
        if pos is None:
            continue
        seen.add(pid)
        closed.append(
            {
                'passage_id': pid,
                'from_region': fr,
                'to_region': tr,
                'position': pos,
                'timestep': row.get('timestep'),
            }
        )
    return closed


def plan_survey_based_route_planning(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Survey layout judgments: direction/distance and optional conditional_detour."""
    from cm_benchmark.generation.nav_graph import (
        direction_distance_between_landmarks,
        first_hop_direction_label,
        format_survey_relation,
        is_valid_untraversed_shortcut,
        remove_edges_near_position,
        sanitize_world_layout,
        shortest_path,
        snap_trajectory_to_graph,
        traversed_node_ids,
    )
    import networkx as nx

    if episode.get('world_layout'):
        episode = dict(episode)
        episode['world_layout'] = sanitize_world_layout(episode['world_layout'])

    graph = _load_nav_graph_or_none(episode)
    if graph is None:
        return [
            PlannedFact(
                construct='survey_based_route_planning',
                status='unsupported',
                reason='missing_or_invalid_nav_graph',
            )
        ]

    snapped = snap_trajectory_to_graph(graph, episode.get('agent_trajectory') or [])
    traversed = traversed_node_ids(snapped)
    landmarks = select_landmark_candidates(episode)
    id_to_node, _node_to_name, meta = _build_landmark_node_map(graph, landmarks)
    ids = [lm['obj_id'] for lm in landmarks if lm['obj_id'] in id_to_node]
    if len(ids) < 2:
        return [
            PlannedFact(
                construct='survey_based_route_planning',
                status='unsupported',
                reason='need_ge2_distinguishable_landmarks_snapped',
            )
        ]

    closures = _recorded_passage_closures(episode)
    out: list[PlannedFact] = []

    def _emit_direction_distance(src_id, goal_id, n0, n1, cand):
        nonlocal out
        if not _connection_perceptually_evidenced(episode, src_id, goal_id, meta):
            return
        rel = direction_distance_between_landmarks(
            meta[src_id]['position'], meta[goal_id]['position'], episode=episode
        )
        if not rel:
            return
        direction, distance = rel
        source = meta[src_id]['name']
        goal = meta[goal_id]['name']
        if source == goal:
            return
        answer = format_survey_relation(direction, distance, source_name=source)
        opp = {
            'ahead of': 'behind',
            'behind': 'ahead of',
            'to the left of': 'to the right of',
            'to the right of': 'to the left of',
        }.get(direction, direction)
        alt_dists = [d for d in ('within_reach', 'nearby', 'far', 'beyond') if d != distance]
        pool = [answer]
        seeds: list[str] = []
        decoy1 = format_survey_relation(opp, distance, source_name=source)
        if decoy1 not in pool:
            pool.append(decoy1)
            seeds += ['opposite_direction', _mode_seed('opposite_direction', decoy1)]
        if alt_dists:
            decoy2 = format_survey_relation(direction, alt_dists[0], source_name=source)
            if decoy2 not in pool:
                pool.append(decoy2)
                seeds += ['wrong_distance', _mode_seed('wrong_distance', decoy2)]
        if alt_dists and opp != direction:
            decoy3 = format_survey_relation(opp, alt_dists[-1], source_name=source)
            if decoy3 not in pool:
                pool.append(decoy3)
                seeds += ['known_route_answer', _mode_seed('known_route_answer', decoy3)]
        if len(pool) < 2:
            return
        t0 = int(meta[src_id]['first_seen_step'])
        t1 = int(meta[goal_id]['first_seen_step'])
        if _agent_near_landmark(episode, meta[src_id]['position'], [t0, t1]):
            return
        images, image_roles = merge_role_images(
            [
                (_img(step_by_index(episode, t0)), f'source · {source}'),
                (_img(step_by_index(episode, t1)), f'goal · {goal}'),
            ]
        )
        out.append(
            PlannedFact(
                construct='survey_based_route_planning',
                status='ok',
                query_step=max(t0, t1),
                encoding_step=min(t0, t1),
                answer_label=answer,
                answer_source=[
                    f'landmarks[{src_id}].position',
                    f'landmarks[{goal_id}].position',
                    f'nav_graph.shortest_path[{n0}→{n1}] (untraversed)',
                ],
                image_paths=images,
                options_pool=pool[:4],
                distractor_seeds=seeds,
                extra={
                    'source': source,
                    'goal': goal,
                    'A': source,
                    'B': goal,
                    'template_mode': 'direction_distance',
                    'source_landmark_id': src_id,
                    'goal_landmark_id': goal_id,
                    'path_nodes': cand,
                    'direction': direction,
                    'distance_label': distance,
                    'object_type': goal,
                    'frame_of_reference': 'allocentric',
                    'image_roles': image_roles,
                },
            )
        )

    def _emit_conditional_detour(src_id, goal_id, n0, n1, closure):
        nonlocal out
        blocked = remove_edges_near_position(graph, closure['position'])
        try:
            detour = shortest_path(blocked, n0, n1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return
        if len(detour) < 2:
            return
        # Must differ from open-graph first hop (otherwise condition is inert)
        try:
            open_path = shortest_path(graph, n0, n1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return
        if len(open_path) >= 2 and open_path[1] == detour[1]:
            return
        label = first_hop_direction_label(
            blocked,
            detour,
            source_pos=meta[src_id]['position'],
            goal_pos=meta[goal_id]['position'],
        )
        if not label or label not in EGO_DIRECTION_OPTIONS:
            return
        open_label = first_hop_direction_label(
            graph,
            open_path,
            source_pos=meta[src_id]['position'],
            goal_pos=meta[goal_id]['position'],
        )
        source = meta[src_id]['name']
        goal = meta[goal_id]['name']
        if source == goal:
            return
        pool = [label]
        seeds: list[str] = []
        opp = OPPOSITE.get(label)
        if opp and opp not in pool:
            pool.append(opp)
            seeds += ['opposite_direction', _mode_seed('opposite_direction', opp)]
        if open_label and open_label not in pool:
            pool.append(open_label)
            seeds += ['known_route_answer', _mode_seed('known_route_answer', open_label)]
        mir = MIRRORED_LR.get(label)
        if mir and mir not in pool:
            pool.append(mir)
            seeds += ['wrong_distance', _mode_seed('wrong_distance', mir)]
        for filler in EGO_DIRECTION_OPTIONS:
            if len(pool) >= 4:
                break
            if filler not in pool:
                pool.append(filler)
        if len(pool) < 2:
            return
        pid = closure['passage_id']
        condition = f'the {object_type_from_id(pid)} is closed'
        t0 = int(meta[src_id]['first_seen_step'])
        t1 = int(meta[goal_id]['first_seen_step'])
        if _agent_near_landmark(episode, meta[src_id]['position'], [t0, t1]):
            return
        images, image_roles = merge_role_images(
            [
                (_img(step_by_index(episode, t0)), f'source · {source}'),
                (_img(step_by_index(episode, t1)), f'goal · {goal}'),
            ]
        )
        out.append(
            PlannedFact(
                construct='survey_based_route_planning',
                status='ok',
                query_step=max(t0, t1),
                encoding_step=min(t0, t1),
                answer_label=label,
                answer_source=[
                    f'passage_state[{pid}].is_open=false',
                    f'nav_graph.shortest_path[{n0}→{n1}] (edge removed near {pid})',
                    'first_hop_direction_label',
                ],
                image_paths=images,
                options_pool=pool[:4],
                distractor_seeds=seeds,
                extra={
                    'source': source,
                    'goal': goal,
                    'A': source,
                    'B': goal,
                    'condition': condition,
                    'template_mode': 'conditional_detour',
                    'passage_id': pid,
                    'source_landmark_id': src_id,
                    'goal_landmark_id': goal_id,
                    'path_nodes': detour,
                    'object_type': goal,
                    'frame_of_reference': 'allocentric',
                    'image_roles': image_roles,
                },
            )
        )

    for i, src_id in enumerate(ids):
        for goal_id in ids[i + 1 :]:
            n0, n1 = id_to_node[src_id], id_to_node[goal_id]
            if n0 == n1:
                continue
            try:
                cand = shortest_path(graph, n0, n1)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if not is_valid_untraversed_shortcut(cand, traversed, graph):
                continue
            before = len(out)
            _emit_direction_distance(src_id, goal_id, n0, n1, cand)
            if len(out) > before and len(out) >= max_items:
                return out
            for closure in closures:
                before = len(out)
                _emit_conditional_detour(src_id, goal_id, n0, n1, closure)
                if len(out) > before and len(out) >= max_items:
                    return out

    if not out:
        return [
            PlannedFact(
                construct='survey_based_route_planning',
                status='unsupported',
                reason='no_novel_untraversed_landmark_pair',
            )
        ]
    return out


def plan_perspective_taking(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Object Perspective / Spatial Orientation Test: stand at A facing B, locate C."""
    import math as _math

    landmarks = select_landmark_candidates(episode, top_n_per_region=4, max_total=18)
    if len(landmarks) < 3:
        return [
            PlannedFact(
                construct='perspective_taking',
                status='unsupported',
                reason='need_ge3_distinguishable_landmarks',
            )
        ]

    out: list[PlannedFact] = []
    # Prefer a late query step where all three have been seen
    for i, a in enumerate(landmarks):
        for j, b in enumerate(landmarks):
            if i == j:
                continue
            for k, c in enumerate(landmarks):
                if k in (i, j):
                    continue
                pos_a = xyz_as_dict(a['position'])
                pos_b = xyz_as_dict(b['position'])
                pos_c = xyz_as_dict(c['position'])
                if not pos_a or not pos_b or not pos_c:
                    continue
                if a['name'] == b['name'] or a['name'] == c['name'] or b['name'] == c['name']:
                    continue
                label = imagined_perspective_label(
                    pos_a, pos_b, pos_c, ahead_half_width=AHEAD_HALF_WIDTH_FULL
                )
                if not label or label not in EGO_DIRECTION_OPTIONS:
                    continue
                query_step = max(
                    int(a['first_seen_step']),
                    int(b['first_seen_step']),
                    int(c['first_seen_step']),
                )
                # Camera-frame distractor: C relative to actual agent pose (FOV)
                ag_pos, ag_rot = agent_pose_at_step(episode, query_step)
                cam = (
                    ego_label_from_world_pose(
                        ag_pos,
                        ag_rot,
                        c['position'],
                        episode,
                        ahead_half_width=AHEAD_HALF_WIDTH_FOV,
                    )
                    if ag_pos is not None
                    else None
                )
                mirrored = MIRRORED_LR.get(label)
                # Wrong facing: stand at A facing away from B (toward -B)
                wrong_b = {
                    'x': pos_a['x'] - (pos_b['x'] - pos_a['x']),
                    'y': pos_a['y'],
                    'z': pos_a['z'] - (pos_b['z'] - pos_a['z']),
                }
                wrong = imagined_perspective_label(
                    pos_a, wrong_b, pos_c, ahead_half_width=AHEAD_HALF_WIDTH_FULL
                )
                pool = [label]
                seeds: list[str] = []
                if cam and cam not in pool:
                    pool.append(cam)
                    seeds += ['camera_frame_answer', _mode_seed('camera_frame_answer', cam)]
                if mirrored and mirrored not in pool:
                    pool.append(mirrored)
                    seeds += ['mirrored_left_right', _mode_seed('mirrored_left_right', mirrored)]
                if wrong and wrong not in pool:
                    pool.append(wrong)
                    seeds += [
                        'wrong_facing_assumption',
                        _mode_seed('wrong_facing_assumption', wrong),
                    ]
                for filler in EGO_DIRECTION_OPTIONS:
                    if len(pool) >= 4:
                        break
                    if filler not in pool:
                        pool.append(filler)
                if len(pool) < 2:
                    continue
                shift = None
                pose = _agent_pose_dict(episode, query_step)
                if pose is not None:
                    imag_h = _math.degrees(
                        _math.atan2(pos_b['x'] - pos_a['x'], pos_b['z'] - pos_a['z'])
                    )
                    shift = abs((imag_h - pose['heading'] + 180) % 360 - 180)
                images = merge_role_images(
                    [
                        (
                            _img(step_by_index(episode, a['first_seen_step'])),
                            f"A · stand here ({a['name']})",
                        ),
                        (
                            _img(step_by_index(episode, b['first_seen_step'])),
                            f"B · face toward ({b['name']})",
                        ),
                        (
                            _img(step_by_index(episode, c['first_seen_step'])),
                            f"C · locate ({c['name']})",
                        ),
                    ]
                )
                uniq_images, image_roles = images
                out.append(
                    PlannedFact(
                        construct='perspective_taking',
                        status='ok',
                        query_step=query_step,
                        encoding_step=min(
                            int(a['first_seen_step']),
                            int(b['first_seen_step']),
                            int(c['first_seen_step']),
                        ),
                        queried_object_id=c['obj_id'],
                        reference_object_id=a['obj_id'],
                        answer_label=label,
                        answer_source=[
                            f'landmarks[{a["obj_id"]}].position (A)',
                            f'landmarks[{b["obj_id"]}].position (B)',
                            f'landmarks[{c["obj_id"]}].position (C)',
                            'imagined_perspective_label(A, A→B, C)',
                        ],
                        image_paths=uniq_images,
                        options_pool=pool[:4],
                        distractor_seeds=seeds,
                        extra={
                            'A': a['name'],
                            'B': b['name'],
                            'C': c['name'],
                            'source': a['name'],
                            'goal': b['name'],
                            'object_type': c['name'],
                            'landmark_a_id': a['obj_id'],
                            'landmark_b_id': b['obj_id'],
                            'landmark_c_id': c['obj_id'],
                            'perspective_shift_magnitude': shift,
                            'frame_of_reference': 'allocentric',
                            'image_roles': image_roles,
                        },
                    )
                )
                if len(out) >= max_items:
                    return out
    if not out:
        return [
            PlannedFact(
                construct='perspective_taking',
                status='unsupported',
                reason='no_valid_ABC_landmark_triple',
            )
        ]
    return out


PLANNERS = {
    'egocentric_encoding': plan_egocentric_encoding,
    'spatial_working_memory': plan_spatial_working_memory,
    'invisible_displacement': plan_invisible_displacement,
    'spatial_updating': plan_spatial_updating,
    'allocentric_encoding': plan_allocentric_encoding,
    'route_knowledge': plan_route_knowledge,
    'survey_based_route_planning': plan_survey_based_route_planning,
    'perspective_taking': plan_perspective_taking,
}


def plan_episode(
    episode: dict,
    constructs: Optional[list[str]] = None,
    max_per_construct: int = 3,
    *,
    swm_min_delay: int = 2,
    swm_max_delay: Optional[int] = None,
    su_min_delay: int = 2,
    su_max_delay: Optional[int] = None,
) -> list[PlannedFact]:
    from cm_benchmark.generator.visibility_filters import (
        apply_question_visibility_to_episode,
    )

    # Draft-time safety net: drop tiny/indistinct FOV blobs even if the episode
    # JSON was exported with question_visibility all-null / no joblib model.
    episode = apply_question_visibility_to_episode(episode, inplace=False)

    keys = constructs or list(PLANNERS.keys())
    facts: list[PlannedFact] = []
    for key in keys:
        fn = PLANNERS.get(key)
        if fn is None:
            continue
        if key == 'spatial_working_memory':
            facts.extend(
                fn(
                    episode,
                    max_items=max_per_construct,
                    min_delay=swm_min_delay,
                    max_delay=swm_max_delay,
                )
            )
        elif key == 'spatial_updating':
            facts.extend(
                fn(
                    episode,
                    max_items=max_per_construct,
                    min_delay=su_min_delay,
                    max_delay=su_max_delay,
                )
            )
        else:
            facts.extend(fn(episode, max_items=max_per_construct))
    return facts
