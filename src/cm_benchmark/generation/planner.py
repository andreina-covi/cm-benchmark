"""Task planner: select construct-eligible facts from episode GT (deterministic)."""

from __future__ import annotations

import math
from collections import Counter, deque
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
    camera_size_wh,
    find_ego_edge,
    fov_metrics_ok,
    humanize_receptacle,
    imagined_perspective_label,
    net_pose_changed,
    object_type_from_id,
    pick_template_index,
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
    from cm_benchmark.generator.visibility_filters import visibility_model_from_episode

    return resolve_referring_disambiguator(
        step,
        obj_id,
        _agent_pose_dict(episode, int(step_idx)),
        model=visibility_model_from_episode(episode),
    )


def _referring_display_name(
    episode: dict,
    step_idx: int,
    obj_id: str,
    *,
    min_bbox_area: float = QUERY_FOV_MIN_BBOX_AREA,
    min_side: float = QUERY_FOV_MIN_SIDE,
    min_visible_pixels: float = QUERY_FOV_MIN_VISIBLE_PIXELS,
) -> Optional[str]:
    """Human name for questions: ``Chair`` or ``Chair close to the Table``.

    Requires a distinguishable FOV sighting at ``step_idx``. Returns None when the
    object is weak/missing or a duplicate category cannot be uniquely referred.
    """
    if not _distinguishable_encoding_sighting(
        episode,
        step_idx,
        obj_id,
        min_bbox_area=min_bbox_area,
        min_side=min_side,
        min_visible_pixels=min_visible_pixels,
    ):
        return None
    phrase = _referring_disambiguator(episode, step_idx, obj_id)
    if phrase is None:
        return None
    step = step_by_index(episode, int(step_idx))
    catalog = (step or {}).get('visible_objects') or {}
    base = object_type_from_id(obj_id, catalog)
    return f'{base}{phrase}'


def _distinguishable_encoding_sighting(
    episode: dict,
    step_idx: int,
    obj_id: str,
    *,
    min_bbox_area: float = QUERY_FOV_MIN_BBOX_AREA,
    min_side: float = QUERY_FOV_MIN_SIDE,
    min_visible_pixels: float = QUERY_FOV_MIN_VISIBLE_PIXELS,
) -> bool:
    """Object must be clearly visible at encode (metrics + ego edge), not just listed.

    When a DecisionTree visibility model is attached on the episode, that model
    decides distinguishability (static ``min_*`` floors are ignored). Otherwise
    the static floors apply. Thin border-clipped scrapes are always rejected
    when ``episode_meta.camera`` size is available.
    """
    from cm_benchmark.generator.visibility_filters import visibility_model_from_episode

    step = step_by_index(episode, int(step_idx))
    if step is None:
        return False
    odata = (step.get('visible_objects') or {}).get(obj_id)
    model = visibility_model_from_episode(episode)
    if not fov_metrics_ok(
        odata,
        min_bbox_area=min_bbox_area,
        min_side=min_side,
        min_visible_pixels=min_visible_pixels,
        image_wh=camera_size_wh(episode),
        model=model,
    ):
        return False
    return find_ego_edge(step, obj_id) is not None


# Soft FOV kwargs for invisible-displacement encode when no DecisionTree is loaded.
# With a tree attached on the episode, ``_distinguishable_encoding_sighting`` uses
# the tree instead. Query-time invisibility is separate (``_object_hidden_through``).
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
            if not _distinguishable_encoding_sighting(episode, step_idx, obj_id):
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
    """Count distinct ids of ``category`` with a distinguishable sighting ≤ up_to_step."""
    seen: set[str] = set()
    for step in episode.get('steps') or []:
        si = int(step['step'])
        if si > int(up_to_step):
            continue
        for oid, odata in (step.get('visible_objects') or {}).items():
            if object_type_from_id(oid, {oid: odata}) != category:
                continue
            if oid in seen:
                continue
            if _distinguishable_encoding_sighting(episode, si, oid):
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

# Survey stays allocentric: agent inside the ObjectNav/GOAT success zone of the
# source (1.0 m Euclidean) collapses "from the Chair" into "from you".
# Check the source sighting frame only (not the goal frame).
SURVEY_MIN_AGENT_SOURCE_DIST_M = 1.0
# HM3D-OVON / GOAT-Bench / HSSD-200 (2024): geodesic start→goal in [1, 30] m.
# Not a Euclidean percentile; not R2R's 5 m / 4–6 hops (wrong graph scale).
MIN_PAIR_GEODESIC_M = 1.0
MAX_PAIR_GEODESIC_M = 30.0
# That band is absolute and was calibrated on large real-scanned homes. A small
# ProcTHOR house can put every candidate pair just above the 1 m floor, so the
# band stops discriminating and ranking only picks the hardest of a uniformly
# easy set. Pairs must also clear this percentile of the scene's OWN landmark
# geodesic distribution — both halves apply, not either/or.
SCENE_PAIR_GEODESIC_PERCENTILE = 0.5
# HSSD-200 CVPR 2024 supplement: geo/eucl < 1.05 is "nearly straight-line".
# These are caps: the working threshold is min(cap, this scene's own p50).
SURVEY_MIN_GEODESIC_EUCLIDEAN_RATIO = 1.05
ROUTE_MAX_GEODESIC_EUCLIDEAN_RATIO_CAP = 1.1

# Route-knowledge: snap landmarks onto the walked path with a slightly larger
# radius than the default 1.5 m so room-scale furniture near the trajectory
# still becomes an endpoint. Action-sequence length caps stay explicit (not R2R).
ROUTE_LANDMARK_SNAP_M = 2.5
ROUTE_MIN_HOP_COUNT = 2  # at least one real edge; length gate is geodesic metres
ROUTE_MAX_SUBPATH_NODES = 512  # lattice hops, not decisions; turns are the real cap
# Real direction changes (contiguous rotations count once), not rotation tokens.
# Habitat pairs its distance band with episode_min_steps=11; a turn floor is the
# analogue for an action-sequence answer — a straight walk is guessable blind.
ROUTE_MIN_TURN_COUNT = 2
ROUTE_MAX_TURN_COUNT = 12


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
    Display name includes a referring disambiguator when the category is ambiguous.
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
            if not _distinguishable_encoding_sighting(episode, si, oid):
                continue
            pos = odata.get('position')
            xz = _xz_from_any(pos)
            if xz is None:
                continue
            d = ((xz[0] - target[0]) ** 2 + (xz[1] - target[1]) ** 2) ** 0.5
            if d <= radius_m and d < best_d:
                name = _referring_display_name(episode, si, oid)
                if name is None:
                    continue
                best_d = d
                best = (oid, name)
    return best


def _landmark_matches(visible_id: str, landmark_id: str) -> bool:
    if visible_id == landmark_id:
        return True
    return str(visible_id).split('|')[0] == str(landmark_id).split('|')[0]


def _visible_oid_matching(step: dict, landmark_id: str) -> Optional[str]:
    for oid in step.get('visible_objects') or {}:
        if _landmark_matches(oid, landmark_id):
            return oid
    return None


def _object_mark_at(
    episode: dict, step_idx: int, obj_id: Optional[str]
) -> Optional[dict]:
    """Slide overlay cue from episode GT at ``step_idx``.

    Overlay draws the 2D ``bbox`` center ``(cmin, rmin, cmax, rmax)`` only
    (camera pixels, origin top-left). ``local_point`` is stored for traceability
    but is the projected Unity pivot, not the marker position.
    """
    if not obj_id:
        return None
    step = step_by_index(episode, step_idx)
    if not step:
        return None
    vis = step.get('visible_objects') or {}
    odata = vis.get(obj_id)
    if odata is None:
        matches = [oid for oid in vis if _landmark_matches(oid, obj_id)]
        if len(matches) == 1:
            odata = vis.get(matches[0])
    if not isinstance(odata, dict):
        return None
    mark: dict = {}
    bbox = odata.get('bbox')
    if bbox and len(bbox) >= 4:
        mark['bbox'] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    lp = odata.get('local_point')
    if lp and len(lp) >= 2 and lp[0] is not None and lp[1] is not None:
        mark['local_point'] = [float(lp[0]), float(lp[1])]
    size = camera_size_wh(episode)
    if size:
        mark['frame_wh'] = [int(size[0]), int(size[1])]
    img = step.get('image_path')
    if img:
        mark['image_path'] = img
    return mark or None


def _xyz3(pos) -> Optional[tuple[float, float, float]]:
    d = xyz_as_dict(pos)
    if d is None:
        return None
    return (float(d['x']), float(d.get('y') or 0.0), float(d['z']))


def _rot3(rot) -> Optional[tuple[float, float, float]]:
    if rot is None:
        return None
    if isinstance(rot, dict):
        return (
            float(rot.get('x') or 0.0),
            float(rot.get('y') or 0.0),
            float(rot.get('z') or 0.0),
        )
    if isinstance(rot, (list, tuple)) and len(rot) >= 3:
        return (float(rot[0]), float(rot[1]), float(rot[2]))
    return None


def _landmark_distinguishable_in_frames(
    episode: dict,
    landmark_id: Optional[str],
    frame_steps: list[int],
    *,
    min_bbox_area: float = QUERY_FOV_MIN_BBOX_AREA,
    min_side: float = QUERY_FOV_MIN_SIDE,
    min_visible_pixels: float = QUERY_FOV_MIN_VISIBLE_PIXELS,
) -> bool:
    """True if landmark appears with QUERY-level FOV metrics in some shown frame."""
    if not landmark_id or _is_floor_receptacle(landmark_id):
        return False
    for si in frame_steps:
        step = step_by_index(episode, si)
        if not step:
            continue
        oid = _visible_oid_matching(step, landmark_id)
        if oid is None:
            continue
        if _distinguishable_encoding_sighting(
            episode,
            si,
            oid,
            min_bbox_area=min_bbox_area,
            min_side=min_side,
            min_visible_pixels=min_visible_pixels,
        ):
            return True
    return False


def _landmark_display_in_frames(
    episode: dict, landmark_id: Optional[str], frame_steps: list[int]
) -> Optional[str]:
    """Referring display name for a destination landmark visible in shown frames."""
    if not landmark_id or _is_floor_receptacle(landmark_id):
        return None
    for si in frame_steps:
        step = step_by_index(episode, si)
        if not step:
            continue
        oid = _visible_oid_matching(step, landmark_id)
        if oid is None:
            continue
        name = _referring_display_name(episode, si, oid)
        if name is not None:
            return name
    return None


def _object_distinguishable_in_frames(
    episode: dict,
    obj_id: str,
    frame_steps: list[int],
    *,
    min_bbox_area: float = QUERY_FOV_MIN_BBOX_AREA,
    min_side: float = QUERY_FOV_MIN_SIDE,
    min_visible_pixels: float = QUERY_FOV_MIN_VISIBLE_PIXELS,
) -> bool:
    for si in frame_steps:
        if _distinguishable_encoding_sighting(
            episode,
            si,
            obj_id,
            min_bbox_area=min_bbox_area,
            min_side=min_side,
            min_visible_pixels=min_visible_pixels,
        ):
            return True
    return False


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
    """True if obj is in ``visible_objects`` or has a track row at this step in FOV.

    Track lookup is **exact-step** (no stale carry-forward of an older True).
    """
    step = step_by_index(episode, step_idx)
    if step and obj_id in (step.get('visible_objects') or {}):
        return True
    track = (episode.get('object_state_track') or {}).get(obj_id) or {}
    for entry in track.get('entries') or []:
        t = entry.get('step', entry.get('timestep'))
        if t is None or int(t) != int(step_idx):
            continue
        return bool(entry.get('in_camera_fov') or entry.get('visible'))
    return False


def _id_encode_kwargs(episode: dict) -> dict:
    """FOV kwargs for ID encode distinguishability.

    Soft ``ID_ENCODE_FOV_*`` floors apply when no DecisionTree is attached;
    ``_distinguishable_encoding_sighting`` prefers the tree when present.
    """
    return dict(_ID_ENCODE_FOV)


def _id_displaced_object_name(
    episode: dict, step_idx: int, obj_id: str
) -> Optional[tuple[str, str]]:
    """Display name for a displaced prop at a distinguishable encode step.

    Returns ``(object_type, disambiguator)``. Requires the object to be
    uniquely nameable at ``step_idx`` (referring phrase when category is
    ambiguous).
    """
    if not _distinguishable_encoding_sighting(
        episode, int(step_idx), obj_id, **_id_encode_kwargs(episode)
    ):
        return None
    step = step_by_index(episode, int(step_idx))
    if step is None:
        return None
    vo = step.get('visible_objects') or {}
    base = object_type_from_id(obj_id, vo)
    if not base:
        return None
    phrase = _referring_disambiguator(episode, int(step_idx), obj_id)
    if phrase is None:
        return None
    return base, phrase


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


def _pad_id_ego_options(
    answer: str, pool: list[str], seeds: list[str]
) -> tuple[list[str], list[str]]:
    """Pad diagnostic ego labels to four with opposite / orthogonal / bank fillers.

    Diagnostic candidate bearings stay first. Fillers are tagged so rationale
    does not pretend they came from receptacle teleports.
    """
    out = [lab for lab in pool if lab]
    seeds_out = list(seeds)
    if answer and answer not in out:
        out.insert(0, answer)

    def _add(lab: str, mode: str) -> None:
        if lab and lab not in out and len(out) < 4:
            out.append(lab)
            seeds_out.append(mode)
            seeds_out.append(_mode_seed(mode, lab))

    _add(OPPOSITE.get(answer) or '', 'opposite_direction')
    _add(ORTHOGONAL.get(answer) or '', 'orthogonal_direction')
    for lab in EGO_DIRECTION_OPTIONS:
        if len(out) >= 4:
            break
        _add(lab, 'ego_bank_filler')
    return out[:4], seeds_out


def _pick_best_id_query_step(
    episode: dict, obj_id: str, at_t: int, specs: list[dict]
) -> Optional[int]:
    """Prefer a hidden query step where candidate poses yield ≥2 unique ego labels.

    Score: more unique diagnostic labels first, then later step among ties.
    Options are padded to four later in ``_try_ego_direction_fact``.
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
    disambiguator: str = '',
) -> Optional[PlannedFact]:
    answer_dir, dir_pool, dir_seeds = _ego_pool_from_specs(episode, specs, query_step)
    if not answer_dir or answer_dir not in dir_pool or len(dir_pool) < 2:
        return None
    dir_pool, dir_seeds = _pad_id_ego_options(answer_dir, dir_pool, dir_seeds)
    if len(dir_pool) < 4:
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

    Displaced objects (and swap partners) must be **distinguishable** at an encode
    step before the move (DecisionTree if attached, else soft ``ID_ENCODE_FOV_*``),
    and **invisible** from the move through the query step. Destination landmarks
    for direct moves still must be nameable in shown frames. Ego MC options are
    diagnostic candidate bearings first, then padded to four from the ego bank.
    """
    events = episode.get('displacement_events') or []
    if not events:
        return []

    enc_kw = _id_encode_kwargs(episode)
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
        # Encode: last distinguishable sighting before the hidden move.
        enc_idx = _last_distinguishable_sighting(
            episode, obj_id, int(at_t), **enc_kw
        )
        if enc_idx is None:
            continue
        named = _id_displaced_object_name(episode, enc_idx, obj_id)
        if named is None:
            continue
        object_type, disambiguator = named

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
                episode, partner_id, int(at_t), **enc_kw
            )
            if partner_enc is None:
                continue
            if not _object_hidden_through(
                episode, partner_id, int(at_t), int(query_step)
            ):
                continue
            frame_steps = _build_id_frame_steps(
                enc_idx, int(at_t), int(query_step), extra_steps=[partner_enc]
            )
            if not _object_distinguishable_in_frames(
                episode, partner_id, frame_steps, **enc_kw
            ):
                continue
            images = _images_for_steps(episode, frame_steps)
            if not images:
                continue
            partner_named = _id_displaced_object_name(episode, partner_enc, partner_id)
            if partner_named is None:
                continue
            partner_type, _partner_disamb = partner_named
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
                disambiguator=disambiguator,
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
            dest_name = _landmark_display_in_frames(episode, to_r, frame_steps)
            if dest_name is None:
                continue
            new_location = dest_name
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
            disambiguator=disambiguator,
        )
        if fact is not None:
            out.append(fact)

    if not out and events:
        return [
            PlannedFact(
                construct='invisible_displacement',
                status='unsupported',
                reason=(
                    'no_event_with_distinguishable_encode_hidden_query_and_unique_options'
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


def _agent_yaw_at_step(episode: dict, step_idx: int) -> Optional[float]:
    """Agent yaw (AI2-THOR, +Z = 0) at ``step_idx``, else None."""
    _pos, rot = agent_pose_at_step(episode, int(step_idx))
    r = _rot3(rot)
    if r is None:
        return None
    return float(r[1]) % 360.0


def _reference_nav_actions(
    graph,
    path_nodes: Sequence[str],
    rotation_deg: float,
    *,
    start_heading_deg: Optional[float] = None,
    verify_edges=None,
):
    """Reference collected-format actions for a node path (not exclusive gold).

    Uses the greedy follower, not one ``move_ahead`` per lattice hop: the
    exported grid is axis-aligned while the agent walks on 15°-offset headings,
    so a per-hop conversion emits a rotation at every staircase step.

    ``start_heading_deg`` should be the agent's real yaw at the source step.
    Rotations are quantized, so the reachable headings are
    ``start + 45k``; seeding from a lattice hop bearing instead puts that set
    15° off the corridors the agent actually walked and the follower oscillates.

    The sequence is replayed through the shared scorer simulator and rejected
    unless it lands on the goal and — when ``verify_edges`` is given — crosses
    only those edges. A stored reference must score as a valid success [CODE].
    """
    from cm_benchmark.generation.nav_graph import (
        count_direction_changes,
        follow_path_actions,
        format_nav_actions,
        node_within_goal_tolerance,
        path_start_heading_deg,
        path_to_nav_actions,
        simulate_nav_action_sequence,
        trajectory_snap_tolerance,
    )

    heading = start_heading_deg
    if heading is None:
        heading = path_start_heading_deg(graph, path_nodes)
    if heading is None:
        return None

    grid = float(graph.graph.get('grid_size') or 0.25)
    move = float(graph.graph.get('agent_move_m') or grid)
    # Smoothest first. A wide tolerance cuts staircase corners and can leave the
    # walked edge set, so fall back to tighter ones and finally to the exact
    # per-hop conversion, which stays on the chain by construction.
    ladder = [max(grid, move) * 1.5, max(grid, move), grid, grid / 2.0]

    def _verify(actions):
        if not actions:
            return None
        sim = simulate_nav_action_sequence(
            graph,
            path_nodes[0],
            actions,
            start_heading_deg=heading,
            rotation_deg=rotation_deg,
        )
        if not sim.get('ok'):
            return None
        if not node_within_goal_tolerance(
            graph, sim.get('end_node'), path_nodes[-1], trajectory_snap_tolerance(graph)
        ):
            return None
        if verify_edges is not None:
            legal = {tuple(sorted(e)) for e in verify_edges}
            crossed = sim.get('snapped') or []
            for a, b in zip(crossed, crossed[1:]):
                if a != b and tuple(sorted((a, b))) not in legal:
                    return None
        return actions

    chosen = None
    for tol in ladder:
        followed = follow_path_actions(
            graph,
            path_nodes,
            start_heading_deg=heading,
            rotation_deg=rotation_deg,
            simplify_tolerance_m=tol,
        )
        if followed and _verify(followed['actions']):
            chosen = followed['actions']
            break
    if chosen is None:
        chosen = _verify(
            path_to_nav_actions(
                path_nodes,
                graph,
                start_heading_deg=heading,
                rotation_deg=rotation_deg,
            )
        )
    if not chosen:
        return None
    return {
        'actions': chosen,
        'answer_label': format_nav_actions(chosen),
        'start_heading_deg': heading,
        'turn_count': count_direction_changes(chosen),
    }


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
    """Landmarks with a QUERY-FOV distinguishable sighting (metrics + ego edge).

    ``first_seen_step`` is the earliest such sighting, not a weak FOV scrape.
    Prefer world_layout landmarks when they are also distinguishable.
    """
    seen: dict[str, dict] = {}
    sighting_count: dict[str, int] = {}
    for step in episode.get('steps') or []:
        si = int(step['step'])
        for oid, odata in (step.get('visible_objects') or {}).items():
            if not _distinguishable_encoding_sighting(episode, si, oid):
                continue
            sighting_count[oid] = sighting_count.get(oid, 0) + 1
            if oid in seen:
                continue
            pos = odata.get('position')
            if pos is None:
                continue
            # Must be uniquely nameable at this sighting (or category-unique).
            if _referring_display_name(episode, si, oid) is None:
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
    for lm in (episode.get('world_layout') or {}).get('landmarks') or []:
        lid = lm.get('landmark_id') or lm.get('obj_id')
        if not lid:
            continue
        if lid in seen:
            seen[lid]['from_layout'] = True
            seen[lid]['name'] = _landmark_display_name(lid, episode)
            seen[lid]['region_id'] = lm.get('region_id')
            wp = _object_world_pos(episode, lid)
            if wp is not None:
                seen[lid]['position'] = wp
            seen[lid]['salience'] = 2.0 + 0.1 * sighting_count.get(lid, 1)
        # Layout-only (never distinguishable in FOV) — skip
    for oid, row in seen.items():
        if not row['from_layout']:
            row['salience'] = 1.0 + 0.05 * sighting_count.get(oid, 1)
    items = list(seen.values())
    items.sort(key=lambda r: (-r['salience'], r['first_seen_step'], r['obj_id']))
    return items


def select_landmark_candidates(
    episode: dict,
    *,
    top_n_per_region: int = 5,
    max_total: int = 40,
    unique_category: bool = True,
) -> list[dict]:
    """Salience-filtered landmarks: visibility first, then top-N weighted per region.

    Never nearest-distance-only and never uniform-random. Class-4 endpoints
    pass ``unique_category=False`` and uniquely name duplicates via
    ``_referring_display_name`` (GOAT-Bench 2024 instance language; shared
    taxonomy referring-expression exception). Other callers keep the default
    True so source/goal names stay category-unique.
    """
    candidates = _distinguishable_landmark_candidates(episode)
    if not candidates:
        return []
    if unique_category:
        counts = Counter(lm.get('name') for lm in candidates)
        candidates = [lm for lm in candidates if counts.get(lm.get('name')) == 1]
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


def _full_graph_geodesic_m(graph, n0: str, n1: str) -> Optional[float]:
    """Weighted shortest-path length on the full nav_graph, or None."""
    import networkx as nx

    if n0 not in graph or n1 not in graph:
        return None
    try:
        return float(nx.shortest_path_length(graph, n0, n1, weight='weight'))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _percentile_of(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(v) for v in values)
    idx = max(0, int(float(percentile) * (len(ordered) - 1)))
    return ordered[idx]


def _scene_pair_geodesics_and_ratios(
    graph, node_ids: Sequence[str]
) -> tuple[list[float], list[float]]:
    """Full-graph geodesic metres and geo/eucl ratios for unique node pairs.

    One Dijkstra per source. Pairs with no path or zero Euclidean are skipped
    (same filters ``_class4_pair_reject_reason`` applies before the ratio gate).
    """
    import networkx as nx

    uniq = [n for n in dict.fromkeys(node_ids) if n in graph]
    if len(uniq) < 2:
        return [], []
    geos: list[float] = []
    ratios: list[float] = []
    xz: dict[str, Optional[tuple[float, float]]] = {}
    for nid in uniq:
        d = xyz_as_dict(graph.nodes[nid].get('pos'))
        xz[nid] = None if d is None else (d['x'], d['z'])
    for i, src in enumerate(uniq):
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, src, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        p0 = xz[src]
        for dst in uniq[i + 1 :]:
            d = lengths.get(dst)
            if d is None or d <= 1e-9:
                continue
            geo = float(d)
            geos.append(geo)
            p1 = xz[dst]
            if p0 is None or p1 is None:
                continue
            eucl = math.hypot(p0[0] - p1[0], p0[1] - p1[1])
            if eucl > 1e-9:
                ratios.append(geo / eucl)
    return geos, ratios


def calibrate_scene_geodesic_floor_m(
    graph,
    node_ids: Sequence[str],
    *,
    percentile: float = SCENE_PAIR_GEODESIC_PERCENTILE,
) -> Optional[float]:
    """Scene-relative pair-length floor: a percentile of this scene's own spread.

    Same shape as ``calibrate_view_radius_m``. Computed once per episode.
    Returns None when there is no usable distribution (absolute band only).
    """
    geos, _ratios = _scene_pair_geodesics_and_ratios(graph, node_ids)
    if not geos:
        return None
    return _percentile_of(geos, percentile)


def calibrate_scene_ratio_floor(
    graph,
    node_ids: Sequence[str],
    *,
    percentile: float = SCENE_PAIR_GEODESIC_PERCENTILE,
    cap: float = ROUTE_MAX_GEODESIC_EUCLIDEAN_RATIO_CAP,
) -> float:
    """Scene-relative geo/eucl floor, never above ``cap``.

    Same pairs and percentile as ``calibrate_scene_geodesic_floor_m``. An
    open-plan scene whose median detour sits below the Habitat 1.1 cap gets
    that lower, achievable bar instead of zero items.
    """
    _geos, ratios = _scene_pair_geodesics_and_ratios(graph, node_ids)
    if not ratios:
        return float(cap)
    return min(float(cap), _percentile_of(ratios, percentile))


def _scene_class4_floors(
    graph, node_ids: Sequence[str], *, ratio_cap: float
) -> tuple[Optional[float], float]:
    """One Dijkstra pass: (geodesic floor or None, ratio floor capped)."""
    geos, ratios = _scene_pair_geodesics_and_ratios(graph, node_ids)
    geo_floor = (
        _percentile_of(geos, SCENE_PAIR_GEODESIC_PERCENTILE) if geos else None
    )
    if not ratios:
        return geo_floor, float(ratio_cap)
    return geo_floor, min(float(ratio_cap), _percentile_of(ratios, SCENE_PAIR_GEODESIC_PERCENTILE))


def _class4_pair_reject_reason(
    graph,
    src_pos,
    goal_pos,
    n0: str,
    n1: str,
    *,
    min_ratio: Optional[float] = None,
    min_scene_geodesic_m: Optional[float] = None,
) -> Optional[str]:
    """None if the pair passes the 2024 object-goal geodesic band.

    Hard gate: full-graph geodesic in ``[MIN_PAIR_GEODESIC_M, MAX_PAIR_GEODESIC_M]``
    (HM3D-OVON / GOAT-Bench / HSSD-200). ``min_scene_geodesic_m`` adds the
    scene-relative half of that gate (see
    ``calibrate_scene_geodesic_floor_m``); both must pass. ``min_ratio`` is the
    scene-calibrated geo/eucl floor from ``calibrate_scene_ratio_floor``
    (capped at 1.1 for route, 1.05 for survey).
    """
    geo = _full_graph_geodesic_m(graph, n0, n1)
    if geo is None:
        return 'no_full_graph_path'
    if geo < float(MIN_PAIR_GEODESIC_M):
        return 'geodesic_lt_1m'
    if geo > float(MAX_PAIR_GEODESIC_M):
        return 'geodesic_gt_30m'
    if min_scene_geodesic_m is not None and geo < float(min_scene_geodesic_m):
        return 'geodesic_below_scene_median'
    if min_ratio is None:
        return None
    a = xyz_as_dict(src_pos)
    b = xyz_as_dict(goal_pos)
    if a is None or b is None:
        return 'missing_landmark_pos'
    eucl = math.hypot(a['x'] - b['x'], a['z'] - b['z'])
    if eucl < 1e-9:
        return 'euclidean_zero'
    if (geo / eucl) < float(min_ratio):
        return f'geodesic_euclidean_ratio_lt_{float(min_ratio):g}'
    return None


def _format_pair_rejects(rejects: Counter, fallback: str) -> str:
    if not rejects:
        return fallback
    parts = ','.join(f'{k}={v}' for k, v in rejects.most_common())
    return f'{fallback}:{parts}'


def _scene_spacing_positions(episode: dict, landmarks: Sequence[dict]) -> list:
    """Landmark and region poses for view-radius calibration."""
    out: list = []
    for lm in landmarks or []:
        if isinstance(lm, dict) and lm.get('position') is not None:
            out.append(lm['position'])
    layout = episode.get('world_layout') or {}
    for region in layout.get('regions') or []:
        if not isinstance(region, dict):
            continue
        pos = region.get('position') or region.get('centroid') or region.get('center')
        if pos is not None:
            out.append(pos)
    return out


def _serialize_traversed_edges(edges) -> list[list[str]]:
    out: list[list[str]] = []
    for edge in edges or []:
        if not edge or len(edge) != 2:
            continue
        a, b = edge[0], edge[1]
        out.append([a, b] if a <= b else [b, a])
    out.sort()
    return out


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
            nid = snap_landmark_to_graph(
                graph, lm['position'], max_distance_m=max_distance_m
            )
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


def _episode_traversed_subgraph(episode: dict, graph):
    """Snap trajectory once; return (snapped, traversed_nodes, traversed_sub)."""
    from cm_benchmark.generation.nav_graph import (
        snap_trajectory_to_graph,
        subgraph_from_traversed_edges,
        traversed_edges_from_snapped,
        traversed_node_ids,
    )

    snapped = snap_trajectory_to_graph(graph, episode.get('agent_trajectory') or [])
    traversed = traversed_node_ids(snapped)
    edges = traversed_edges_from_snapped(snapped)
    sub = subgraph_from_traversed_edges(graph, edges)
    return snapped, traversed, edges, sub


def plan_route_knowledge(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Retrace an EXPERIENCED path as collected-format actions on traversed_edges."""
    from cm_benchmark.generation.nav_graph import (
        shortest_path,
    )
    import networkx as nx

    graph = _load_nav_graph_or_none(episode)
    if graph is None:
        return [
            PlannedFact(
                construct='route_knowledge',
                status='unsupported',
                reason='missing_or_invalid_nav_graph',
            )
        ]

    _snapped, traversed, edges, walked = _episode_traversed_subgraph(episode, graph)
    if len(traversed) < 3 or walked.number_of_edges() < 1:
        return [
            PlannedFact(
                construct='route_knowledge',
                status='unsupported',
                reason='trajectory_too_short_after_snap',
            )
        ]

    landmarks = select_landmark_candidates(episode, unique_category=False)
    id_to_node, _node_to_name, meta = _build_landmark_node_map(
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
    min_hops = ROUTE_MIN_HOP_COUNT
    traversed_edge_list = _serialize_traversed_edges(edges)
    rejects: Counter = Counter()

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
    route_nodes = [id_to_node[oid] for oid in ordered_ids]
    scene_geodesic_floor, scene_ratio_floor = _scene_class4_floors(
        graph, route_nodes, ratio_cap=ROUTE_MAX_GEODESIC_EUCLIDEAN_RATIO_CAP
    )
    # Walk order enumerates pairs; the emitted subset is ranked by difficulty
    # below. Taking the first passing pairs in walk order yields the walk's
    # nearest neighbours, which are the easiest pairs in the scene.
    candidates: list[tuple[tuple, PlannedFact]] = []
    for i, src_id in enumerate(ordered_ids):
        for goal_id in ordered_ids[i + 1 :]:
            if src_id == goal_id:
                continue
            n0, n1 = id_to_node[src_id], id_to_node[goal_id]
            if n0 == n1:
                rejects['same_snapped_node'] += 1
                continue
            pair_reason = _class4_pair_reject_reason(
                graph,
                meta[src_id]['position'],
                meta[goal_id]['position'],
                n0,
                n1,
                min_ratio=scene_ratio_floor,
                min_scene_geodesic_m=scene_geodesic_floor,
            )
            if pair_reason:
                rejects[pair_reason] += 1
                continue
            source = _referring_display_name(
                episode, int(meta[src_id]['first_seen_step']), src_id
            )
            goal = _referring_display_name(
                episode, int(meta[goal_id]['first_seen_step']), goal_id
            )
            if not source or not goal:
                rejects['unnamed_endpoint'] += 1
                continue
            if source == goal:
                rejects['source_name_eq_goal'] += 1
                continue
            try:
                path = shortest_path(walked, n0, n1)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                rejects['no_walked_path'] += 1
                continue
            if len(path) < 2:
                rejects['empty_path'] += 1
                continue
            hops = len(path) - 1
            if hops < min_hops:
                rejects['hops_lt_2'] += 1
                continue
            if len(path) > ROUTE_MAX_SUBPATH_NODES:
                rejects['path_too_long'] += 1
                continue
            if not _has_real_move_between(
                episode,
                int(meta[src_id]['first_seen_step']),
                max(
                    int(meta[goal_id]['first_seen_step']),
                    int(meta[src_id]['first_seen_step']) + 1,
                ),
            ):
                if len(path) < 3:
                    rejects['no_real_move'] += 1
                    continue

            ref = _reference_nav_actions(
                walked,
                path,
                rot,
                start_heading_deg=_agent_yaw_at_step(
                    episode, int(meta[src_id]['first_seen_step'])
                ),
                verify_edges=edges,
            )
            if ref is None:
                rejects['actions_unusable'] += 1
                continue
            turn_count = int(ref['turn_count'])
            if turn_count < ROUTE_MIN_TURN_COUNT:
                # A straight walk is answerable from a language prior alone.
                rejects[f'turns_lt_{ROUTE_MIN_TURN_COUNT}'] += 1
                continue
            if turn_count > ROUTE_MAX_TURN_COUNT:
                rejects[f'turns_gt_{ROUTE_MAX_TURN_COUNT}'] += 1
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
            geodesic_m = _full_graph_geodesic_m(graph, n0, n1) or 0.0
            fact = (
                PlannedFact(
                    construct='route_knowledge',
                    status='ok',
                    query_step=t1,
                    encoding_step=t0,
                    answer_label=ref['answer_label'],
                    answer_source=[
                        f'nav_graph.traversed_edges.shortest_path[{n0}→{n1}]',
                        'follow_path_actions(traversed_subgraph)',
                    ],
                    image_paths=images,
                    extra={
                        'source': source,
                        'goal': goal,
                        'A': source,
                        'B': goal,
                        'source_landmark_id': src_id,
                        'goal_landmark_id': goal_id,
                        'source_mark': _object_mark_at(
                            episode,
                            int(meta[src_id]['first_seen_step']),
                            src_id,
                        ),
                        'goal_mark': _object_mark_at(
                            episode,
                            int(meta[goal_id]['first_seen_step']),
                            goal_id,
                        ),
                        'source_node': n0,
                        'goal_node': n1,
                        'path_nodes': path,
                        'min_hop_count': min_hops,
                        'hop_count': hops,
                        'turn_count': turn_count,
                        'geodesic_m': round(geodesic_m, 3),
                        'action_sequence': ref['actions'],
                        'start_heading_deg': ref['start_heading_deg'],
                        'traversed_edges': traversed_edge_list,
                        'answer_format': 'action_sequence',
                        'scoring': 'success_validity_efficiency',
                        'graph_scope': 'traversed',
                        'object_type': goal,
                        'frame_of_reference': 'egocentric',
                        'image_roles': image_roles,
                        'template_index': pick_template_index(
                            'route_knowledge', None, src_id, goal_id
                        ),
                    },
                )
            )
            # Harder first: more real turns, then longer geodesic.
            candidates.append(((turn_count, geodesic_m, hops), fact))

    if not candidates:
        return [
            PlannedFact(
                construct='route_knowledge',
                status='unsupported',
                reason=_format_pair_rejects(
                    rejects, 'no_experienced_landmark_to_landmark_walk'
                ),
                extra={'pair_reject_counts': dict(rejects)},
            )
        ]
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [fact for _rank, fact in candidates[:max_items]]


def plan_survey_based_route_planning(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Plan a never-walked source→goal as collected-format actions on viewed_edges.

    Pair gates use full-nav_graph geodesic, but the reference path and its
    actions live on the viewed subgraph, because that is what validity scores.
    """
    from cm_benchmark.generation.nav_graph import (
        calibrate_view_radius_m,
        path_exists,
        sanitize_world_layout,
        shortest_path,
        subgraph_from_traversed_edges,
        viewed_edges_from_trajectory,
    )

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

    _snapped, _traversed, edges, walked = _episode_traversed_subgraph(episode, graph)
    landmarks = select_landmark_candidates(episode, unique_category=False)
    id_to_node, _node_to_name, meta = _build_landmark_node_map(
        graph, landmarks, max_distance_m=ROUTE_LANDMARK_SNAP_M
    )
    ids = [lm['obj_id'] for lm in landmarks if lm['obj_id'] in id_to_node]
    if len(ids) < 2:
        return [
            PlannedFact(
                construct='survey_based_route_planning',
                status='unsupported',
                reason='need_ge2_distinguishable_landmarks_snapped',
            )
        ]

    rot = _rotation_deg(episode)
    view_radius = calibrate_view_radius_m(
        graph, _scene_spacing_positions(episode, [meta[oid] for oid in ids])
    )
    viewed_edges, _viewed_nodes = viewed_edges_from_trajectory(
        graph,
        episode.get('agent_trajectory') or [],
        view_radius,
        traversed_edges=edges,
    )
    # Validity is scored on viewed_edges, so the reference must be planned there
    # too: a full-graph shortest path can leave the viewed corridor and would be
    # stored as an answer that the [CODE] scorer marks invalid.
    viewed_sub = subgraph_from_traversed_edges(graph, viewed_edges)
    viewed_edge_list = _serialize_traversed_edges(viewed_edges)
    traversed_edge_list = _serialize_traversed_edges(edges)
    survey_nodes = [id_to_node[oid] for oid in ids]
    scene_geodesic_floor, scene_ratio_floor = _scene_class4_floors(
        graph, survey_nodes, ratio_cap=SURVEY_MIN_GEODESIC_EUCLIDEAN_RATIO
    )
    # Same as route: collect every valid pair, then emit the hardest. Returning
    # at the first max_items hits takes them in `ids` order, which is salience,
    # not difficulty.
    candidates: list[tuple[tuple, PlannedFact]] = []
    rejects: Counter = Counter()

    for i, src_id in enumerate(ids):
        for goal_id in ids[i + 1 :]:
            n0, n1 = id_to_node[src_id], id_to_node[goal_id]
            if n0 == n1:
                rejects['same_snapped_node'] += 1
                continue
            # Never-traversed: no path on the walked-edge subgraph.
            if path_exists(walked, n0, n1):
                rejects['path_on_traversed'] += 1
                continue
            # Visual grounding: a path on the same viewed_edges subgraph scoring uses.
            if not path_exists(viewed_sub, n0, n1):
                rejects['no_viewed_path'] += 1
                continue
            pair_reason = _class4_pair_reject_reason(
                graph,
                meta[src_id]['position'],
                meta[goal_id]['position'],
                n0,
                n1,
                min_ratio=scene_ratio_floor,
                min_scene_geodesic_m=scene_geodesic_floor,
            )
            if pair_reason:
                rejects[pair_reason] += 1
                continue
            path = shortest_path(viewed_sub, n0, n1)
            if len(path) < 2:
                rejects['empty_path'] += 1
                continue
            source = _referring_display_name(
                episode, int(meta[src_id]['first_seen_step']), src_id
            )
            goal = _referring_display_name(
                episode, int(meta[goal_id]['first_seen_step']), goal_id
            )
            if not source or not goal:
                rejects['unnamed_endpoint'] += 1
                continue
            if source == goal:
                rejects['source_name_eq_goal'] += 1
                continue
            t0 = int(meta[src_id]['first_seen_step'])
            t1 = int(meta[goal_id]['first_seen_step'])
            ref = _reference_nav_actions(
                viewed_sub,
                path,
                rot,
                start_heading_deg=_agent_yaw_at_step(episode, t0),
                verify_edges=viewed_edges,
            )
            if ref is None:
                rejects['actions_unusable'] += 1
                continue
            if _agent_near_landmark(episode, meta[src_id]['position'], [t0]):
                rejects['agent_near_source'] += 1
                continue
            images, image_roles = merge_role_images(
                [
                    (
                        _img(step_by_index(episode, t0)),
                        f'source sighted · {source}',
                    ),
                    (
                        _img(step_by_index(episode, t1)),
                        f'goal sighted · {goal}',
                    ),
                ]
            )
            hops = len(path) - 1
            turn_count = int(ref['turn_count'])
            geodesic_m = _full_graph_geodesic_m(graph, n0, n1) or 0.0
            fact = (
                PlannedFact(
                    construct='survey_based_route_planning',
                    status='ok',
                    query_step=max(t0, t1),
                    encoding_step=min(t0, t1),
                    answer_label=ref['answer_label'],
                    answer_source=[
                        f'nav_graph.viewed_edges.shortest_path[{n0}→{n1}]',
                        'follow_path_actions(viewed_subgraph)',
                        f'nav_graph.traversed_edges.no_path[{n0}→{n1}]',
                        f'nav_graph.viewed_edges.radius={view_radius:.3f}',
                    ],
                    image_paths=images,
                    extra={
                        'source': source,
                        'goal': goal,
                        'A': source,
                        'B': goal,
                        'template_index': pick_template_index(
                            'survey_based_route_planning', None, src_id, goal_id
                        ),
                        'source_landmark_id': src_id,
                        'goal_landmark_id': goal_id,
                        'source_mark': _object_mark_at(episode, t0, src_id),
                        'goal_mark': _object_mark_at(episode, t1, goal_id),
                        'source_node': n0,
                        'goal_node': n1,
                        'path_nodes': path,
                        'hop_count': hops,
                        'turn_count': turn_count,
                        'geodesic_m': round(geodesic_m, 3),
                        'action_sequence': ref['actions'],
                        'start_heading_deg': ref['start_heading_deg'],
                        'traversed_edges': traversed_edge_list,
                        'viewed_edges': viewed_edge_list,
                        'view_radius_m': view_radius,
                        'answer_format': 'action_sequence',
                        'scoring': 'success_validity_efficiency',
                        'graph_scope': 'viewed',
                        'object_type': goal,
                        'frame_of_reference': 'allocentric',
                        'image_roles': image_roles,
                    },
                )
            )
            # Harder first: more real turns, then longer geodesic.
            candidates.append(((turn_count, geodesic_m, hops), fact))

    if not candidates:
        return [
            PlannedFact(
                construct='survey_based_route_planning',
                status='unsupported',
                reason=_format_pair_rejects(
                    rejects, 'no_untraversed_viewed_landmark_pair'
                ),
                extra={'pair_reject_counts': dict(rejects)},
            )
        ]
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [fact for _rank, fact in candidates[:max_items]]


def _co_visible_distinguishable_step(
    episode: dict, obj_ids: Sequence[str]
) -> Optional[int]:
    """Latest step where every obj_id is QUERY-FOV distinguishable."""
    best = None
    for step in episode.get('steps') or []:
        si = int(step['step'])
        if all(
            _distinguishable_encoding_sighting(episode, si, oid) for oid in obj_ids
        ):
            best = si
    return best


def plan_perspective_taking(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Object Perspective / Spatial Orientation Test: stand at A facing B, locate C.

    A, B, and C must each be FOV-distinguishable and uniquely nameable.
    ``select_landmark_candidates`` keeps unique-category landmarks here
    (default); class-4 is the construct that allows referring phrases for
    duplicates. Prefer a frame where all three are co-visible; otherwise use
    each landmark's clear sighting frame.
    """
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

                sa = int(a['first_seen_step'])
                sb = int(b['first_seen_step'])
                sc = int(c['first_seen_step'])
                name_a = _referring_display_name(episode, sa, a['obj_id'])
                name_b = _referring_display_name(episode, sb, b['obj_id'])
                name_c = _referring_display_name(episode, sc, c['obj_id'])
                if name_a is None or name_b is None or name_c is None:
                    continue

                label = imagined_perspective_label(pos_a, pos_b, pos_c)
                if not label or label not in EGO_DIRECTION_OPTIONS:
                    continue

                co_step = _co_visible_distinguishable_step(
                    episode, [a['obj_id'], b['obj_id'], c['obj_id']]
                )
                if co_step is not None:
                    # Re-resolve names at the co-visible frame (same ids).
                    name_a = _referring_display_name(episode, co_step, a['obj_id']) or name_a
                    name_b = _referring_display_name(episode, co_step, b['obj_id']) or name_b
                    name_c = _referring_display_name(episode, co_step, c['obj_id']) or name_c
                    query_step = co_step
                    encoding_step = co_step
                    images, image_roles = merge_role_images(
                        [
                            (
                                _img(step_by_index(episode, co_step)),
                                f'A/B/C co-visible · stand at {name_a}, face {name_b}, locate {name_c}',
                            )
                        ]
                    )
                else:
                    # Each mentioned landmark must be distinguishable in its own frame.
                    if not (
                        _distinguishable_encoding_sighting(episode, sa, a['obj_id'])
                        and _distinguishable_encoding_sighting(episode, sb, b['obj_id'])
                        and _distinguishable_encoding_sighting(episode, sc, c['obj_id'])
                    ):
                        continue
                    query_step = max(sa, sb, sc)
                    encoding_step = min(sa, sb, sc)
                    images, image_roles = merge_role_images(
                        [
                            (
                                _img(step_by_index(episode, sa)),
                                f'A · stand here ({name_a})',
                            ),
                            (
                                _img(step_by_index(episode, sb)),
                                f'B · face toward ({name_b})',
                            ),
                            (
                                _img(step_by_index(episode, sc)),
                                f'C · locate ({name_c})',
                            ),
                        ]
                    )
                if not images:
                    continue

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
                wrong_b = {
                    'x': pos_a['x'] - (pos_b['x'] - pos_a['x']),
                    'y': pos_a['y'],
                    'z': pos_a['z'] - (pos_b['z'] - pos_a['z']),
                }
                wrong = imagined_perspective_label(pos_a, wrong_b, pos_c)
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
                out.append(
                    PlannedFact(
                        construct='perspective_taking',
                        status='ok',
                        query_step=query_step,
                        encoding_step=encoding_step,
                        queried_object_id=c['obj_id'],
                        reference_object_id=a['obj_id'],
                        answer_label=label,
                        answer_source=[
                            f'landmarks[{a["obj_id"]}].position (A)',
                            f'landmarks[{b["obj_id"]}].position (B)',
                            f'landmarks[{c["obj_id"]}].position (C)',
                            'imagined_perspective_label(A, A→B, C; signed angle)',
                        ],
                        image_paths=images,
                        options_pool=pool[:4],
                        distractor_seeds=seeds,
                        extra={
                            'A': name_a,
                            'B': name_b,
                            'C': name_c,
                            'source': name_a,
                            'goal': name_b,
                            'object_type': name_c,
                            'landmark_a_id': a['obj_id'],
                            'landmark_b_id': b['obj_id'],
                            'landmark_c_id': c['obj_id'],
                            'perspective_shift_magnitude': shift,
                            'frame_of_reference': 'allocentric',
                            'image_roles': image_roles,
                            'abc_co_visible': co_step is not None,
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
    visibility_model_path: Optional[str] = None,
) -> list[PlannedFact]:
    from cm_benchmark.generator.visibility_filters import (
        apply_question_visibility_to_episode,
    )

    # Prefer DecisionTree joblib when present; else static question_visibility.
    # Also attaches the model for distinguishability gates during planning.
    episode = apply_question_visibility_to_episode(
        episode,
        inplace=False,
        model_path=visibility_model_path,
    )

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
