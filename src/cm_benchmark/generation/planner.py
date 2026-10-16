"""Task planner: select construct-eligible facts from episode GT (deterministic)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

import networkx as nx

from cm_benchmark.generation.config import get_generation_config, legacy_constant
from cm_benchmark.generation.disambiguation import (
    active_margin_m,
    aggregate_disambiguation,
    object_aabb_gap_m,
    passing_objects_in_frames,
    resolve_named_objects,
)
from cm_benchmark.generation.synsets import (
    confusable_components,
    group_is_unique,
    objects_are_confusable,
    type_of,
)
from cm_benchmark.generation.constructs import (
    EGO_DIRECTION_OPTIONS,
    MIRRORED_LR,
    OPPOSITE,
    ORTHOGONAL,
    camera_size_wh,
    find_ego_edge,
    fov_metrics_ok,
    is_fov_step,
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
    point_in_camera_frustum,
)
from cm_benchmark.generation.nav_graph import (
    calibrate_view_radius_m,
    continuous_bfs_actions,
    continuous_bfs_to_goals,
    count_direction_changes,
    episode_motion,
    follow_path_actions,
    format_nav_actions,
    node_within_goal_tolerance,
    path_exists,
    path_start_heading_deg,
    path_to_nav_actions,
    sanitize_world_layout,
    score_route_action_sequence,
    score_survey_action_sequence,
    shortest_path,
    simulate_nav_action_sequence,
    snap_landmark_to_graph,
    snap_to_nearest_of,
    snap_trajectory_to_graph,
    subgraph_from_traversed_edges,
    trajectory_snap_tolerance,
    traversed_edges_from_snapped,
    traversed_node_ids,
    viewed_edges_from_trajectory,
    xz_reaches_goal,
)
from cm_benchmark.generation.route_replay import recorded_actions_between
from cm_benchmark.generator.visibility_filters import (
    apply_question_visibility_to_episode,
    visibility_model_from_episode,
)


def _cfg():
    """Active generation config (quality numbers)."""
    return get_generation_config()


def _id_encode_fov() -> dict[str, float]:
    """Soft FOV kwargs for invisible-displacement encode when no DecisionTree is loaded."""
    return _cfg().visibility.id_encode.as_kwargs()


def _n_options() -> int:
    return _cfg().options.n_options


@dataclass
class PlannedFact:
    """Locked spatial fact for one item (before wording)."""

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
        """Replace None list/dict fields with fresh empty containers."""
        if self.image_paths is None:
            self.image_paths = []
        if self.options_pool is None:
            self.options_pool = []
        if self.distractor_seeds is None:
            self.distractor_seeds = []
        if self.extra is None:
            self.extra = {}


def _img(step: Optional[dict]) -> list[str]:
    """The step's frame path as a 0/1-element list (lets callers concatenate)."""
    p = (step or {}).get('image_path')
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


def _step_for_image_path(episode: dict, path: str) -> Optional[int]:
    """Step index whose ``image_path`` is ``path``, or None."""
    for step in episode.get('steps') or []:
        if step.get('image_path') == path:
            return int(step['step'])
    return None


def _named_ids_for_fact(episode: dict, fact: PlannedFact, frames: list[int]) -> list[str]:
    """Object ids the question names (queried, landmarks, ID dest / partner)."""
    extra = fact.extra or {}
    ids: list[str] = []

    def add(oid):
        if oid and oid not in ids:
            ids.append(oid)

    add(fact.queried_object_id)
    for key in (
        'swap_partner_id',
        'floor_anchor_id',
        'destination_landmark_id',
        'landmark_a_id',
        'landmark_b_id',
        'landmark_c_id',
        'source_landmark_id',
        'goal_landmark_id',
    ):
        add(extra.get(key))
    to_r = extra.get('to_receptacle')
    if to_r and not _is_floor_receptacle(to_r) and not extra.get('destination_landmark_id'):
        for si in frames:
            step = step_by_index(episode, si)
            if not step:
                continue
            oid = _visible_oid_matching(step, to_r)
            if oid:
                add(oid)
                extra['destination_landmark_id'] = oid
                break
    return ids


def _named_ids_confusable(episode: dict, obj_ids: Sequence[str]) -> bool:
    """True when two different named objects are confusable under the synset rule."""
    ids = [oid for oid in dict.fromkeys(obj_ids) if oid]
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            if objects_are_confusable(episode, left, right):
                return True
    return False


_PASSAGE_STEMS = frozenset({'door', 'doorway', 'doorframe'})


def _is_passage_landmark(obj_id: str, category: Optional[str] = None) -> bool:
    """Doors and doorways. SPOC ``salient_landmarks`` does not use them as endpoints."""
    stem = (category or str(obj_id).split('|')[0]).strip().lower()
    if stem in _PASSAGE_STEMS:
        return True
    lowered = str(obj_id).lower()
    return lowered.startswith(('door|', 'doorway|', 'doorframe|'))


def _salient_floor_landmark_ids(episode: dict) -> Optional[set[str]]:
    """Floor-standing endpoint ids from ``world_layout`` landmarks.

    That list is the collector's ``salient_landmarks`` set: non-pickupable
    objects whose box sits on the floor. Passages are removed. ``None`` means
    the episode has no layout list, so callers keep every non-passage landmark
    (unit fixtures).
    """
    layout = (episode.get('world_layout') or {}).get('landmarks') or []
    if not layout:
        return None
    ids: set[str] = set()
    for lm in layout:
        if not isinstance(lm, dict):
            continue
        oid = lm.get('landmark_id') or lm.get('obj_id')
        if not oid:
            continue
        category = lm.get('obj-type') or lm.get('category')
        if _is_passage_landmark(str(oid), None if category is None else str(category)):
            continue
        ids.add(str(oid))
    return ids


def _rewrite_fact_names(fact: PlannedFact, names: dict) -> None:
    """Overwrite display phrases on ``fact.extra`` from whole-item results."""
    extra = fact.extra or {}
    qid = fact.queried_object_id
    if extra.get('landmark_a_id') in names:
        extra['A'] = names[extra['landmark_a_id']].display_name
        extra['source'] = extra['A']
    if extra.get('landmark_b_id') in names:
        extra['B'] = names[extra['landmark_b_id']].display_name
        extra['goal'] = extra['B']
    if extra.get('landmark_c_id') in names:
        extra['C'] = names[extra['landmark_c_id']].display_name
        extra['object_type'] = extra['C']
    if extra.get('source_landmark_id') in names:
        extra['source'] = names[extra['source_landmark_id']].display_name
        extra['A'] = extra['source']
    if extra.get('goal_landmark_id') in names:
        extra['goal'] = names[extra['goal_landmark_id']].display_name
        extra['B'] = extra['goal']
        extra['object_type'] = extra['goal']
    if extra.get('swap_partner_id') in names:
        extra['other_object_type'] = names[extra['swap_partner_id']].display_name
    if extra.get('floor_anchor_id') in names:
        extra['floor_anchor_landmark'] = names[extra['floor_anchor_id']].display_name
        extra['new_location'] = humanize_receptacle(
            extra.get('to_receptacle'),
            floor_anchor_landmark=extra['floor_anchor_landmark'],
        )
    elif extra.get('destination_landmark_id') in names:
        extra['new_location'] = names[extra['destination_landmark_id']].display_name
    if qid in names and fact.construct not in (
        'perspective_taking',
        'route_knowledge',
        'survey_based_route_planning',
    ):
        extra['disambiguator'] = names[qid].phrase
        extra['object_type'] = names[qid].object_type


def _relabel_fact(episode: dict, fact: PlannedFact) -> Optional[PlannedFact]:
    """Re-resolve every named object across the frames the item actually shows."""
    extra = fact.extra if fact.extra is not None else {}
    fact.extra = extra
    frames = [int(s) for s in (extra.get('shown_frame_steps') or [])]
    if not frames:
        frames = [
            si
            for p in (fact.image_paths or [])
            if (si := _step_for_image_path(episode, p)) is not None
        ]
    extra['shown_frame_steps'] = frames
    ids = _named_ids_for_fact(episode, fact, frames)
    if not ids:
        extra.setdefault('disambiguation', 'none')
        extra.setdefault('disambiguation_landmark', None)
        extra.setdefault('marks', {})
        return fact
    names = resolve_named_objects(episode, frames or [0], ids)
    if names is None:
        return None
    extra.update(aggregate_disambiguation(names))
    _rewrite_fact_names(fact, names)
    return fact


def _apply_whole_item_names(episode: dict, facts: list[PlannedFact]) -> list[PlannedFact]:
    """Drop facts that cannot be named across their shown frames."""
    by_c: dict[str, list[PlannedFact]] = {}
    order: list[str] = []
    for fact in facts:
        if fact.construct not in by_c:
            order.append(fact.construct)
            by_c[fact.construct] = []
        by_c[fact.construct].append(fact)
    out: list[PlannedFact] = []
    for construct in order:
        group = by_c[construct]
        kept: list[PlannedFact] = []
        had_ok = False
        dropped_confusable = False
        for fact in group:
            if fact.status != 'ok':
                kept.append(fact)
                continue
            had_ok = True
            frames = [int(s) for s in ((fact.extra or {}).get('shown_frame_steps') or [])]
            named_ids = _named_ids_for_fact(episode, fact, frames)
            if _named_ids_confusable(episode, named_ids):
                dropped_confusable = True
                continue
            updated = _relabel_fact(episode, fact)
            if updated is not None:
                kept.append(updated)
        if had_ok and not any(f.status == 'ok' for f in kept):
            kept.append(
                PlannedFact(
                    construct=construct,
                    status='unsupported',
                    reason=(
                        'confusable_named_objects'
                        if dropped_confusable
                        else 'duplicate_unnameable_across_shown_frames'
                    ),
                )
            )
        out.extend(kept)
    return out


def _images_between(episode: dict, start_step: int, end_step: int) -> list[str]:
    """Image paths for the inclusive step window, in navigation order.

    Intermediate frames that share the previous kept frame's horizontal agent
    position are dropped (rotate-in-place / look-only). Encoding and query
    endpoints are always kept when present.
    """
    window: list[tuple[int, str]] = []
    for step in episode.get('steps') or []:
        step_idx = int(step['step'])
        if (
            start_step <= step_idx <= end_step
            and step.get('image_path')
            and is_fov_step(episode, step_idx)
        ):
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


def validate_delay_window(name: str, min_delay: int, max_delay: Optional[int]) -> None:
    """Raise ValueError unless ``1 <= min_delay <= max_delay`` (max may be None).

    Shared by the planners and the CLI so both reject the same windows.
    """
    if min_delay < 1:
        raise ValueError(f'{name} min_delay must be at least 1')
    if max_delay is not None and max_delay < min_delay:
        raise ValueError(f'{name} max_delay must be greater than or equal to min_delay')


def _delay_is_allowed(k: int, min_delay: int, max_delay: Optional[int]) -> bool:
    """True if a delay of ``k`` steps lies in ``[min_delay, max_delay]``."""
    return k >= min_delay and (max_delay is None or k <= max_delay)


def _displaced_ids(episode: dict) -> set[str]:
    """Object ids moved by any displacement event (excluded from static-object items)."""
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
    n = _n_options()
    for filler in EGO_DIRECTION_OPTIONS:
        if len(pool) >= n:
            break
        if filler not in pool:
            pool.append(filler)
    return pool[:n], seeds


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
    eps: Optional[float] = None,
) -> bool:
    """True if two (x, z) positions differ by more than ``eps``; False if either is missing."""
    if eps is None:
        eps = _cfg().motion.position_eps_m
    return a is not None and b is not None and (abs(a[0] - b[0]) > eps or abs(a[1] - b[1]) > eps)


def _has_real_move_between(
    episode: dict, t0: int, t1: int, *, eps: Optional[float] = None
) -> bool:
    """True if the agent translates on the floor plane between t0 and t1 (inclusive end).

    Action names alone are not enough: rotate/look/turn in place leave position
    unchanged and are not treated as navigation for this benchmark.
    """
    t0, t1 = int(t0), int(t1)
    start = _agent_horizontal_pos(episode, t0) if t1 > t0 else None
    if start is None:
        return False
    later = [t1] + [
        int(s['step']) for s in episode.get('steps') or [] if t0 < int(s['step']) <= t1
    ]
    return any(
        _horizontal_pos_differs(start, _agent_horizontal_pos(episode, si), eps=eps)
        for si in later
    )


def _referring_disambiguator(
    episode: dict, step_idx: int, obj_id: str
) -> Optional[str]:
    """'' if unique; phrase if duplicate+landmark; None to skip candidate."""
    step = step_by_index(episode, int(step_idx))
    if step is None:
        return None
    return resolve_referring_disambiguator(
        step,
        obj_id,
        _agent_pose_dict(episode, int(step_idx)),
        model=visibility_model_from_episode(episode),
        episode=episode,
    )


def _display_name_parts(
    episode: dict, step_idx: int, obj_id: str, **fov
) -> Optional[tuple[str, str]]:
    """``(object_type, disambiguator)`` for an object seen clearly at ``step_idx``.

    None when the sighting is not distinguishable (``fov`` overrides the
    QUERY floors) or when a duplicate category has no unique referring phrase.
    """
    if not _distinguishable_encoding_sighting(episode, step_idx, obj_id, **fov):
        return None
    phrase = _referring_disambiguator(episode, step_idx, obj_id)
    if phrase is None:
        return None
    step = step_by_index(episode, int(step_idx))
    return object_type_from_id(obj_id, (step or {}).get('visible_objects') or {}), phrase


def _referring_display_name(
    episode: dict,
    step_idx: int,
    obj_id: str,
    *,
    min_bbox_area: Optional[float] = None,
    min_side: Optional[float] = None,
    min_visible_pixels: Optional[float] = None,
) -> Optional[str]:
    """Human name for questions: ``Chair`` or ``Chair close to the Table``.

    Requires a distinguishable FOV sighting at ``step_idx``. Returns None when the
    object is weak/missing or a duplicate category cannot be uniquely referred.
    """
    parts = _display_name_parts(
        episode,
        step_idx,
        obj_id,
        min_bbox_area=min_bbox_area,
        min_side=min_side,
        min_visible_pixels=min_visible_pixels,
    )
    return None if parts is None else ''.join(parts)


def _distinguishable_encoding_sighting(
    episode: dict,
    step_idx: int,
    obj_id: str,
    *,
    min_bbox_area: Optional[float] = None,
    min_side: Optional[float] = None,
    min_visible_pixels: Optional[float] = None,
) -> bool:
    """Object must be clearly visible at encode (metrics + ego edge), not just listed.

    When a DecisionTree visibility model is attached on the episode, that model
    decides distinguishability (static ``min_*`` floors are ignored). Otherwise
    the static floors apply. Thin border-clipped scrapes are always rejected
    when ``episode_meta.camera`` size is available.
    """
    if not is_fov_step(episode, step_idx):
        return False
    step = step_by_index(episode, int(step_idx))
    if step is None:
        return False
    return fov_metrics_ok(
        (step.get('visible_objects') or {}).get(obj_id),
        min_bbox_area=min_bbox_area,
        min_side=min_side,
        min_visible_pixels=min_visible_pixels,
        image_wh=camera_size_wh(episode),
        model=visibility_model_from_episode(episode),
    ) and find_ego_edge(step, obj_id) is not None




def _finite_xyz(value):
    """``(x, y, z)`` floats if all three are finite, else None."""
    if value is None:
        return None
    try:
        xyz = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        return None
    if any(v != v or v in (float('inf'), float('-inf')) for v in xyz):
        return None
    return xyz


def _ego_label_at(
    episode: dict,
    step_idx: int,
    obj_id: str,
    *,
    obj_pos=None,
    ahead_half_width: Optional[float] = None,
) -> Optional[str]:
    """Ego MC label from yaw-only bearing of the object center.

    Prefer the catalog AABB center (``bbox_center``) over the mesh pivot when
    the object is in view. ``ang_width_deg`` refuses a label whose footprint
    crosses a sector boundary. An explicit ``obj_pos`` is the fallback for
    hidden / displaced poses that have no current center.
    """
    ag_pos, ag_rot = agent_pose_at_step(episode, int(step_idx))
    step = step_by_index(episode, int(step_idx))
    odata = {}
    if step is not None:
        odata = (step.get('visible_objects') or {}).get(obj_id) or {}
    pos = _finite_xyz(odata.get('bbox_center'))
    if pos is None:
        pos = obj_pos if obj_pos is not None else odata.get('position')
    if pos is None and step is not None:
        mem = (step.get('non_visible_objects') or {}).get(obj_id) or {}
        pos = mem.get('position') or (mem.get('last_known') or {}).get('position')
    if pos is None:
        pos = _object_world_pos(episode, obj_id)
    ang_width = odata.get('ang_width_deg')
    return ego_label_from_world_pose(
        ag_pos,
        ag_rot,
        pos,
        episode,
        ahead_half_width=ahead_half_width,
        ang_width_deg=ang_width,
    )


def plan_egocentric_encoding(episode: dict, max_items: int = 3) -> list[PlannedFact]:
    """Class 1: bearing of an object visible in the current frame.

    Walks steps in order and emits the first ``max_items`` (step, object) pairs
    whose object is distinguishable, has an ego edge, gets a 20° FOV-wedge label
    and can be named uniquely. Answer = ego label at that pose.
    """
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
                ahead_half_width=_cfg().directions.ahead_half_width_fov_deg,
            )
            if not label or label not in EGO_DIRECTION_OPTIONS:
                continue
            disambiguator = _referring_disambiguator(episode, step_idx, obj_id)
            if disambiguator is None:
                continue
            pool, seeds = _ego_pool_with_diagnostics(label)
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


def _ids_seen_distinguishable(episode: dict, up_to_step: int) -> set[str]:
    """Distinct ids with a distinguishable FOV sighting ≤ ``up_to_step``."""
    seen: set[str] = set()
    for step in episode.get('steps') or []:
        si = int(step['step'])
        if si > int(up_to_step):
            continue
        for oid in step.get('visible_objects') or {}:
            if oid not in seen and _distinguishable_encoding_sighting(episode, si, oid):
                seen.add(oid)
    return seen


def _count_category_seen(episode: dict, category: str, up_to_step: int) -> int:
    """Count distinct ids confusable with ``category`` seen ≤ up_to_step."""
    seen = _ids_seen_distinguishable(episode, up_to_step)
    seed = next((oid for oid in seen if type_of(episode, oid) == category), None)
    if seed is None:
        return 0
    return len([oid for oid in seen if objects_are_confusable(episode, seed, oid)])


def _first_step_with_category(episode: dict, category: str) -> int:
    """First FOV step listing any object confusable with ``category`` (0 if none)."""
    for step in episode.get('steps') or []:
        if not is_fov_step(episode, step.get('step')):
            continue
        for oid, odata in (step.get('visible_objects') or {}).items():
            if type_of(episode, oid, odata) == category:
                return int(step['step'])
    return 0


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
    validate_delay_window('SWM', min_delay, max_delay)

    displaced = _displaced_ids(episode)
    out: list[PlannedFact] = []
    # Taxonomy lists both relation and count. Relation used to fill ``max_items``
    # and return early, so recall_count never ran. Reserve count slots when
    # the budget allows more than one item.
    count_budget = (
        0
        if max_items < 2
        else max(1, max_items // _cfg().spatial_working_memory.count_divisor)
    )
    relation_budget = max_items - count_budget

    for step in episode.get('steps') or []:
        if len(out) >= relation_budget:
            break
        step_idx = int(step['step'])
        if not is_fov_step(episode, step_idx):
            continue
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
                ahead_half_width=_cfg().directions.ahead_half_width_fov_deg,
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
                    ahead_half_width=_cfg().directions.ahead_half_width_fov_deg,
                )
                if lab and lab != label:
                    current_view_lab = lab
                    break

            pool, seeds = _ego_pool_with_diagnostics(
                label, current_view_lab or ''
            )
            if current_view_lab:
                seeds.append(_mode_seed('current_view_answer', current_view_lab))
                seeds.append('current_view_answer')

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
            if step_idx < min_delay or not is_fov_step(episode, step_idx):
                continue
            seen_ids = _ids_seen_distinguishable(episode, step_idx)
            categories = {type_of(episode, oid) for oid in seen_ids}
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
                if n < _cfg().spatial_working_memory.count_min_instances or cat in seen_cats:
                    continue
                # n >= 2, so the four counts are distinct and non-negative.
                answer, off1, off2 = str(n), str(n - 1), str(n + 1)
                pool = [answer, off1, off2, str(n + 2)]
                seeds = [
                    'off_by_one_count',
                    _mode_seed('off_by_one_count', off1),
                    _mode_seed('off_by_one_count', off2),
                ]
                enc_idx = _first_step_with_category(episode, cat)
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
                        options_pool=pool,
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
    """True for a missing receptacle or a structural one (floor/wall/ceiling/room)."""
    return not receptacle_id or str(receptacle_id).split('|')[0].strip().lower() in (
        'floor', 'wall', 'ceiling', 'room'
    )


def _agent_near_landmark(
    episode: dict,
    landmark_pos,
    steps: Sequence[Optional[int]],
    *,
    min_dist_m: Optional[float] = None,
) -> bool:
    """True if agent XZ is within ``min_dist_m`` of the landmark at any step."""
    if min_dist_m is None:
        min_dist_m = _cfg().class4.survey.min_agent_source_dist_m
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
    """Floor-plane ``(x, z)`` of a list/tuple/dict position, else None."""
    d = xyz_as_dict(pos)
    return None if d is None else (d['x'], d['z'])


def _nearest_floor_anchor(
    episode: dict,
    final_pos,
    frame_steps: list[int],
    *,
    radius_m: Optional[float] = None,
) -> Optional[tuple[str, str]]:
    """Nearest distinguishable landmark within radius of a Floor destination.

    Nearest by AABB-gap, uniquely so by the active proximity margin. Naming
    uses ``resolve_named_objects`` (proximity, then mark); confusable
    landmarks are not skipped. ``radius_m`` comes from the placement CSV
    when present.
    """
    if radius_m is None:
        radius_m = _cfg().invisible_displacement.floor_anchor_radius_m
    target = _xz_from_any(final_pos)
    if target is None:
        return None
    dest_data = {'aabb_min': (target[0], 0.0, target[1]), 'aabb_max': (target[0], 0.0, target[1])}
    scored: list[tuple[float, str, str]] = []
    seen: set[str] = set()
    for si in frame_steps:
        step = step_by_index(episode, si)
        if not step:
            continue
        for oid, odata in (step.get('visible_objects') or {}).items():
            if oid in seen or _is_floor_receptacle(oid):
                continue
            if not _distinguishable_encoding_sighting(episode, si, oid):
                continue
            gap = object_aabb_gap_m(episode, oid, odata, '_dest', dest_data)
            if gap is None or gap > float(radius_m):
                continue
            seen.add(oid)
            scored.append((gap, oid, object_type_from_id(oid)))
    if not scored:
        return None
    scored.sort(key=lambda row: (row[0], row[1]))
    best_gap, best_id, _best_name = scored[0]
    margin_m, ratio = active_margin_m(episode)
    if len(scored) > 1:
        next_gap = scored[1][0]
        if next_gap - best_gap < float(margin_m):
            return None
        if ratio is not None and next_gap < best_gap * float(ratio):
            return None
    names = resolve_named_objects(episode, frame_steps, [best_id])
    if names is None or best_id not in names:
        return None
    return best_id, names[best_id].display_name


def _landmark_matches(visible_id: str, landmark_id: str) -> bool:
    """Same id, or same id stem (``CounterTop|1|2`` matches ``CounterTop|9``)."""
    return visible_id == landmark_id or (
        str(visible_id).split('|')[0] == str(landmark_id).split('|')[0]
    )


def _visible_oid_matching(step: dict, landmark_id: str) -> Optional[str]:
    """First visible id that ``_landmark_matches`` ``landmark_id``, else None."""
    return next(
        (oid for oid in step.get('visible_objects') or {} if _landmark_matches(oid, landmark_id)),
        None,
    )


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


def _rot3(rot) -> Optional[tuple[float, float, float]]:
    """``(x, y, z)`` Euler degrees from a dict or 3-sequence, else None."""
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
    min_bbox_area: Optional[float] = None,
    min_side: Optional[float] = None,
    min_visible_pixels: Optional[float] = None,
) -> bool:
    """True if ``obj_id`` is distinguishable in at least one of ``frame_steps``."""
    return any(
        _distinguishable_encoding_sighting(
            episode,
            si,
            obj_id,
            min_bbox_area=min_bbox_area,
            min_side=min_side,
            min_visible_pixels=min_visible_pixels,
        )
        for si in frame_steps
    )


def _relation_shift_magnitude(
    episode: dict, from_pos, to_pos, query_step: int
) -> Optional[str]:
    """``same_side`` / ``flipped_side``: did the move change the 45°-wedge bearing?

    Compares the pre-move and post-move positions from the query pose. Stored
    as an ID difficulty axis only; it does not gate items.
    """
    ag_pos, ag_rot = agent_pose_at_step(episode, query_step)
    if ag_pos is None or ag_rot is None:
        return None
    pre = ego_label_from_world_pose(
        ag_pos, ag_rot, from_pos, episode, ahead_half_width=_cfg().directions.ahead_half_width_full_deg
    )
    post = ego_label_from_world_pose(
        ag_pos, ag_rot, to_pos, episode, ahead_half_width=_cfg().directions.ahead_half_width_full_deg
    )
    if not pre or not post:
        return None
    return 'same_side' if pre == post else 'flipped_side'


def _id_raw_frames(step: dict) -> dict:
    """Per-frame detections before the question-visibility filter, if stored."""
    from cm_benchmark.generation.step_derived import raw_visible_frames

    frames = raw_visible_frames(step)
    return frames if frames else {}


def _id_raw_odata(episode: dict, step: dict, obj_id: str) -> Optional[dict]:
    """Detection for ``obj_id`` on the raw FOV, else the step's visible map."""
    frames = _id_raw_frames(step)
    if obj_id in frames:
        from cm_benchmark.generation.step_derived import hydrate_visible_object

        return hydrate_visible_object(episode, step, obj_id, frames[obj_id])
    odata = (step.get('visible_objects') or {}).get(obj_id)
    return odata if isinstance(odata, dict) else None


def _id_from_position(episode: dict, obj_id: str, at_step: int):
    """Pre-move pose: object-state at ``at_step``, else the raw encode detection."""
    from cm_benchmark.generator.ai2thor_nav_generator import state_at_step

    track = (episode.get('object_state_track') or {}).get(obj_id) or {}
    st = state_at_step(track.get('entries') or [], int(at_step))
    if st and st.get('position') is not None:
        pos = st.get('position')
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return (float(pos[0]), float(pos[1]), float(pos[2]))
    step = step_by_index(episode, int(at_step))
    if step is not None:
        odata = _id_raw_odata(episode, step, obj_id)
        if isinstance(odata, dict) and odata.get('position') is not None:
            pos = odata.get('position')
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                return (float(pos[0]), float(pos[1]), float(pos[2]))
    return _object_world_pos(episode, obj_id)


def _id_in_raw_fov(episode: dict, obj_id: str, step_idx: int) -> bool:
    """True when ``obj_id`` is in the camera on this FOV step, before Q&A filtering."""
    if not is_fov_step(episode, step_idx):
        return False
    step = step_by_index(episode, step_idx)
    if step is None:
        return False
    frames = _id_raw_frames(step)
    if frames:
        return obj_id in frames
    return obj_id in (step.get('visible_objects') or {})


def _id_encode_sighting(episode: dict, step_idx: int, obj_id: str) -> bool:
    """Soft ``id_encode`` floors on the raw detection, plus an egocentric edge.

    The question-visibility DecisionTree drops small props (ambiguo) from
    ``visible_objects``. Encode still uses the raw frame and the soft floors,
    so a later hide can open a window.
    """
    if not is_fov_step(episode, step_idx):
        return False
    step = step_by_index(episode, step_idx)
    if step is None:
        return False
    odata = _id_raw_odata(episode, step, obj_id)
    if not isinstance(odata, dict):
        return False
    if not fov_metrics_ok(
        odata,
        image_wh=camera_size_wh(episode),
        model=None,
        **_id_encode_fov(),
    ):
        return False
    return find_ego_edge(step, obj_id) is not None


def _id_displaced_object_name(
    episode: dict, step_idx: int, obj_id: str
) -> Optional[tuple[str, str]]:
    """``(object_type, disambiguator)`` for a displaced prop at its encode step.

    Sighting uses the raw FOV and soft ``id_encode`` floors. Naming still goes
    through ``resolve_named_objects``; a prop the question filter removed is
    put back on that one step so the name is not lost with the filter.
    """
    if not _id_encode_sighting(episode, int(step_idx), obj_id):
        return None
    step = step_by_index(episode, int(step_idx))
    if step is None:
        return None
    odata = _id_raw_odata(episode, step, obj_id)
    vis = dict(step.get('visible_objects') or {})
    if isinstance(odata, dict) and obj_id not in vis:
        vis[obj_id] = odata
    patched = {
        'step': int(step_idx),
        'visible_objects': vis,
        'agent': step.get('agent'),
        'edges_egocentric': step.get('edges_egocentric') or [],
        'image_path': step.get('image_path'),
    }
    named_ep = dict(episode)
    named_ep['steps'] = [patched]
    names = resolve_named_objects(named_ep, [int(step_idx)], [obj_id])
    if names is None or obj_id not in names:
        return None
    return names[obj_id].object_type, names[obj_id].phrase


def _object_hidden_through(
    episode: dict, obj_id: str, start_step: int, end_step: int
) -> bool:
    """True if ``obj_id`` is never in view (FOV list or track) in ``[start, end]``."""
    for step in episode.get('steps') or []:
        si = int(step['step'])
        if (
            start_step <= si <= end_step
            and is_fov_step(episode, si)
            and obj_id in (step.get('visible_objects') or {})
        ):
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
    min_bbox_area: Optional[float] = None,
    min_side: Optional[float] = None,
    min_visible_pixels: Optional[float] = None,
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
    """``displacement_candidates`` rows (chosen / nearby / decoy) for one event + object."""
    return [
        row
        for row in episode.get('displacement_candidates') or []
        if row.get('event_id') == event_id and row.get('obj_id') == obj_id
    ]


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


def _hidden_query_candidates(episode: dict, obj_id: str, at_t: int) -> list[int]:
    """All steps ≥ at_t where object stays hidden from at_t through that step."""
    out: list[int] = []
    for step in episode.get('steps') or []:
        si = int(step['step'])
        if si < int(at_t) or not is_fov_step(episode, si):
            continue
        if _object_hidden_through(episode, obj_id, int(at_t), si):
            out.append(si)
    return out


def _unused_ego_label(used: set[str]) -> Optional[str]:
    """First ego-bank direction not already in ``used``, or None if none remain."""
    for lab in EGO_DIRECTION_OPTIONS:
        if lab not in used:
            return lab
    return None


def _ego_pool_from_specs(
    episode: dict, specs: list[dict], query_step: int
) -> tuple[Optional[str], list[str], list[str]]:
    """Map option specs to unique ego labels at query_step.

    Answer is kept first and is never duplicated. A distractor that lands on
    the answer's label is replaced with an unused ego-bank direction (tagged
    ``ego_bank_filler``, so the new label is not claimed as that pose). When
    no unused label remains, that distractor is omitted; the caller drops the
    item if the pool is still shorter than ``min_diagnostic_labels``.
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
            ahead_half_width=_cfg().directions.ahead_half_width_full_deg,
        )
        if not lab:
            continue
        if spec.get('is_answer'):
            if answer_dir is None:
                answer_dir = lab
                labels_seen.add(lab)
                dir_pool.append(lab)
            continue
        replaced = False
        if lab in labels_seen:
            if answer_dir is not None and lab == answer_dir:
                repaired = _unused_ego_label(labels_seen)
                if repaired is None:
                    continue
                lab = repaired
                replaced = True
            else:
                continue
        labels_seen.add(lab)
        dir_pool.append(lab)
        if replaced:
            dir_seeds.append('ego_bank_filler')
            dir_seeds.append(_mode_seed('ego_bank_filler', lab))
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
        if lab and lab not in out and len(out) < _n_options():
            out.append(lab)
            seeds_out.append(mode)
            seeds_out.append(_mode_seed(mode, lab))

    _add(OPPOSITE.get(answer) or '', 'opposite_direction')
    _add(ORTHOGONAL.get(answer) or '', 'orthogonal_direction')
    for lab in EGO_DIRECTION_OPTIONS:
        if len(out) >= _n_options():
            break
        _add(lab, 'ego_bank_filler')
    return out[: _n_options()], seeds_out


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
        if not answer or answer not in pool or len(pool) < _cfg().invisible_displacement.min_diagnostic_labels:
            continue
        key = (len(pool), si)
        if key > best_key:
            best_key = key
            best = si
    return best


def _is_swap_event(ev: dict) -> bool:
    """True if the collector logged this move as a two-object swap."""
    return (
        str(ev.get('moved_via') or '').lower() == 'swap'
        or 'object_swap' in str(ev.get('notes') or '').lower()
        or bool(ev.get('swap_partner_id'))
    )


def _partner_event(episode: dict, ev: dict) -> Optional[dict]:
    """The partner's row of the same swap event, or None if it was not logged."""
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
    """Sorted ID frames: encode, the step before the hidden move, extras, query."""
    frame_steps = [enc_idx]
    if int(at_t) - 1 > enc_idx:
        frame_steps.append(int(at_t) - 1)
    for si in extra_steps or []:
        if si not in frame_steps:
            frame_steps.append(int(si))
    if query_step not in frame_steps:
        frame_steps.append(query_step)
    return sorted({int(si) for si in frame_steps})


def _images_for_steps(episode: dict, frame_steps: list[int]) -> list[str]:
    """Frame paths for FOV-aligned ``frame_steps`` (steps without an image are skipped)."""
    return [
        p
        for si in frame_steps
        if is_fov_step(episode, si)
        for p in _img(step_by_index(episode, si))
    ]


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
    """Invisible-displacement fact whose answer is the moved object's bearing now.

    None unless the option specs give the answer plus at least one distinct
    diagnostic bearing at ``query_step``; the pool is then padded to four.
    """
    answer_dir, dir_pool, dir_seeds = _ego_pool_from_specs(episode, specs, query_step)
    if not answer_dir or len(dir_pool) < _cfg().invisible_displacement.min_diagnostic_labels:
        return None
    dir_pool, dir_seeds = _pad_id_ego_options(answer_dir, dir_pool, dir_seeds)
    return PlannedFact(
        construct='invisible_displacement',
        status='ok',
        query_step=query_step,
        encoding_step=enc_idx,
        queried_object_id=obj_id,
        answer_label=answer_dir,
        answer_source=answer_source,
        image_paths=images,
        options_pool=dir_pool,
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
    """Placement-candidate events when present; else ``displacement_events`` rows."""
    if episode.get('placement_candidates'):
        return _plan_id_from_placements(episode, max_items)
    return _plan_id_from_events(episode, max_items)


def _plan_id_from_events(episode: dict, max_items: int = 3) -> list[PlannedFact]:
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

    enc_kw = _id_encode_fov()
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
            if fact is not None and not _named_ids_confusable(
                episode, [obj_id, partner_id]
            ):
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
            extra = fact.extra or {}
            named_ids = [
                obj_id,
                extra.get('floor_anchor_id'),
                extra.get('destination_landmark_id'),
            ]
            if not _named_ids_confusable(episode, named_ids):
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


def _object_current_receptacle(episode: dict, obj_id: str, at_step: int) -> Optional[str]:
    """Parent receptacle from object_state at ``at_step``, else the catalog."""
    from cm_benchmark.generator.ai2thor_nav_generator import state_at_step

    track = (episode.get('object_state_track') or {}).get(obj_id) or {}
    st = state_at_step(track.get('entries') or [], int(at_step))
    if st and st.get('parent_receptacle'):
        return st['parent_receptacle']
    rec = (episode.get('object_catalog') or {}).get(obj_id) or {}
    return rec.get('parent_receptacle') or rec.get('receptacle')


def _xz_distance(a, b) -> Optional[float]:
    pa, pb = _xz_from_any(a), _xz_from_any(b)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def _dest_outside_frustum_through(episode: dict, dest_pos, start_step: int, end_step: int) -> bool:
    """True when ``dest_pos`` is outside the camera in every step of the window."""
    hits: list[Optional[bool]] = []
    for step in episode.get('steps') or []:
        si = int(step['step'])
        if si < int(start_step) or si > int(end_step):
            continue
        agent = step.get('agent') or {}
        hits.append(
            point_in_camera_frustum(
                agent.get('position'), agent.get('rotation'), dest_pos, episode
            )
        )
    if not hits or any(h is True for h in hits) or any(h is None for h in hits):
        return False
    return True


def _partner_occupying(episode: dict, dest_pos, exclude_id: str) -> Optional[str]:
    """Object whose AABB contains ``dest_pos`` (swap partner), else None."""
    dest_xz = _xz_from_any(dest_pos)
    if dest_xz is None:
        return None
    dest_data = {
        'aabb_min': (dest_xz[0], 0.0, dest_xz[1]),
        'aabb_max': (dest_xz[0], 0.0, dest_xz[1]),
    }
    for oid, rec in (episode.get('object_catalog') or {}).items():
        if oid == exclude_id or _is_floor_receptacle(oid):
            continue
        gap = object_aabb_gap_m(episode, oid, rec, '_dest', dest_data)
        if gap is not None and gap <= 0.0:
            return oid
    return None


def _anchor_in_frames(
    episode: dict,
    anchor_id: Optional[str],
    dest_pos,
    frame_steps: list[int],
    *,
    radius_m: Optional[float],
) -> Optional[tuple[str, str, bool]]:
    """``(anchor_id, display_name, is_floor)`` when the anchor is nameable.

    Confusable anchors are disambiguated like any named object (proximity,
    then mark) via ``resolve_named_objects``; they are not skipped.
    """
    if _is_floor_receptacle(anchor_id):
        hit = _nearest_floor_anchor(episode, dest_pos, frame_steps, radius_m=radius_m)
        if hit is None:
            return None
        return hit[0], hit[1], True
    oid = None
    for vis in passing_objects_in_frames(episode, frame_steps).values():
        for vid in vis:
            if _landmark_matches(vid, anchor_id):
                oid = vid
                break
        if oid is not None:
            break
    if oid is None:
        return None
    names = resolve_named_objects(episode, frame_steps, [oid])
    if names is None or oid not in names:
        return None
    return oid, names[oid].display_name, False


def _receptacle_parent_is_anchor(cand: dict) -> bool:
    """Receptacle dests require ``parent_receptacle == anchor_id`` (floor always ok)."""
    if _is_floor_receptacle(cand.get('anchor_id')):
        return True
    anchor = cand.get('anchor_id')
    if not anchor:
        return False
    parent = cand.get('parent_receptacle')
    if parent is None:
        return True
    return str(parent) == str(anchor)


def _id_object_not_visible(episode: dict, obj_id: str, step_idx: int) -> bool:
    """True when ``obj_id`` is absent from the raw camera list at ``step_idx``."""
    step = step_by_index(episode, step_idx)
    return step is not None and not _id_in_raw_fov(episode, obj_id, step_idx)


def _id_hidden_through(episode: dict, obj_id: str, start_step: int, end_step: int) -> bool:
    """True if ``obj_id`` stays out of the raw camera on every FOV step in ``[start, end]``.

    ``start == end`` is the query itself (latest hide is that frame). An inverted
    range is not a window. The object-state track is checked on the same
    inclusive range so a logged in-camera sighting still rejects the window.
    """
    start_step, end_step = int(start_step), int(end_step)
    if end_step < start_step:
        return False
    saw = False
    for step in episode.get('steps') or []:
        si = int(step['step'])
        if si < start_step or si > end_step or not is_fov_step(episode, si):
            continue
        saw = True
        if _id_in_raw_fov(episode, obj_id, si):
            return False
    if not saw:
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


def _id_windows_for_object(episode: dict, obj_id: str) -> list[tuple[int, int, int]]:
    """``(encode, move, query)`` windows; move is the latest hide in ``(encode, q]``.

    Encode is the latest raw-FOV sighting that clears the soft ``id_encode``
    floors. For each later query ``q``, ``move`` is the latest step in
    ``(encode, q]`` at which the object is not in the raw camera. When that
    step is ``q`` itself, the destination is checked on that query frame only.
    The agent must translate between encode and query; a hide that lands on
    the query does not need a further step after the hide.
    """
    fov = [
        int(step['step'])
        for step in episode.get('steps') or []
        if is_fov_step(episode, step.get('step'))
    ]
    enc = None
    for si in fov:
        if _id_encode_sighting(episode, si, obj_id):
            enc = si
    if enc is None:
        return []
    later = [si for si in fov if si > enc]
    windows: list[tuple[int, int, int]] = []
    for q in later:
        hidden = [si for si in later if si <= q and _id_object_not_visible(episode, obj_id, si)]
        if not hidden:
            continue
        move = hidden[-1]
        if not _id_hidden_through(episode, obj_id, move, q):
            continue
        if not _has_real_move_between(episode, enc, q):
            continue
        windows.append((enc, move, q))
    # Latest queries only; ranking stays cheap.
    return windows[-4:]


def _placement_is_receptacle_or_floor(cand: dict) -> bool:
    """Receptacle and floor trials are place/recall items, not required swaps."""
    kind = str(cand.get('anchor_kind') or '').lower()
    if kind in ('receptacle', 'floor'):
        return True
    return _is_floor_receptacle(cand.get('anchor_id'))


def _swap_partner_ready(
    episode: dict, partner_id: str, move: int, query: int
) -> Optional[tuple[int, tuple[str, str]]]:
    """``(partner_encode, (type, phrase))`` when the occupying object can be a swap."""
    enc_kw = _id_encode_fov()
    partner_enc = _last_distinguishable_sighting(episode, partner_id, move, **enc_kw)
    if partner_enc is None:
        return None
    if not _object_hidden_through(episode, partner_id, move, query):
        return None
    named = _id_displaced_object_name(episode, partner_enc, partner_id)
    if named is None:
        return None
    return partner_enc, named


def _placement_swap_decision(
    episode: dict, cand: dict, obj_id: str, move: int, query: int
) -> tuple[str, Optional[str], Optional[int], Optional[tuple[str, str]]]:
    """``(recall|swap|drop, partner_id, partner_encode, partner_name)``.

    A partner that occupies the destination makes a swap only when that partner
    can be encoded, hidden, and named. Receptacle and floor placements that
    fail those checks stay recall items. Any other placement that would have
    been a swap is dropped when the partner fails.
    """
    partner_id = _partner_occupying(episode, cand.get('position'), obj_id)
    if not partner_id:
        return 'recall', None, None, None
    ready = _swap_partner_ready(episode, partner_id, move, query)
    if ready is not None:
        partner_enc, named = ready
        return 'swap', partner_id, partner_enc, named
    if _placement_is_receptacle_or_floor(cand):
        return 'recall', None, None, None
    return 'drop', None, None, None


def _specs_from_placement(chosen: dict, siblings: list[dict], ev: dict) -> list[dict]:
    """Chosen dest + original location + other successful candidates as distractors."""
    specs = [
        {
            'role': 'chosen',
            'receptacle': chosen.get('anchor_id'),
            'position': chosen.get('position'),
            'is_answer': True,
        }
    ]
    if ev.get('from_position') is not None or ev.get('from_receptacle'):
        specs.append(
            {
                'role': 'original_location',
                'receptacle': ev.get('from_receptacle'),
                'position': ev.get('from_position'),
                'is_answer': False,
            }
        )
    for row in siblings:
        if row is chosen:
            continue
        if row.get('position') == chosen.get('position') and row.get('anchor_id') == chosen.get(
            'anchor_id'
        ):
            continue
        specs.append(
            {
                'role': 'nearby_receptacle',
                'receptacle': row.get('anchor_id'),
                'position': row.get('position'),
                'is_answer': False,
            }
        )
    return specs


def id_placement_funnel(episode: dict) -> list[tuple[str, int]]:
    """Count remaining ``(obj, candidate, window)`` triples after each ID gate."""
    successful = [
        row
        for row in episode.get('placement_candidates') or []
        if row.get('success') and row.get('obj_id') and row.get('position') is not None
    ]
    counts: list[tuple[str, int]] = [('successful_candidates', len(successful))]
    min_disp = _cfg().invisible_displacement.min_displacement_m
    by_obj: dict[str, list[dict]] = {}
    for row in successful:
        by_obj.setdefault(row['obj_id'], []).append(row)

    triples: list[tuple[str, dict, tuple[int, int, int]]] = []
    for obj_id, rows in by_obj.items():
        windows = _id_windows_for_object(episode, obj_id)
        for cand in rows:
            for window in windows:
                triples.append((obj_id, cand, window))
    counts.append(('encode_hidden_query_window', len(triples)))

    named: list[tuple[str, dict, tuple[int, int, int], tuple[str, str]]] = []
    for obj_id, cand, window in triples:
        named_obj = _id_displaced_object_name(episode, window[0], obj_id)
        if named_obj is not None:
            named.append((obj_id, cand, window, named_obj))
    counts.append(('displaced_object_nameable', len(named)))

    after_disp = []
    for obj_id, cand, window, named_obj in named:
        from_pos = _id_from_position(episode, obj_id, window[0])
        disp = _xz_distance(from_pos, cand['position'])
        if disp is not None and disp >= min_disp:
            after_disp.append((obj_id, cand, window, named_obj, from_pos, disp))
    counts.append(('min_displacement', len(after_disp)))

    after_rec = []
    for obj_id, cand, window, named_obj, from_pos, disp in after_disp:
        from_rec = _object_current_receptacle(episode, obj_id, window[0])
        if from_rec and cand.get('anchor_id') and _landmark_matches(cand['anchor_id'], from_rec):
            continue
        after_rec.append((obj_id, cand, window, named_obj, from_pos, disp, from_rec))
    counts.append(('dest_not_current_receptacle', len(after_rec)))

    after_parent = []
    for row in after_rec:
        if _receptacle_parent_is_anchor(row[1]):
            after_parent.append(row)
    counts.append(('receptacle_parent_eq_anchor', len(after_parent)))

    after_frustum = []
    for obj_id, cand, window, named_obj, from_pos, disp, from_rec in after_parent:
        _enc, move, query = window
        if _dest_outside_frustum_through(episode, cand['position'], move, query):
            after_frustum.append((obj_id, cand, window, named_obj, from_pos, disp, from_rec))
    counts.append(('dest_outside_frustum', len(after_frustum)))

    after_anchor = []
    for obj_id, cand, window, named_obj, from_pos, disp, from_rec in after_frustum:
        enc, move, query = window
        radius = cand.get('radius_m')
        if radius is None:
            radius = _cfg().invisible_displacement.floor_anchor_radius_m
        frame_steps = _build_id_frame_steps(enc, move, query)
        anchor = _anchor_in_frames(
            episode, cand.get('anchor_id'), cand['position'], frame_steps, radius_m=float(radius)
        )
        if anchor is not None:
            after_anchor.append(
                (obj_id, cand, window, named_obj, from_pos, disp, from_rec, anchor, radius)
            )
    counts.append(('anchor_nameable', len(after_anchor)))

    after_partner = []
    for obj_id, cand, window, named_obj, from_pos, disp, from_rec, anchor, radius in after_anchor:
        _enc, move, query = window
        decision, _partner, _penc, _pnamed = _placement_swap_decision(
            episode, cand, obj_id, move, query
        )
        if decision == 'drop':
            continue
        after_partner.append(
            (obj_id, cand, window, named_obj, from_pos, disp, from_rec, anchor, radius)
        )
    counts.append(('swap_partner_ok', len(after_partner)))

    n_options = 0
    for obj_id, cand, window, named_obj, from_pos, disp, from_rec, anchor, radius in after_partner:
        enc, move, query = window
        object_type, disambiguator = named_obj
        anchor_id, new_location, is_floor = anchor
        decision, partner_id, _penc, _pnamed = _placement_swap_decision(
            episode, cand, obj_id, move, query
        )
        if decision == 'drop':
            continue
        extra_fields: dict = {
            'new_location': new_location,
            'to_receptacle': cand.get('anchor_id'),
            'from_receptacle': from_rec,
            'floor_anchor_id': anchor_id if is_floor else None,
            'floor_anchor_landmark': new_location if is_floor else None,
        }
        ev = {
            'obj_id': obj_id,
            'at_timestep': move,
            'from_receptacle': from_rec,
            'to_receptacle': cand.get('anchor_id'),
            'from_position': from_pos,
            'to_position': cand['position'],
            'hidden_during': True,
        }
        images = _images_for_steps(episode, _build_id_frame_steps(enc, move, query))
        if not images:
            continue
        fact = _try_ego_direction_fact(
            episode,
            ev,
            obj_id=obj_id,
            object_type=object_type,
            enc_idx=enc,
            query_step=query,
            at_t=move,
            images=images,
            specs=_specs_from_placement(cand, by_obj[obj_id], ev),
            template_mode='swap' if decision == 'swap' else 'recall_direction',
            answer_source=['funnel'],
            extra_fields=extra_fields,
            disambiguator=disambiguator,
        )
        if fact is not None:
            n_options += 1
    counts.append(('diagnostic_options', n_options))
    return counts


def _plan_id_from_placements(episode: dict, max_items: int = 3) -> list[PlannedFact]:
    """Build ID items from successful ``placement_candidates`` + hidden walk windows."""
    successful = [
        row
        for row in episode.get('placement_candidates') or []
        if row.get('success') and row.get('obj_id') and row.get('position') is not None
    ]
    if not successful:
        return [
            PlannedFact(
                construct='invisible_displacement',
                status='unsupported',
                reason='no_successful_placement_candidates',
            )
        ]

    min_disp = _cfg().invisible_displacement.min_displacement_m
    by_obj: dict[str, list[dict]] = {}
    for row in successful:
        by_obj.setdefault(row['obj_id'], []).append(row)

    ranked: list[tuple[tuple, PlannedFact]] = []
    for obj_id, rows in by_obj.items():
        named_cache: dict[int, Optional[tuple[str, str]]] = {}
        for enc, move, query in _id_windows_for_object(episode, obj_id):
            if enc not in named_cache:
                named_cache[enc] = _id_displaced_object_name(episode, enc, obj_id)
            named = named_cache[enc]
            if named is None:
                continue
            object_type, disambiguator = named
            from_rec = _object_current_receptacle(episode, obj_id, enc)
            from_pos = _id_from_position(episode, obj_id, enc)
            frame_steps = _build_id_frame_steps(enc, move, query)
            for cand in rows:
                dest = cand['position']
                disp = _xz_distance(from_pos, dest)
                if disp is None or disp < min_disp:
                    continue
                if from_rec and cand.get('anchor_id') and _landmark_matches(
                    cand['anchor_id'], from_rec
                ):
                    continue
                if not _receptacle_parent_is_anchor(cand):
                    continue
                if not _dest_outside_frustum_through(episode, dest, move, query):
                    continue
                radius = cand.get('radius_m')
                if radius is None:
                    radius = _cfg().invisible_displacement.floor_anchor_radius_m
                anchor = _anchor_in_frames(
                    episode,
                    cand.get('anchor_id'),
                    dest,
                    frame_steps,
                    radius_m=float(radius),
                )
                if anchor is None:
                    continue
                anchor_id, new_location, is_floor = anchor
                decision, partner_id, partner_enc, partner_named = _placement_swap_decision(
                    episode, cand, obj_id, move, query
                )
                if decision == 'drop':
                    continue
                extra_steps = []
                extra_fields: dict = {
                    'new_location': new_location,
                    'to_receptacle': cand.get('anchor_id'),
                    'from_receptacle': from_rec,
                    'floor_anchor_id': anchor_id if is_floor else None,
                    'floor_anchor_landmark': new_location if is_floor else None,
                    'floor_anchor_radius_m': float(radius) if is_floor else None,
                    'relation_shift_magnitude': _relation_shift_magnitude(
                        episode, from_pos, dest, query
                    ),
                }
                template_mode = 'recall_direction'
                answer_source = [
                    f'placement_candidates[obj_id={obj_id},anchor_id={cand.get("anchor_id")}].position',
                    f'object_state_track[{obj_id}]',
                    f'agent_trajectory[{query}]',
                ]
                if decision == 'swap' and partner_id and partner_named is not None:
                    extra_steps.append(int(partner_enc))
                    extra_fields['other_object_type'] = partner_named[0]
                    extra_fields['swap_partner_id'] = partner_id
                    template_mode = 'swap'
                    answer_source.append(
                        f'placement_candidates[obj_id={obj_id}].swap_partner={partner_id}'
                    )
                ev = {
                    'obj_id': obj_id,
                    'at_timestep': move,
                    'from_receptacle': from_rec,
                    'to_receptacle': cand.get('anchor_id'),
                    'from_position': from_pos,
                    'to_position': dest,
                    'hidden_during': True,
                    'swap_partner_id': partner_id if decision == 'swap' else None,
                    'moved_via': 'swap' if decision == 'swap' else 'place',
                    'radius_m': radius,
                }
                images = _images_for_steps(
                    episode, _build_id_frame_steps(enc, move, query, extra_steps=extra_steps)
                )
                if not images:
                    continue
                specs = _specs_from_placement(cand, rows, ev)
                fact = _try_ego_direction_fact(
                    episode,
                    ev,
                    obj_id=obj_id,
                    object_type=object_type,
                    enc_idx=enc,
                    query_step=query,
                    at_t=move,
                    images=images,
                    specs=specs,
                    template_mode=template_mode,
                    answer_source=answer_source,
                    extra_fields=extra_fields,
                    disambiguator=disambiguator,
                )
                if fact is None:
                    continue
                n_diag = len(fact.options_pool or [])
                ranked.append(((-int(query), -n_diag, -float(disp), obj_id), fact))

    if not ranked:
        return [
            PlannedFact(
                construct='invisible_displacement',
                status='unsupported',
                reason=(
                    'no_placement_with_distinguishable_encode_hidden_query_and_unique_options'
                ),
            )
        ]
    ranked.sort(key=lambda row: row[0])
    return [fact for _key, fact in ranked[: max(0, int(max_items))]]


def _agent_pose_dict(episode: dict, step_idx: int) -> Optional[dict]:
    """``{x, y, z, heading}`` of the agent at a step (heading = yaw), or None."""
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
    episode: dict, obj_id: str, t0: int, t1: int, *, eps: Optional[float] = None
) -> bool:
    """Confirm object position unchanged via object_state_track (when present)."""
    if eps is None:
        eps = _cfg().spatial_updating.object_static_eps_m
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
    episode: dict,
    t0: int,
    t1: int,
    *,
    pos_tol: Optional[float] = None,
    heading_tol_deg: Optional[float] = None,
) -> bool:
    """True if encode→query has real position OR heading change (not action count)."""
    su = _cfg().spatial_updating
    if pos_tol is None:
        pos_tol = su.min_pos_change_m
    if heading_tol_deg is None:
        heading_tol_deg = su.min_heading_change_deg
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
    validate_delay_window('spatial_updating', min_delay, max_delay)

    displaced = _displaced_ids(episode)
    out: list[PlannedFact] = []
    # (obj_id, encode_step) -> answer_label of first emitted item
    seen_answers: dict[tuple[str, int], str] = {}
    for step in episode.get('steps') or []:
        step_idx = int(step['step'])
        if not is_fov_step(episode, step_idx):
            continue
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
            ag_pos, ag_rot = agent_pose_at_step(episode, step_idx)
            obj_pos = mem.get('position') or (mem.get('last_known') or {}).get('position')
            if obj_pos is None:
                obj_pos = _object_world_pos(episode, obj_id)
            label = ego_label_from_world_pose(
                ag_pos, ag_rot, obj_pos, episode, ahead_half_width=_cfg().directions.ahead_half_width_full_deg
            )
            key = (obj_id, enc_idx)
            if not label or seen_answers.get(key) == label:
                continue  # no label, or same encode/answer already emitted
            enc_vis = (enc.get('visible_objects') or {}).get(obj_id) or {}
            pre = _ego_label_at(
                episode,
                enc_idx,
                obj_id,
                obj_pos=enc_vis.get('position'),
                ahead_half_width=_cfg().directions.ahead_half_width_fov_deg,
            )
            if pre and pre not in EGO_DIRECTION_OPTIONS:
                pre = None
            pool, seeds = _ego_pool_with_diagnostics(label, pre or '')
            if pre and pre != label:
                seeds.append('pre_move_bearing')
                seeds.append(_mode_seed('pre_move_bearing', pre))
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



def _rotation_deg(episode: dict, graph=None) -> float:
    """Agent rotation step from ``episode_meta.motion``, else agent/config/graph."""
    _move, rot = episode_motion(episode, graph)
    return float(rot)


def _agent_yaw_at_step(episode: dict, step_idx: int) -> Optional[float]:
    """Agent yaw (AI2-THOR, +Z = 0) at ``step_idx``, else None."""
    _pos, rot = agent_pose_at_step(episode, int(step_idx))
    r = _rot3(rot)
    if r is None:
        return None
    return float(r[1]) % 360.0


def _xyz_tuple(pos) -> Optional[tuple[float, float, float]]:
    """``(x, y, z)`` from a dict or sequence, else None."""
    d = xyz_as_dict(pos)
    if d is None:
        return None
    return (float(d['x']), float(d['y']), float(d['z']))


def _start_pose_at_step(episode: dict, step_idx: int):
    """``([x, y, z], heading_deg)`` of the agent at ``step_idx``, else ``(None, None)``."""
    rec = _agent_pose_dict(episode, int(step_idx))
    if rec is None:
        return None, None
    return [rec['x'], rec['y'], rec['z']], rec['heading']


def _recorded_route_reference(
    episode: dict,
    graph,
    *,
    t0: int,
    t1: int,
    source_node: str,
    goal_pos,
    start_pos,
    start_heading_deg: Optional[float],
):
    """Recorded actions between source and goal, verified by continuous replay."""
    actions = recorded_actions_between(episode, t0, t1)
    if not actions:
        return None
    heading = 0.0 if start_heading_deg is None else float(start_heading_deg)
    sim = simulate_nav_action_sequence(
        graph,
        source_node,
        actions,
        start_heading_deg=heading,
        start_pos=start_pos,
    )
    if not sim.get('ok'):
        return None
    goal_xyz = _xyz_tuple(goal_pos)
    if not xz_reaches_goal(sim.get('end_xz'), goal_xyz, sim['goal_tolerance']):
        return None
    return {
        'actions': actions,
        'answer_label': format_nav_actions(actions),
        'start_heading_deg': heading,
        'turn_count': count_direction_changes(actions),
    }


def _survey_bfs_reference(
    graph,
    *,
    source_node: str,
    start_pos,
    start_heading_deg: Optional[float],
    goal_pos,
    allowed_nodes,
    allowed_edges=None,
    actions=None,
):
    """Continuous reference from the source pose to the goal.

    Callers must already have passed the graph geodesic gates. ``actions`` is
    the sequence from a shared source flood; when omitted, one goal is searched
    on its own. The search does not apply the geodesic gates itself.
    """
    heading = 0.0 if start_heading_deg is None else float(start_heading_deg)
    if actions is None:
        actions = continuous_bfs_actions(
            graph,
            start_pos,
            heading,
            goal_pos,
            allowed_nodes=allowed_nodes,
            allowed_edges=allowed_edges,
        )
    if not actions:
        return None
    sim = simulate_nav_action_sequence(
        graph,
        source_node,
        actions,
        start_heading_deg=heading,
        start_pos=start_pos,
        allowed_nodes=allowed_nodes,
        allowed_edges=allowed_edges,
    )
    if not sim.get('ok'):
        return None
    if not xz_reaches_goal(sim.get('end_xz'), _xyz_tuple(goal_pos), sim['goal_tolerance']):
        return None
    return {
        'actions': actions,
        'answer_label': format_nav_actions(actions),
        'start_heading_deg': heading,
        'turn_count': count_direction_changes(actions),
    }


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

    heading = start_heading_deg
    if heading is None:
        heading = path_start_heading_deg(graph, path_nodes)
    if heading is None:
        return None

    fb = _cfg().class4.fallback
    grid = float(graph.graph.get('grid_size') or fb.grid_size_m)
    move = float(graph.graph.get('agent_move_m') or grid)
    # Smoothest first. A wide tolerance cuts staircase corners and can leave the
    # walked edge set, so fall back to tighter ones and finally to the exact
    # per-hop conversion, which stays on the chain by construction.
    s = max(grid, move)
    mults = _cfg().class4.follower.ladder_multipliers
    ladder = [s * mults[0], s * mults[1], grid * mults[2], grid * mults[3]]

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
    """Category name from the ``world_layout`` landmark row, else from the id stem."""
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
            lm_cfg = _cfg().landmarks
            seen[lid]['salience'] = lm_cfg.layout_base + lm_cfg.layout_per_sighting * sighting_count.get(lid, 1)
        # Layout-only (never distinguishable in FOV) — skip
    for oid, row in seen.items():
        if not row['from_layout']:
            lm_cfg = _cfg().landmarks
            row['salience'] = lm_cfg.other_base + lm_cfg.other_per_sighting * sighting_count.get(oid, 1)
    items = list(seen.values())
    items.sort(key=lambda r: (-r['salience'], r['first_seen_step'], r['obj_id']))
    return items


def select_landmark_candidates(
    episode: dict,
    *,
    top_n_per_region: Optional[int] = None,
    max_total: Optional[int] = None,
    unique_category: bool = True,
    floor_standing: bool = False,
) -> list[dict]:
    """Salience-filtered landmarks: visibility first, then top-N weighted per region.

    Never nearest-distance-only and never uniform-random. Class-4 endpoints
    pass ``unique_category=False`` and uniquely name duplicates via
    ``_referring_display_name`` (GOAT-Bench 2024 instance language; shared
    taxonomy referring-expression exception). Route and survey also pass
    ``floor_standing=True`` so endpoints are the collector's floor-standing
    landmarks (SPOC ``salient_landmarks``), not every visible object.
    Other callers keep the default True so source/goal names stay category-unique.
    """
    lm_cfg = _cfg().landmarks
    if top_n_per_region is None:
        top_n_per_region = lm_cfg.top_n_per_region
    if max_total is None:
        max_total = lm_cfg.max_total
    candidates = _distinguishable_landmark_candidates(episode)
    if not candidates:
        return []
    if floor_standing:
        allowed = _salient_floor_landmark_ids(episode)
        if allowed is not None:
            candidates = [lm for lm in candidates if lm.get('obj_id') in allowed]
        else:
            candidates = [
                lm
                for lm in candidates
                if lm.get('obj_id') and not _is_passage_landmark(lm['obj_id'], lm.get('name'))
            ]
        if not candidates:
            return []
    if unique_category:
        ids = [lm['obj_id'] for lm in candidates if lm.get('obj_id')]
        candidates = [
            lm for lm in candidates if lm.get('obj_id') and group_is_unique(episode, lm['obj_id'], ids)
        ]
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

    if n0 not in graph or n1 not in graph:
        return None
    try:
        return float(nx.shortest_path_length(graph, n0, n1, weight='weight'))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _percentile_of(values: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile (index ``int(p * (n - 1))``) of a non-empty list."""
    ordered = sorted(float(v) for v in values)
    idx = max(0, int(float(percentile) * (len(ordered) - 1)))
    return ordered[idx]


def _pair_component_ids(pair_graph, node_ids: Sequence[str]) -> dict[str, int]:
    """Connected-component id on ``pair_graph`` for nodes that appear in it."""
    component_of: dict[str, int] = {}
    cid = 0
    seen: set[str] = set()
    for nid in node_ids:
        if nid in seen or nid not in pair_graph:
            continue
        for member in nx.node_connected_component(pair_graph, nid):
            if member not in component_of:
                component_of[member] = cid
            seen.add(member)
        cid += 1
    return component_of


def _scene_pair_geodesics_and_ratios(
    graph,
    node_ids: Sequence[str],
    *,
    pair_graph=None,
) -> tuple[list[float], list[float]]:
    """Full-graph geodesic metres and geo/eucl ratios for unique node pairs.

    One Dijkstra per source. Pairs with no path or zero Euclidean are skipped
    (same filters ``_class4_pair_reject_reason`` applies before the ratio gate).
    When ``pair_graph`` is set (route: the traversed_edges subgraph), a pair
    is included only if both nodes share a connected component there. Lengths
    stay on ``graph`` so the floor matches the full-graph pair gate.
    """

    uniq = [n for n in dict.fromkeys(node_ids) if n in graph]
    if len(uniq) < 2:
        return [], []
    component_of = _pair_component_ids(pair_graph, uniq) if pair_graph is not None else None
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
        src_comp = None if component_of is None else component_of.get(src)
        for dst in uniq[i + 1 :]:
            if component_of is not None and (
                src_comp is None or component_of.get(dst) != src_comp
            ):
                continue
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
    percentile: Optional[float] = None,
    pair_graph=None,
) -> Optional[float]:
    """Scene-relative pair-length floor: a percentile of this scene's own spread.

    Same shape as ``calibrate_view_radius_m``. Computed once per episode.
    Returns None when there is no usable distribution (absolute band only).
    ``pair_graph`` limits the pool (route passes the traversed subgraph).
    """
    if percentile is None:
        percentile = _cfg().class4.scene_geodesic_percentile
    return _scene_class4_floors(
        graph,
        node_ids,
        ratio_cap=math.inf,
        percentile=percentile,
        pair_graph=pair_graph,
    )[0]


def calibrate_scene_ratio_floor(
    graph,
    node_ids: Sequence[str],
    *,
    percentile: Optional[float] = None,
    cap: Optional[float] = None,
    pair_graph=None,
) -> float:
    """Scene-relative geo/eucl floor, never above ``cap``.

    Same pairs and percentile as ``calibrate_scene_geodesic_floor_m``. An
    open-plan scene whose median detour sits below the Habitat 1.1 cap gets
    that lower, achievable bar instead of zero items. ``pair_graph`` limits
    the pool the same way as the geodesic floor.
    """
    if percentile is None:
        percentile = _cfg().class4.scene_geodesic_percentile
    if cap is None:
        cap = _cfg().class4.route.max_ratio_floor
    return _scene_class4_floors(
        graph,
        node_ids,
        ratio_cap=cap,
        percentile=percentile,
        pair_graph=pair_graph,
    )[1]


def _scene_class4_floors(
    graph,
    node_ids: Sequence[str],
    *,
    ratio_cap: float,
    percentile: Optional[float] = None,
    pair_graph=None,
) -> tuple[Optional[float], float]:
    """Both scene-relative class-4 floors from one Dijkstra pass.

    Returns ``(geodesic floor or None, min(ratio_cap, ratio percentile))``;
    the ratio floor is ``ratio_cap`` when the scene has no usable pair.
    ``pair_graph`` restricts both floors to the same pair list (route:
    walked landmark pairs). Survey leaves it unset and keeps every pair.
    """
    if percentile is None:
        percentile = _cfg().class4.scene_geodesic_percentile
    geos, ratios = _scene_pair_geodesics_and_ratios(
        graph, node_ids, pair_graph=pair_graph
    )
    geo_floor = _percentile_of(geos, percentile) if geos else None
    if not ratios:
        return geo_floor, float(ratio_cap)
    return geo_floor, min(float(ratio_cap), _percentile_of(ratios, percentile))


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
    if geo < float(_cfg().class4.min_pair_geodesic_m):
        return 'geodesic_below_min'
    if geo > float(_cfg().class4.max_pair_geodesic_m):
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
    """Unsupported reason string: ``fallback:reason=count,...`` (most common first)."""
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
    """Edges as sorted ``[a, b]`` lists in sorted order (stable JSON)."""
    out: list[list[str]] = []
    for edge in edges or []:
        if not edge or len(edge) != 2:
            continue
        a, b = edge[0], edge[1]
        out.append([a, b] if a <= b else [b, a])
    out.sort()
    return out


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
    """The ``episode_start`` nav graph, or None if missing or unparsable.

    Fat episodes may still embed ``nav_graph``. Slim episodes store
    ``nav_graph_path`` and the graph is read from that SPOC file.
    """
    from cm_benchmark.generation.nav_graph import load_episode_nav_graph

    return load_episode_nav_graph(episode)


def _episode_traversed_subgraph(episode: dict, graph):
    """Snap trajectory once; return (snapped, traversed_nodes, traversed_sub)."""

    snapped = snap_trajectory_to_graph(graph, episode.get('agent_trajectory') or [])
    traversed = traversed_node_ids(snapped)
    edges = traversed_edges_from_snapped(snapped)
    sub = subgraph_from_traversed_edges(graph, edges)
    return snapped, traversed, edges, sub


def _episode_confusable_pool(episode: dict, extra_ids: Sequence[str]) -> list[str]:
    """Catalog ids plus the endpoints being named (episode-wide confusable pool)."""
    ids = {oid for oid in extra_ids if oid}
    ids.update(oid for oid in (episode.get('object_catalog') or {}) if oid)
    return list(ids)


def _unique_endpoint_ids(episode: dict, obj_ids: Sequence[str]) -> set[str]:
    """Endpoints with no confusable sibling in the episode pool."""
    pool = _episode_confusable_pool(episode, obj_ids)
    return {oid for oid in obj_ids if oid and group_is_unique(episode, oid, pool)}


def _class4_emit_key(
    unique_pair: bool, turn_count: int, geodesic_m: float, hops: int
) -> tuple:
    """Unique endpoint pairs sort ahead of confusable ones, then harder routes.

    Colour marks stay the naming fallback when a confusable landmark is needed
    to fill the item. A unique pair that passed the other gates is emitted first.
    """
    return (1 if unique_pair else 0, int(turn_count), float(geodesic_m), int(hops))


def plan_route_knowledge(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Retrace an EXPERIENCED path as collected-format actions on traversed_edges."""

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

    landmarks = select_landmark_candidates(
        episode, unique_category=False, floor_standing=True
    )
    id_to_node, _node_to_name, meta = _build_landmark_node_map(
        graph,
        landmarks,
        candidate_nodes=traversed,
        max_distance_m=_cfg().class4.landmark_snap_m,
    )
    if len(id_to_node) < 2:
        return [
            PlannedFact(
                construct='route_knowledge',
                status='unsupported',
                reason='need_ge2_distinguishable_landmarks_snapped',
            )
        ]

    min_hops = _cfg().class4.route.min_hop_count
    traversed_edge_list = _serialize_traversed_edges(edges)
    rejects: Counter = Counter()

    def _walk_index(oid: str) -> int:
        try:
            return traversed.index(id_to_node[oid])
        except ValueError:
            return 1_000_000_000

    ordered_ids = sorted(
        [lm['obj_id'] for lm in landmarks if lm['obj_id'] in id_to_node],
        key=lambda oid: (
            _walk_index(oid),
            int(meta[oid].get('first_seen_step') or 0),
            oid,
        ),
    )
    unique_ids = _unique_endpoint_ids(episode, ordered_ids)
    route_nodes = [id_to_node[oid] for oid in ordered_ids]
    # Length and detour floors use walked landmark pairs only. Survey keeps
    # the full-scene pool (it does not pass the traversed subgraph).
    scene_geodesic_floor, scene_ratio_floor = _scene_class4_floors(
        graph,
        route_nodes,
        ratio_cap=_cfg().class4.route.max_ratio_floor,
        pair_graph=walked,
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
            if _named_ids_confusable(episode, [src_id, goal_id]):
                rejects['confusable_endpoints'] += 1
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
            if len(path) > _cfg().class4.route.max_subpath_nodes:
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

            t_src = int(meta[src_id]['first_seen_step'])
            t_goal = int(meta[goal_id]['first_seen_step'])
            if t_goal <= t_src:
                rejects['goal_not_after_source'] += 1
                continue
            start_pos, start_heading = _start_pose_at_step(episode, t_src)
            ref = _recorded_route_reference(
                episode,
                graph,
                t0=t_src,
                t1=t_goal,
                source_node=n0,
                goal_pos=meta[goal_id]['position'],
                start_pos=start_pos,
                start_heading_deg=start_heading,
            )
            if ref is None:
                rejects['actions_unusable'] += 1
                continue
            scored = score_route_action_sequence(
                graph,
                n0,
                n1,
                ref['actions'],
                edges,
                start_heading_deg=ref['start_heading_deg'],
                start_pos=start_pos,
                goal_pos=meta[goal_id]['position'],
            )
            if scored.get('outcome') != 'valid_success':
                rejects['actions_unusable'] += 1
                continue
            turn_count = int(ref['turn_count'])
            if turn_count < _cfg().class4.route.min_turns:
                # A straight walk is answerable from a language prior alone.
                rejects[f'turns_lt_{_cfg().class4.route.min_turns}'] += 1
                continue
            if turn_count > _cfg().class4.route.max_turns:
                rejects[f'turns_gt_{_cfg().class4.route.max_turns}'] += 1
                continue

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
                    query_step=t_goal,
                    encoding_step=t_src,
                    answer_label=ref['answer_label'],
                    answer_source=[
                        f'agent_actions[{t_src}<step<={t_goal}]',
                        f'agent_trajectory[{t_src}→{t_goal}]',
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
                        'start_pos': start_pos,
                        'goal_pos': list(_xyz_tuple(meta[goal_id]['position']) or []),
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
            # Unique names first, then harder: more real turns, then longer geodesic.
            candidates.append(
                (
                    _class4_emit_key(
                        src_id in unique_ids and goal_id in unique_ids,
                        turn_count,
                        geodesic_m,
                        hops,
                    ),
                    fact,
                )
            )

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


_ROUTE_PAIR_GATES = (
    'landmark_pairs_on_walked_nodes',
    'same_node',
    'geodesic_scene_floor',
    'detour_ratio',
    'names',
    'walked_path',
    'hops',
    'real_move',
    'recorded_actions_replay',
    'turn_band',
)


def route_pair_funnel(episode: dict) -> dict:
    """Pairs left after each ``plan_route_knowledge`` gate, plus percentile pools.

    Route's length percentile uses walked landmark pairs only
    (``pair_graph`` = traversed subgraph). Survey's floor, reported alongside,
    uses the full-scene landmark pool and does not pass ``pair_graph``.
    """
    zeros = [(name, 0) for name in _ROUTE_PAIR_GATES]
    info = {
        'gates': zeros,
        'walked_pair_count': 0,
        'full_scene_pair_count': 0,
        'survey_pair_count': 0,
        'route_geodesic_floor': None,
        'full_scene_geodesic_floor': None,
        'survey_geodesic_floor': None,
        'route_percentile_pair_graph': 'traversed',
        'survey_percentile_pair_graph': 'full',
    }
    graph = _load_nav_graph_or_none(episode)
    if graph is None:
        return info
    _snapped, traversed, edges, walked = _episode_traversed_subgraph(episode, graph)
    landmarks = select_landmark_candidates(
        episode, unique_category=False, floor_standing=True
    )
    id_to_node, _names, meta = _build_landmark_node_map(
        graph,
        landmarks,
        candidate_nodes=traversed,
        max_distance_m=_cfg().class4.landmark_snap_m,
    )
    ordered_ids = sorted(
        [lm['obj_id'] for lm in landmarks if lm['obj_id'] in id_to_node],
        key=lambda oid: (
            traversed.index(id_to_node[oid]) if id_to_node[oid] in traversed else 1_000_000_000,
            int(meta[oid].get('first_seen_step') or 0),
            oid,
        ),
    )
    route_nodes = [id_to_node[oid] for oid in ordered_ids]
    percentile = _cfg().class4.scene_geodesic_percentile
    walked_geos, _walked_ratios = _scene_pair_geodesics_and_ratios(
        graph, route_nodes, pair_graph=walked
    )
    full_geos, _full_ratios = _scene_pair_geodesics_and_ratios(graph, route_nodes)
    route_floor, _route_ratio = _scene_class4_floors(
        graph,
        route_nodes,
        ratio_cap=_cfg().class4.route.max_ratio_floor,
        pair_graph=walked,
    )
    full_floor, _full_ratio_floor = _scene_class4_floors(
        graph, route_nodes, ratio_cap=_cfg().class4.route.max_ratio_floor
    )
    if walked_geos:
        assert route_floor == _percentile_of(walked_geos, percentile)
    assert len(walked_geos) <= len(full_geos)
    survey_map, _survey_names, _survey_meta = _build_landmark_node_map(
        graph, landmarks, max_distance_m=_cfg().class4.landmark_snap_m
    )
    survey_ids = [lm['obj_id'] for lm in landmarks if lm['obj_id'] in survey_map]
    survey_nodes = [survey_map[oid] for oid in survey_ids]
    survey_geos, _survey_ratios = _scene_pair_geodesics_and_ratios(graph, survey_nodes)
    survey_floor, _survey_ratio_floor = _scene_class4_floors(
        graph, survey_nodes, ratio_cap=_cfg().class4.survey.max_ratio_floor
    )
    info.update(
        {
            'walked_pair_count': len(walked_geos),
            'full_scene_pair_count': len(full_geos),
            'survey_pair_count': len(survey_geos),
            'route_geodesic_floor': route_floor,
            'full_scene_geodesic_floor': full_floor,
            'survey_geodesic_floor': survey_floor,
        }
    )
    print(
        f"route percentile pairs: walked={len(walked_geos)} "
        f"same_walked_nodes_without_component_filter={len(full_geos)} "
        f"floor_walked={route_floor} floor_same_nodes={full_floor}"
    )
    print(
        f"survey percentile pairs: full_scene={len(survey_geos)} "
        f"floor={survey_floor} (no traversed filter)"
    )
    if survey_geos:
        assert survey_floor == _percentile_of(survey_geos, percentile)
    if len(traversed) < 3 or walked.number_of_edges() < 1 or len(ordered_ids) < 2:
        return info

    min_hops = _cfg().class4.route.min_hop_count
    max_nodes = _cfg().class4.route.max_subpath_nodes
    min_turns = _cfg().class4.route.min_turns
    max_turns = _cfg().class4.route.max_turns
    pairs = [(src, goal) for i, src in enumerate(ordered_ids) for goal in ordered_ids[i + 1 :]]
    gates: list[tuple[str, int]] = [('landmark_pairs_on_walked_nodes', len(pairs))]

    pairs = [(s, g) for s, g in pairs if id_to_node[s] != id_to_node[g]]
    gates.append(('same_node', len(pairs)))

    kept = []
    for src, goal in pairs:
        reason = _class4_pair_reject_reason(
            graph,
            meta[src]['position'],
            meta[goal]['position'],
            id_to_node[src],
            id_to_node[goal],
            min_scene_geodesic_m=route_floor,
        )
        if reason is None:
            kept.append((src, goal))
    pairs = kept
    gates.append(('geodesic_scene_floor', len(pairs)))

    kept = []
    for src, goal in pairs:
        reason = _class4_pair_reject_reason(
            graph,
            meta[src]['position'],
            meta[goal]['position'],
            id_to_node[src],
            id_to_node[goal],
            min_ratio=_route_ratio,
            min_scene_geodesic_m=route_floor,
        )
        if reason is None:
            kept.append((src, goal))
    pairs = kept
    gates.append(('detour_ratio', len(pairs)))

    kept = []
    for src, goal in pairs:
        source = _referring_display_name(episode, int(meta[src]['first_seen_step']), src)
        goal_name = _referring_display_name(episode, int(meta[goal]['first_seen_step']), goal)
        if source and goal_name and source != goal_name:
            kept.append((src, goal))
    pairs = kept
    gates.append(('names', len(pairs)))

    kept_paths = []
    for src, goal in pairs:
        n0, n1 = id_to_node[src], id_to_node[goal]
        try:
            path = shortest_path(walked, n0, n1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) < 2:
            continue
        kept_paths.append((src, goal, path))
    gates.append(('walked_path', len(kept_paths)))

    kept_paths = [
        row
        for row in kept_paths
        if (len(row[2]) - 1) >= min_hops and len(row[2]) <= max_nodes
    ]
    gates.append(('hops', len(kept_paths)))

    moved = []
    for src, goal, path in kept_paths:
        t_src = int(meta[src]['first_seen_step'])
        t_goal = int(meta[goal]['first_seen_step'])
        if not _has_real_move_between(episode, t_src, max(t_goal, t_src + 1)) and len(path) < 3:
            continue
        moved.append((src, goal, path, t_src, t_goal))
    gates.append(('real_move', len(moved)))

    replayed = []
    for src, goal, path, t_src, t_goal in moved:
        if t_goal <= t_src:
            continue
        n0, n1 = id_to_node[src], id_to_node[goal]
        start_pos, start_heading = _start_pose_at_step(episode, t_src)
        ref = _recorded_route_reference(
            episode,
            graph,
            t0=t_src,
            t1=t_goal,
            source_node=n0,
            goal_pos=meta[goal]['position'],
            start_pos=start_pos,
            start_heading_deg=start_heading,
        )
        if ref is None:
            continue
        scored = score_route_action_sequence(
            graph,
            n0,
            n1,
            ref['actions'],
            edges,
            start_heading_deg=ref['start_heading_deg'],
            start_pos=start_pos,
            goal_pos=meta[goal]['position'],
        )
        if scored.get('outcome') != 'valid_success':
            continue
        replayed.append((src, goal, ref))
    gates.append(('recorded_actions_replay', len(replayed)))

    turned = [
        row
        for row in replayed
        if min_turns <= int(row[2]['turn_count']) <= max_turns
    ]
    gates.append(('turn_band', len(turned)))
    info['gates'] = gates
    return info


def plan_survey_based_route_planning(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Plan a never-walked source→goal as collected-format actions on viewed_edges.

    Pair gates use full-nav_graph geodesic, but the reference path and its
    actions live on the viewed subgraph, because that is what validity scores.
    """

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
    landmarks = select_landmark_candidates(
        episode, unique_category=False, floor_standing=True
    )
    id_to_node, _node_to_name, meta = _build_landmark_node_map(
        graph, landmarks, max_distance_m=_cfg().class4.landmark_snap_m
    )
    ids = [lm['obj_id'] for lm in landmarks if lm['obj_id'] in id_to_node]
    unique_ids = _unique_endpoint_ids(episode, ids)
    if len(ids) < 2:
        return [
            PlannedFact(
                construct='survey_based_route_planning',
                status='unsupported',
                reason='need_ge2_distinguishable_landmarks_snapped',
            )
        ]

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
        graph, survey_nodes, ratio_cap=_cfg().class4.survey.max_ratio_floor
    )
    # Same as route: collect every valid pair, then emit the hardest. Returning
    # at the first max_items hits takes them in `ids` order, which is salience,
    # not difficulty. Continuous search runs only after the geodesic gate, and
    # one flood covers every goal that shares a source pose.
    candidates: list[tuple[tuple, PlannedFact]] = []
    rejects: Counter = Counter()
    pending: list[dict] = []
    allowed_nodes = set(viewed_sub.nodes)
    allowed_edges = {(a, b) if a <= b else (b, a) for a, b in viewed_edges}

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
            if _named_ids_confusable(episode, [src_id, goal_id]):
                rejects['confusable_endpoints'] += 1
                continue
            t0 = int(meta[src_id]['first_seen_step'])
            t1 = int(meta[goal_id]['first_seen_step'])
            start_pos, start_heading = _start_pose_at_step(episode, t0)
            pending.append(
                {
                    'src_id': src_id,
                    'goal_id': goal_id,
                    'n0': n0,
                    'n1': n1,
                    'source': source,
                    'goal': goal,
                    'path': path,
                    't0': t0,
                    't1': t1,
                    'start_pos': start_pos,
                    'start_heading': start_heading,
                    'goal_pos': meta[goal_id]['position'],
                    'src_pos': meta[src_id]['position'],
                }
            )

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for job in pending:
        start_pos = job['start_pos']
        heading = job['start_heading']
        if start_pos is None:
            pose_key = (None, None)
        else:
            pose_key = (
                tuple(float(v) for v in start_pos),
                None if heading is None else float(heading),
            )
        groups[pose_key].append(job)
    for pose_key, group in groups.items():
        start_tuple, heading = pose_key
        actions_for = continuous_bfs_to_goals(
            graph,
            None if start_tuple is None else list(start_tuple),
            0.0 if heading is None else heading,
            [job['goal_pos'] for job in group],
            allowed_nodes=allowed_nodes,
            allowed_edges=allowed_edges,
        )
        for job, actions in zip(group, actions_for):
            job['actions'] = actions

    for job in pending:
        src_id = job['src_id']
        goal_id = job['goal_id']
        n0 = job['n0']
        n1 = job['n1']
        source = job['source']
        goal = job['goal']
        path = job['path']
        t0 = job['t0']
        t1 = job['t1']
        start_pos = job['start_pos']
        ref = _survey_bfs_reference(
            graph,
            source_node=n0,
            start_pos=start_pos,
            start_heading_deg=job['start_heading'],
            goal_pos=job['goal_pos'],
            allowed_nodes=allowed_nodes,
            allowed_edges=allowed_edges,
            actions=job.get('actions'),
        )
        if ref is None:
            rejects['actions_unusable'] += 1
            continue
        scored = score_survey_action_sequence(
            graph,
            n0,
            n1,
            ref['actions'],
            viewed_edges,
            edges,
            start_heading_deg=ref['start_heading_deg'],
            start_pos=start_pos,
            goal_pos=meta[goal_id]['position'],
        )
        if scored.get('outcome') != 'valid_success':
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
        if turn_count < _cfg().class4.survey.min_turns:
            rejects[f'turns_lt_{_cfg().class4.survey.min_turns}'] += 1
            continue
        geodesic_m = _full_graph_geodesic_m(graph, n0, n1) or 0.0
        fact = (
            PlannedFact(
                construct='survey_based_route_planning',
                status='ok',
                query_step=max(t0, t1),
                encoding_step=min(t0, t1),
                answer_label=ref['answer_label'],
                answer_source=[
                    f'nav_graph.viewed_edges.continuous_bfs[{n0}→{n1}]',
                    'continuous_bfs_actions(source_pose)',
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
                    'start_pos': start_pos,
                    'goal_pos': list(_xyz_tuple(meta[goal_id]['position']) or []),
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
        # Unique names first, then harder: more real turns, then longer geodesic.
        candidates.append(
            (
                _class4_emit_key(
                    src_id in unique_ids and goal_id in unique_ids,
                    turn_count,
                    geodesic_m,
                    hops,
                ),
                fact,
            )
        )

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
    each landmark's clear sighting frame. Triples are tried in landmark
    salience order and the first ``max_items`` valid ones are emitted.
    """
    pt = _cfg().perspective_taking
    landmarks = select_landmark_candidates(
        episode, top_n_per_region=pt.top_n_per_region, max_total=pt.max_total
    )
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
                # Names are already distinct: candidates are category-unique.
                if not pos_a or not pos_b or not pos_c:
                    continue

                sa = int(a['first_seen_step'])
                sb = int(b['first_seen_step'])
                sc = int(c['first_seen_step'])
                name_a = _referring_display_name(episode, sa, a['obj_id'])
                name_b = _referring_display_name(episode, sb, b['obj_id'])
                name_c = _referring_display_name(episode, sc, c['obj_id'])
                if name_a is None or name_b is None or name_c is None:
                    continue
                if _named_ids_confusable(
                    episode, [a['obj_id'], b['obj_id'], c['obj_id']]
                ):
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
                    # Each landmark's own frame; the names above already required
                    # a distinguishable sighting there.
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
                        ahead_half_width=_cfg().directions.ahead_half_width_fov_deg,
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
                pool += [f for f in EGO_DIRECTION_OPTIONS if f not in pool]
                # How far the imagined heading (A→B) is from the camera heading.
                shift = None
                pose = _agent_pose_dict(episode, query_step)
                if pose is not None:
                    imag_h = math.degrees(
                        math.atan2(pos_b['x'] - pos_a['x'], pos_b['z'] - pos_a['z'])
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
                        options_pool=pool[: _n_options()],
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
    max_per_construct: Optional[int] = None,
    *,
    swm_min_delay: Optional[int] = None,
    swm_max_delay: Optional[int] = None,
    su_min_delay: Optional[int] = None,
    su_max_delay: Optional[int] = None,
    visibility_model_path: Optional[str] = None,
    cfg=None,
) -> list[PlannedFact]:
    """Run every requested construct planner on one episode, in ``constructs`` order.

    First filters each step's ``visible_objects`` through the visibility model
    (DecisionTree joblib if found, else static ``question_visibility``) and
    attaches the model so the distinguishability gates use it. Unknown
    construct names are ignored. Returns all facts, including unsupported ones.
    """
    from cm_benchmark.generation.config import set_generation_config, reset_generation_config

    token = None
    if cfg is not None:
        token = set_generation_config(cfg)
    try:
        active = cfg or get_generation_config()
        if max_per_construct is None:
            max_per_construct = active.budget.max_per_construct
        if swm_min_delay is None:
            swm_min_delay = active.spatial_working_memory.min_delay
        if swm_max_delay is None:
            swm_max_delay = active.spatial_working_memory.max_delay
        if su_min_delay is None:
            su_min_delay = active.spatial_updating.min_delay
        if su_max_delay is None:
            su_max_delay = active.spatial_updating.max_delay
        model_path = visibility_model_path or active.visibility.model_path
        from cm_benchmark.generation.step_derived import prepare_episode

        episode = prepare_episode(episode)
        episode = apply_question_visibility_to_episode(
            episode,
            inplace=False,
            model_path=model_path,
        )
        delay_kwargs = {
            'spatial_working_memory': {'min_delay': swm_min_delay, 'max_delay': swm_max_delay},
            'spatial_updating': {'min_delay': su_min_delay, 'max_delay': su_max_delay},
        }
        facts: list[PlannedFact] = []
        for key in constructs or list(PLANNERS):
            fn = PLANNERS.get(key)
            if fn is not None:
                facts.extend(fn(episode, max_items=max_per_construct, **delay_kwargs.get(key, {})))
        return _apply_whole_item_names(episode, facts)
    finally:
        if token is not None:
            reset_generation_config(token)


def __getattr__(name: str):
    """Former module-level quality constants, read from the active config."""
    return legacy_constant(name)
