"""Task planner: select construct-eligible facts from episode GT (deterministic)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

from cm_benchmark.generation.constructs import (
    EGO_DIRECTION_OPTIONS,
    OPPOSITE,
    ORTHOGONAL,
    angle_relation_to_ego_label,
    find_ego_edge,
    find_inferred_edge,
    humanize_receptacle,
    object_type_from_id,
    step_by_index,
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


def plan_egocentric_encoding(episode: dict, max_items: int = 3) -> list[PlannedFact]:
    out = []
    for step in episode.get('steps') or []:
        step_idx = int(step['step'])
        visible = step.get('visible_objects') or {}
        for obj_id in visible:
            edge = find_ego_edge(step, obj_id)
            if not edge:
                continue
            label = angle_relation_to_ego_label(edge.get('angle_relation'))
            if not label or label not in EGO_DIRECTION_OPTIONS:
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
                        f"steps[{step_idx}].edges_egocentric[target={obj_id}].angle_relation"
                    ],
                    image_paths=_img(step),
                    options_pool=pool,
                    distractor_seeds=seeds,
                    extra={
                        'object_type': object_type_from_id(obj_id, visible),
                        'angle_relation': edge.get('angle_relation'),
                        'frame_of_reference': 'egocentric',
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
    """Recall past relation (encoding step) under delay k; optional count/load mode."""
    if min_delay < 1:
        raise ValueError('SWM min_delay must be at least 1')
    if max_delay is not None and max_delay < min_delay:
        raise ValueError('SWM max_delay must be greater than or equal to min_delay')

    displaced = _displaced_ids(episode)
    out: list[PlannedFact] = []

    for step in episode.get('steps') or []:
        step_idx = int(step['step'])
        non_vis = step.get('non_visible_objects') or {}
        for obj_id, mem in non_vis.items():
            if obj_id in displaced:
                continue
            last_seen = mem.get('last_seen_step')
            if last_seen is None:
                continue
            enc_idx = int(last_seen)
            if step_idx <= enc_idx:
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
            # Answer only from encoding-step ego edge (not inferred-now)
            edge = find_ego_edge(enc, obj_id)
            if not edge:
                continue
            label = angle_relation_to_ego_label(edge.get('angle_relation'))
            if not label or label not in EGO_DIRECTION_OPTIONS:
                continue

            # Diagnostic: current-view answer if something else is at that bearing now
            current_view_lab = None
            for oid, odata in (step.get('visible_objects') or {}).items():
                e = find_ego_edge(step, oid)
                if not e:
                    continue
                lab = angle_relation_to_ego_label(e.get('angle_relation'))
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

            out.append(
                PlannedFact(
                    construct='spatial_working_memory',
                    status='ok',
                    query_step=step_idx,
                    encoding_step=enc_idx,
                    queried_object_id=obj_id,
                    answer_label=label,
                    answer_source=[
                        f"steps[{enc_idx}].edges_egocentric[target={obj_id}].angle_relation"
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
                    },
                )
            )
            if len(out) >= max_items:
                return out

    # Optional count / load mode when category counts are computable from GT
    if len(out) < max_items:
        for step in episode.get('steps') or []:
            step_idx = int(step['step'])
            if step_idx < min_delay:
                continue
            categories = {
                object_type_from_id(oid, {oid: odata})
                for s in episode.get('steps') or []
                if int(s['step']) <= step_idx
                for oid, odata in (s.get('visible_objects') or {}).items()
            }
            for cat in sorted(categories):
                n = _count_category_seen(episode, cat, step_idx)
                if n < 1:
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
                if len(out) >= max_items:
                    return out
                break  # one count item per query step
    return out


def _is_floor_receptacle(receptacle_id: Optional[str]) -> bool:
    if not receptacle_id:
        return True
    stem = str(receptacle_id).split('|')[0].strip().lower()
    return stem in ('floor', 'wall', 'ceiling', 'room')


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
    episode: dict, obj_id: str, before_step: int
) -> Optional[int]:
    """Latest step < before_step where obj is in filtered visible_objects (distinguishable)."""
    best = None
    for step in episode.get('steps') or []:
        si = int(step['step'])
        if si >= int(before_step):
            break
        if obj_id in (step.get('visible_objects') or {}):
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
        lab = ego_label_from_world_pose(ag_pos, ag_rot, spec['position'], episode)
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
        enc_idx = _last_distinguishable_sighting(episode, obj_id, int(at_t))
        if enc_idx is None:
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
                episode, partner_id, int(at_t)
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

        # Direct hidden place onto a receptacle landmark
        if _is_floor_receptacle(to_r):
            continue
        frame_steps = _build_id_frame_steps(enc_idx, int(at_t), int(query_step))
        if not _landmark_distinguishable_in_frames(episode, to_r, frame_steps):
            continue
        images = _images_for_steps(episode, frame_steps)
        if not images:
            continue
        new_location = humanize_receptacle(to_r).replace('on/in the ', '')

        # Direct moves: name the destination landmark, ask ego bearing.
        # Do NOT emit receptacle-only "Where is X now?" — that is unanswerable
        # when the move was never witnessed and the question gives no cue.
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


def plan_spatial_updating(
    episode: dict,
    max_items: int = 2,
    *,
    min_delay: int = 2,
    max_delay: Optional[int] = None,
) -> list[PlannedFact]:
    """Bearing at NEW pose after real agent motion; object static and not visible now.

    Includes every navigation frame from encoding through query (same window as SWM).
    """
    if min_delay < 1:
        raise ValueError('spatial_updating min_delay must be at least 1')
    if max_delay is not None and max_delay < min_delay:
        raise ValueError(
            'spatial_updating max_delay must be greater than or equal to min_delay'
        )

    displaced = _displaced_ids(episode)
    out: list[PlannedFact] = []
    for step in episode.get('steps') or []:
        step_idx = int(step['step'])
        for obj_id, mem in (step.get('non_visible_objects') or {}).items():
            if obj_id in displaced:
                continue
            last_seen = mem.get('last_seen_step')
            if last_seen is None or step_idx <= int(last_seen):
                continue
            enc_idx = int(last_seen)
            k = step_idx - enc_idx
            if not _delay_is_allowed(k, min_delay, max_delay):
                continue
            enc = step_by_index(episode, enc_idx)
            if enc is None:
                continue
            if not _has_real_move_between(episode, enc_idx, step_idx):
                continue
            # Queried object must NOT be visible at final pose
            if obj_id in (step.get('visible_objects') or {}):
                continue
            now_edge = find_inferred_edge(step, obj_id)
            past_edge = find_ego_edge(enc, obj_id)
            if not now_edge:
                continue
            label = angle_relation_to_ego_label(now_edge.get('angle_relation'))
            if not label or label not in EGO_DIRECTION_OPTIONS:
                continue
            pre = (
                angle_relation_to_ego_label(past_edge.get('angle_relation'))
                if past_edge
                else None
            )
            if pre and pre not in EGO_DIRECTION_OPTIONS:
                pre = None
            pool, seeds = _ego_pool_with_diagnostics(label, pre or '')
            if pre and pre != label:
                seeds.append('pre_move_bearing')
                seeds.append(_mode_seed('pre_move_bearing', pre))
            if len(pool) < 2:
                continue
            out.append(
                PlannedFact(
                    construct='spatial_updating',
                    status='ok',
                    query_step=step_idx,
                    encoding_step=enc_idx,
                    queried_object_id=obj_id,
                    answer_label=label,
                    answer_source=[
                        f"steps[{step_idx}].edges_inferred[target={obj_id}].angle_relation",
                        f"agent_trajectory[{step_idx}]",
                    ],
                    image_paths=_images_between(episode, enc_idx, step_idx),
                    options_pool=pool,
                    distractor_seeds=seeds,
                    extra={
                        'object_type': object_type_from_id(obj_id, {obj_id: mem}),
                        'pre_move_label': pre,
                        'k': k,
                        'frame_of_reference': 'egocentric',
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
    Sample diversity: prefer world_layout landmarks when they are also visible.
    """
    seen: dict[str, dict] = {}
    for step in episode.get('steps') or []:
        si = int(step['step'])
        for oid, odata in (step.get('visible_objects') or {}).items():
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
            # Prefer catalog/layout pose for snapping (may be on receptacle)
            wp = _object_world_pos(episode, lid)
            if wp is not None:
                seen[lid]['position'] = wp
    # Layout-first ordering, then by first_seen_step for diversity
    items = list(seen.values())
    items.sort(key=lambda r: (not r['from_layout'], r['first_seen_step'], r['obj_id']))
    return items


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
            nid = snap_to_nearest_of(graph, lm['position'], candidate_nodes)
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

    landmarks = _distinguishable_landmark_candidates(episode)
    id_to_node, node_to_name, meta = _build_landmark_node_map(
        graph, landmarks, candidate_nodes=traversed
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
    # Candidate pairs: landmarks whose nodes appear in order on the walk.
    ordered_ids = [lm['obj_id'] for lm in landmarks if lm['obj_id'] in id_to_node]
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
            # Keep MC routes short: decision-point sequences, not full-episode dumps
            if len(sub) > 25:
                continue
            if not was_traversed(sub, traversed):
                continue
            # Require some real translation along the walk
            if not _has_real_move_between(
                episode,
                int(meta[src_id]['first_seen_step']),
                max(int(meta[goal_id]['first_seen_step']), int(meta[src_id]['first_seen_step']) + 1),
            ):
                # Still OK if subpath has multiple distinct nodes (walked)
                if len(sub) < 3:
                    continue

            turns = derive_turns(
                sub, graph, rotation_deg=rot, landmark_at_node=node_to_name
            )
            if not turns:
                continue
            # Route knowledge needs at least one heading change (not landmark parade)
            if not any((t.get('label') or 'straight') != 'straight' for t in turns):
                continue
            answer = format_turn_sequence(turns)
            if not answer or answer.count('→') > 12:
                continue
            pool = [answer]
            seeds: list[str] = []
            for mode in (
                'opposite_direction',
                'wrong_decision_point',
                'extra_turn',
                'no_turn',
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
            images = _img(step_by_index(episode, t0)) + _img(step_by_index(episode, t1))
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
                        'turn_labels': [t.get('label') for t in turns],
                        'object_type': goal,
                        'frame_of_reference': 'egocentric',
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


def plan_survey_based_route_planning(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Direction/distance between landmarks whose graph path was never walked."""
    from cm_benchmark.generation.nav_graph import (
        direction_distance_between_landmarks,
        format_survey_relation,
        is_valid_untraversed_shortcut,
        sanitize_world_layout,
        shortest_path,
        snap_trajectory_to_graph,
        traversed_node_ids,
    )
    import networkx as nx

    # Guard: drop invalid same-region connectivity rows if layout present.
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
    landmarks = _distinguishable_landmark_candidates(episode)
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

    out: list[PlannedFact] = []
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
            rel = direction_distance_between_landmarks(
                meta[src_id]['position'], meta[goal_id]['position'], episode=episode
            )
            if not rel:
                continue
            direction, distance = rel
            source = meta[src_id]['name']
            goal = meta[goal_id]['name']
            answer = format_survey_relation(direction, distance, source_name=source)

            # Distractors: flip direction / alternate distance labels
            opp = {
                'ahead of': 'behind',
                'behind': 'ahead of',
                'to the left of': 'to the right of',
                'to the right of': 'to the left of',
            }.get(direction, direction)
            alt_dists = [
                d
                for d in ('within_reach', 'nearby', 'far', 'beyond')
                if d != distance
            ]
            pool = [answer]
            seeds: list[str] = []
            decoy1 = format_survey_relation(opp, distance, source_name=source)
            if decoy1 not in pool:
                pool.append(decoy1)
                seeds.append('opposite_direction')
                seeds.append(_mode_seed('opposite_direction', decoy1))
            if alt_dists:
                decoy2 = format_survey_relation(
                    direction, alt_dists[0], source_name=source
                )
                if decoy2 not in pool:
                    pool.append(decoy2)
                    seeds.append('wrong_distance')
                    seeds.append(_mode_seed('wrong_distance', decoy2))
            if alt_dists and opp != direction:
                decoy3 = format_survey_relation(opp, alt_dists[-1], source_name=source)
                if decoy3 not in pool:
                    pool.append(decoy3)
                    seeds.append('known_route_answer')
                    seeds.append(_mode_seed('known_route_answer', decoy3))
            if len(pool) < 2:
                continue

            t0 = int(meta[src_id]['first_seen_step'])
            t1 = int(meta[goal_id]['first_seen_step'])
            images = _img(step_by_index(episode, t0)) + _img(step_by_index(episode, t1))
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
                        'condition': 'the queried shortcut was never walked',
                        'source_landmark_id': src_id,
                        'goal_landmark_id': goal_id,
                        'path_nodes': cand,
                        'direction': direction,
                        'distance_label': distance,
                        'object_type': goal,
                        'frame_of_reference': 'allocentric',
                    },
                )
            )
            if len(out) >= max_items:
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


def plan_perspective_taking(episode: dict, max_items: int = 1) -> list[PlannedFact]:
    return [
        PlannedFact(
            construct='perspective_taking',
            status='unsupported',
            reason='edges_object_frame_empty_no_trusted_facing',
        )
    ][:max_items]


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
