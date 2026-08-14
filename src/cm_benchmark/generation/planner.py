"""Task planner: select construct-eligible facts from episode GT (deterministic)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

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
    """Image paths for the inclusive step window, in navigation order."""
    paths = []
    for step in episode.get('steps') or []:
        step_idx = int(step['step'])
        if start_step <= step_idx <= end_step and step.get('image_path'):
            paths.append(step['image_path'])
    return paths


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


def _has_real_move_between(episode: dict, t0: int, t1: int) -> bool:
    for s in episode.get('steps') or []:
        si = int(s['step'])
        if t0 < si <= t1:
            act = (s.get('action') or '').lower()
            if any(k in act for k in ('move', 'rotate', 'turn', 'look')):
                return True
    for a in episode.get('agent_actions') or []:
        si = int(a.get('step', -1))
        if t0 < si <= t1:
            act = str(a.get('action') or '').lower()
            if any(k in act for k in ('move', 'rotate', 'turn', 'look')):
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


def plan_invisible_displacement(episode: dict, max_items: int = 3) -> list[PlannedFact]:
    events = episode.get('displacement_events') or []
    if not events:
        return []
    steps = episode.get('steps') or []
    last_step = steps[-1] if steps else None
    last_idx = int(last_step['step']) if last_step else None
    out: list[PlannedFact] = []

    # Real receptacles from events + landmarks only (no invented Shelf/Sofa)
    receptacles: set[str] = set()
    for e in events:
        for key in ('from_receptacle', 'to_receptacle'):
            if e.get(key):
                receptacles.add(e[key])
    layout = episode.get('world_layout') or {}
    for lm in layout.get('landmarks') or []:
        if lm.get('landmark_id'):
            receptacles.add(lm['landmark_id'])

    for ev in events:
        if not ev.get('hidden_during', False):
            continue
        obj_id = ev.get('obj_id')
        to_r = ev.get('to_receptacle')
        from_r = ev.get('from_receptacle')
        at_t = ev.get('at_timestep')
        if not obj_id or at_t is None:
            continue
        # Discriminator: visible → hidden before move
        if not _object_visible_before(episode, obj_id, int(at_t)):
            continue
        # Not visible at final location in last frame
        if last_step and obj_id in (last_step.get('visible_objects') or {}):
            continue

        answer = humanize_receptacle(to_r)
        pool = [answer]
        seeds: list[str] = []

        # A-not-B: original location
        if from_r and from_r != to_r:
            orig = humanize_receptacle(from_r)
            if orig not in pool:
                pool.append(orig)
                seeds.append('original_location')
                seeds.append(_mode_seed('original_location', orig))

        for r in receptacles:
            lab = humanize_receptacle(r)
            if lab != answer and lab not in pool:
                pool.append(lab)
                seeds.append('nearby_receptacle')
                seeds.append(_mode_seed('nearby_receptacle', lab))
            if len(pool) >= 4:
                break

        # Need at least one diagnostic distractor from real scene values
        if len(pool) < 2:
            continue

        images = []
        before = step_by_index(episode, max(0, int(at_t) - 1))
        images.extend(_img(before))
        images.extend(_img(last_step))

        out.append(
            PlannedFact(
                construct='invisible_displacement',
                status='ok',
                query_step=last_idx,
                encoding_step=int(at_t) - 1 if int(at_t) > 0 else 0,
                queried_object_id=obj_id,
                answer_label=answer,
                answer_source=[
                    f"displacement_events[obj_id={obj_id}].to_receptacle",
                    f"object_state_track[{obj_id}]",
                ],
                image_paths=[p for p in images if p],
                options_pool=pool[:4],
                distractor_seeds=seeds,
                displacement_event=ev,
                extra={
                    'object_type': object_type_from_id(obj_id),
                    'to_receptacle': to_r,
                    'from_receptacle': from_r,
                    'frame_of_reference': 'allocentric',
                },
            )
        )
        if len(out) >= max_items:
            break

    if not out and events:
        return [
            PlannedFact(
                construct='invisible_displacement',
                status='unsupported',
                reason='no_event_with_visible_to_hidden_then_hidden_move',
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


def _collapse_action_labels(turns: list[dict], max_tokens: int = 8) -> list[str]:
    """Compress consecutive identical actions: move_ahead ×3 → rotate_right."""
    labels: list[str] = []
    i = 0
    while i < len(turns):
        act = turns[i].get('action') or 'move'
        deg = turns[i].get('degrees')
        j = i + 1
        while j < len(turns) and turns[j].get('action') == act and turns[j].get('degrees') == deg:
            j += 1
        count = j - i
        if deg is not None and 'rotate' in str(act).lower():
            unit = f'{act} {deg}°'
        else:
            unit = str(act)
        labels.append(f'{unit} ×{count}' if count > 1 else unit)
        i = j
    if len(labels) > max_tokens:
        head = labels[: max_tokens - 1]
        head.append('…')
        return head
    return labels


def _region_label(row: dict) -> str:
    return str(row.get('region_type') or row.get('region_id') or 'a room')


def plan_route_knowledge(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Retrace an EXPERIENCED source→goal segment (not full-episode recall)."""
    traj = episode.get('region_trajectory') or []
    turns = (episode.get('route') or {}).get('turns') or []
    if not turns:
        turns = [
            {'step': a.get('step'), 'action': a.get('action'), 'degrees': a.get('degrees')}
            for a in (episode.get('agent_actions') or [])
        ]
    if len(traj) < 2 or not turns:
        return [
            PlannedFact(
                construct='route_knowledge',
                status='unsupported',
                reason='need_region_segments_and_actions_for_source_goal_route',
            )
        ]

    out: list[PlannedFact] = []
    for i in range(len(traj) - 1):
        start, end = traj[i], traj[i + 1]
        if start.get('region_id') == end.get('region_id'):
            continue
        t0, t1 = int(start['timestep']), int(end['timestep'])
        if t1 <= t0:
            continue
        segment = [t for t in turns if t0 <= int(t.get('step', -1)) < t1]
        if len(segment) < 1:
            continue
        if (t1 - t0) > 40 and len(segment) > 20:
            continue

        collapsed = _collapse_action_labels(segment, max_tokens=8)
        answer = ' → '.join(collapsed)
        source = _region_label(start)
        goal = _region_label(end)

        rev = ' → '.join(reversed(collapsed))
        swapped = collapsed[:]
        if len(swapped) >= 2:
            swapped[0], swapped[-1] = swapped[-1], swapped[0]
        swap_s = ' → '.join(swapped)

        pool: list[str] = []
        seeds: list[str] = []
        for lab, mode in (
            (answer, 'correct'),
            (rev, 'reversed_sequence'),
            (swap_s, 'swapped_two_turns'),
        ):
            if lab and lab not in pool:
                pool.append(lab)
                if mode != 'correct':
                    seeds.append(mode)
                    seeds.append(_mode_seed(mode, lab))
        # Another walked segment as decoy if available
        for j in range(len(traj) - 1):
            if j == i:
                continue
            s2, e2 = traj[j], traj[j + 1]
            if s2.get('region_id') == e2.get('region_id'):
                continue
            u0, u1 = int(s2['timestep']), int(e2['timestep'])
            seg2 = [t for t in turns if u0 <= int(t.get('step', -1)) < u1]
            if not seg2:
                continue
            decoy = ' → '.join(_collapse_action_labels(seg2, max_tokens=8))
            if decoy and decoy not in pool:
                pool.append(decoy)
                seeds.append('plausible_but_unwalked_route')
                seeds.append(_mode_seed('plausible_but_unwalked_route', decoy))
                break
        if len(pool) < 2:
            continue

        start_step = step_by_index(episode, t0)
        end_step = step_by_index(episode, t1)
        images = _img(start_step) + _img(end_step)

        out.append(
            PlannedFact(
                construct='route_knowledge',
                status='ok' if (t1 - t0) <= 25 else 'thin',
                query_step=t1,
                encoding_step=t0,
                answer_label=answer,
                answer_source=[
                    f'region_trajectory[{i}:{i + 1}]',
                    f'route.turns[step>={t0} and step<{t1}]',
                ],
                image_paths=images,
                options_pool=pool[:4],
                distractor_seeds=seeds,
                extra={
                    'source': source,
                    'goal': goal,
                    'A': source,
                    'B': goal,
                    'source_region_id': start.get('region_id'),
                    'goal_region_id': end.get('region_id'),
                    'turn_labels': collapsed,
                    'object_type': goal,
                    'frame_of_reference': 'egocentric',
                },
            )
        )
        if len(out) >= max_items:
            break

    if not out:
        return [
            PlannedFact(
                construct='route_knowledge',
                status='unsupported',
                reason='no_short_source_goal_region_segment',
            )
        ]
    return out


def _bfs_region_path(
    connectivity: list[dict], start: str, goal: str
) -> Optional[tuple[list[str], list[str]]]:
    if start == goal:
        return [start], []
    adj: dict[str, list[tuple[str, str]]] = {}
    for c in connectivity:
        a, b = c.get('from_region'), c.get('to_region')
        pid = c.get('passage_id') or f'{a}-{b}'
        if not a or not b:
            continue
        adj.setdefault(a, []).append((b, pid))
        if c.get('bidirectional', True):
            adj.setdefault(b, []).append((a, pid))
    q = deque([(start, [start], [])])
    seen = {start}
    while q:
        node, path, passages = q.popleft()
        for nxt, pid in adj.get(node, []):
            if nxt in seen:
                continue
            npath = path + [nxt]
            npass = passages + [pid]
            if nxt == goal:
                return npath, npass
            seen.add(nxt)
            q.append((nxt, npath, npass))
    return None


def _walked_region_sequences(traj: list[dict]) -> set[tuple[str, ...]]:
    """All contiguous region-id subsequences the agent actually walked."""
    ids = [row.get('region_id') for row in traj if row.get('region_id')]
    # Collapse consecutive duplicates
    collapsed: list[str] = []
    for rid in ids:
        if not collapsed or collapsed[-1] != rid:
            collapsed.append(rid)
    seqs: set[tuple[str, ...]] = set()
    for i in range(len(collapsed)):
        for j in range(i + 1, len(collapsed) + 1):
            seqs.add(tuple(collapsed[i:j]))
    return seqs


def plan_survey_knowledge(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Layout-based novel shortcut; INVALID if it reduces to a walked route."""
    layout = episode.get('world_layout') or {}
    conn = layout.get('connectivity') or []
    regions = layout.get('regions') or []
    traj = episode.get('region_trajectory') or []

    if not conn or len(regions) < 2:
        return [
            PlannedFact(
                construct='survey_knowledge',
                status='unsupported',
                reason='weak_or_missing_layout_connectivity',
            )
        ]

    visited = []
    for row in traj:
        rid = row.get('region_id')
        if rid and rid not in visited:
            visited.append(rid)
    if len(visited) < 2:
        return [
            PlannedFact(
                construct='survey_knowledge',
                status='unsupported',
                reason='need_ge2_observed_regions_for_survey',
            )
        ]

    id_to_label = {
        r.get('region_id'): (r.get('label') or r.get('region_id'))
        for r in regions
        if r.get('region_id')
    }
    walked = _walked_region_sequences(traj)

    out: list[PlannedFact] = []
    pairs = []
    for i, a in enumerate(visited):
        for b in visited[i + 1 :]:
            if a != b:
                pairs.append((a, b))

    # Known walked hop for distractor
    known_route_answer = None
    if len(visited) >= 2:
        result0 = _bfs_region_path(conn, visited[0], visited[1])
        if result0:
            pr, passages = result0
            if passages:
                known_route_answer = (
                    f'use {passages[0]} toward '
                    f'{id_to_label.get(pr[1], pr[1])}'
                )

    for a, b in pairs:
        result = _bfs_region_path(conn, a, b)
        if not result:
            continue
        path_regions, passages = result
        if len(path_regions) < 2:
            continue
        # Discriminator: path must NEVER have been traversed
        if tuple(path_regions) in walked:
            continue
        # Also reject direct one-hop that appears as consecutive visit
        if len(path_regions) == 2 and (path_regions[0], path_regions[1]) in {
            (visited[i], visited[i + 1]) for i in range(len(visited) - 1)
        }:
            continue

        source = id_to_label.get(a, a)
        goal = id_to_label.get(b, b)
        first_pass = passages[0] if passages else None
        if first_pass:
            answer = f'use {first_pass} toward {id_to_label.get(path_regions[1], path_regions[1])}'
        else:
            answer = f'go to {id_to_label.get(path_regions[1], path_regions[1])}'

        pool = [answer]
        seeds: list[str] = []
        if known_route_answer and known_route_answer != answer:
            pool.append(known_route_answer)
            seeds.append('known_route_answer')
            seeds.append(_mode_seed('known_route_answer', known_route_answer))

        for c in conn:
            pid = c.get('passage_id')
            to = c.get('to_region')
            if not pid or pid == first_pass:
                continue
            decoy = f'use {pid} toward {id_to_label.get(to, to)}'
            if decoy not in pool:
                pool.append(decoy)
                seeds.append('dead_end_path')
                seeds.append(_mode_seed('dead_end_path', decoy))
            if len(pool) >= 4:
                break
        if len(pool) < 2:
            continue

        img_paths: list[str] = []
        for row in traj:
            if row.get('region_id') == a:
                img_paths.extend(_img(step_by_index(episode, int(row['timestep']))))
                break
        for row in traj:
            if row.get('region_id') == b:
                img_paths.extend(_img(step_by_index(episode, int(row['timestep']))))
                break
        dedup: list[str] = []
        for p in img_paths:
            if p not in dedup:
                dedup.append(p)

        out.append(
            PlannedFact(
                construct='survey_knowledge',
                status='ok',
                query_step=int(traj[-1]['timestep']) if traj else None,
                encoding_step=int(traj[0]['timestep']) if traj else None,
                answer_label=answer,
                answer_source=[
                    'world_layout.connectivity',
                    f'plan_path[{a}→{b}]={path_regions}',
                    'agent_trajectory (proves path not traversed)',
                ],
                image_paths=dedup[:2],
                options_pool=pool[:4],
                distractor_seeds=seeds,
                extra={
                    'source': source,
                    'goal': goal,
                    'A': source,
                    'B': goal,
                    'path_regions': path_regions,
                    'passages': passages,
                    'object_type': goal,
                    'frame_of_reference': 'allocentric',
                },
            )
        )
        if len(out) >= max_items:
            break

    if not out:
        return [
            PlannedFact(
                construct='survey_knowledge',
                status='unsupported',
                reason='no_novel_untraversed_layout_path',
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
    'survey_knowledge': plan_survey_knowledge,
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
