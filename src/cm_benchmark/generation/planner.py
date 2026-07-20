"""Task planner: select construct-eligible facts from episode GT (deterministic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from cm_benchmark.generation.constructs import (
    angle_relation_to_ego_label,
    find_allocentric_edge,
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


def _displaced_ids(episode: dict) -> set[str]:
    return {e.get('obj_id') for e in (episode.get('displacement_events') or []) if e.get('obj_id')}


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
            if not label:
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
                    options_pool=[
                        'ahead of you',
                        'to your right',
                        'behind you',
                        'to your left',
                    ],
                    distractor_seeds=['opposite_direction', 'orthogonal_direction'],
                    extra={
                        'object_type': object_type_from_id(obj_id, visible),
                        'angle_relation': edge.get('angle_relation'),
                    },
                )
            )
            if len(out) >= max_items:
                return out
    return out


def plan_spatial_working_memory(episode: dict, max_items: int = 3) -> list[PlannedFact]:
    displaced = _displaced_ids(episode)
    out = []
    for step in episode.get('steps') or []:
        step_idx = int(step['step'])
        non_vis = step.get('non_visible_objects') or {}
        for obj_id, mem in non_vis.items():
            if obj_id in displaced:
                continue  # SWM requires static object
            last_seen = mem.get('last_seen_step')
            if last_seen is None:
                continue
            enc = step_by_index(episode, int(last_seen))
            if enc is None:
                continue
            # Prefer ego edge at encoding (past view); fall back to inferred at query
            edge = find_ego_edge(enc, obj_id) or find_inferred_edge(step, obj_id)
            if not edge:
                continue
            label = angle_relation_to_ego_label(edge.get('angle_relation'))
            if not label:
                continue
            # Need agent movement between encode and query
            if step_idx <= int(last_seen):
                continue
            out.append(
                PlannedFact(
                    construct='spatial_working_memory',
                    status='ok',
                    query_step=step_idx,
                    encoding_step=int(last_seen),
                    queried_object_id=obj_id,
                    answer_label=label,
                    answer_source=[
                        f"steps[{last_seen}].edges_egocentric[target={obj_id}].angle_relation"
                        if find_ego_edge(enc, obj_id)
                        else f"steps[{step_idx}].edges_inferred[target={obj_id}].angle_relation"
                    ],
                    image_paths=_img(enc) + _img(step),
                    options_pool=[
                        'ahead of you',
                        'to your right',
                        'behind you',
                        'to your left',
                    ],
                    distractor_seeds=['opposite_direction', 'orthogonal_direction', 'current_view_answer'],
                    extra={
                        'object_type': object_type_from_id(obj_id, {obj_id: mem}),
                        'angle_relation': edge.get('angle_relation'),
                    },
                )
            )
            if len(out) >= max_items:
                return out
    return out


def plan_invisible_displacement(episode: dict, max_items: int = 3) -> list[PlannedFact]:
    events = episode.get('displacement_events') or []
    if not events:
        return []
    steps = episode.get('steps') or []
    last_step = steps[-1] if steps else None
    last_idx = int(last_step['step']) if last_step else None
    out = []
    # Collect receptacle ids for distractors
    receptacles = set()
    for e in events:
        for key in ('from_receptacle', 'to_receptacle'):
            if e.get(key):
                receptacles.add(e[key])
    layout = episode.get('world_layout') or {}
    for lm in layout.get('landmarks') or []:
        if lm.get('landmark_id'):
            receptacles.add(lm['landmark_id'])

    for ev in events:
        if not ev.get('hidden_during', True):
            continue
        obj_id = ev.get('obj_id')
        to_r = ev.get('to_receptacle')
        from_r = ev.get('from_receptacle')
        if not obj_id:
            continue
        answer = humanize_receptacle(to_r)
        seeds = [humanize_receptacle(from_r)] if from_r and from_r != to_r else []
        for r in receptacles:
            lab = humanize_receptacle(r)
            if lab != answer and lab not in seeds:
                seeds.append(lab)
            if len(seeds) >= 3:
                break
        # Ensure pool has 4 distinct options
        pool = [answer] + seeds
        extras = ['on/in the Shelf', 'on/in the Sofa', 'on/in the Table', 'on the floor']
        for e in extras:
            if e not in pool:
                pool.append(e)
            if len(pool) >= 4:
                break
        images = []
        at_t = ev.get('at_timestep')
        if at_t is not None:
            before = step_by_index(episode, max(0, int(at_t) - 1))
            images.extend(_img(before))
        images.extend(_img(last_step))
        # Object should not be visible at final location in last frame
        if last_step and obj_id in (last_step.get('visible_objects') or {}):
            continue
        out.append(
            PlannedFact(
                construct='invisible_displacement',
                status='ok',
                query_step=last_idx,
                encoding_step=int(at_t) if at_t is not None else None,
                queried_object_id=obj_id,
                answer_label=answer,
                answer_source=[
                    f"displacement_events[obj_id={obj_id}].to_receptacle",
                    f"object_state_track[{obj_id}]",
                ],
                image_paths=[p for p in images if p],
                options_pool=pool[:4],
                distractor_seeds=['original_location', 'nearby_receptacle', 'salient_decoy_location'],
                displacement_event=ev,
                extra={
                    'object_type': object_type_from_id(obj_id),
                    'to_receptacle': to_r,
                    'from_receptacle': from_r,
                },
            )
        )
        if len(out) >= max_items:
            break
    return out


def plan_spatial_updating(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Thin draft: remembered object after agent motion; answer = inferred bearing now."""
    displaced = _displaced_ids(episode)
    out = []
    for step in episode.get('steps') or []:
        step_idx = int(step['step'])
        for obj_id, mem in (step.get('non_visible_objects') or {}).items():
            if obj_id in displaced:
                continue
            last_seen = mem.get('last_seen_step')
            if last_seen is None or step_idx <= int(last_seen):
                continue
            enc = step_by_index(episode, int(last_seen))
            if enc is None:
                continue
            # Require a real move/turn between encode and query
            moved = False
            for s in episode.get('steps') or []:
                si = int(s['step'])
                if int(last_seen) < si <= step_idx:
                    act = (s.get('action') or '').lower()
                    if any(k in act for k in ('move', 'rotate', 'turn', 'look')):
                        moved = True
                        break
            if not moved:
                continue
            now_edge = find_inferred_edge(step, obj_id)
            past_edge = find_ego_edge(enc, obj_id)
            if not now_edge:
                continue
            label = angle_relation_to_ego_label(now_edge.get('angle_relation'))
            if not label:
                continue
            pre = (
                angle_relation_to_ego_label(past_edge.get('angle_relation'))
                if past_edge
                else None
            )
            seeds = [pre] if pre and pre != label else []
            out.append(
                PlannedFact(
                    construct='spatial_updating',
                    status='thin',
                    query_step=step_idx,
                    encoding_step=int(last_seen),
                    queried_object_id=obj_id,
                    answer_label=label,
                    answer_source=[
                        f"steps[{step_idx}].edges_inferred[target={obj_id}].angle_relation"
                    ],
                    image_paths=_img(enc) + _img(step),
                    options_pool=[
                        'ahead of you',
                        'to your right',
                        'behind you',
                        'to your left',
                    ],
                    distractor_seeds=seeds + ['pre_move_bearing', 'opposite_direction'],
                    extra={
                        'object_type': object_type_from_id(obj_id, {obj_id: mem}),
                        'pre_move_label': pre,
                    },
                )
            )
            if len(out) >= max_items:
                return out
    return out


def plan_allocentric_encoding(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """Thin draft: object–object edge without intrinsic facing."""
    out = []
    for step in episode.get('steps') or []:
        step_idx = int(step['step'])
        edge = find_allocentric_edge(step)
        if not edge:
            continue
        label = angle_relation_to_ego_label(edge.get('angle_relation'))
        if not label:
            # Map to relative-to-reference wording
            ar = edge.get('angle_relation') or ['', '', '']
            if ar[0] == 'left':
                label = 'to the left of it'
            elif ar[0] == 'right':
                label = 'to the right of it'
            elif ar[2] == 'front':
                label = 'in front of it'
            elif ar[2] == 'behind':
                label = 'behind it'
            else:
                continue
        src, tgt = edge['source'], edge['target']
        visible = step.get('visible_objects') or {}
        if src not in visible or tgt not in visible:
            continue
        pool = [
            'to the left of it',
            'to the right of it',
            'in front of it',
            'behind it',
        ]
        if label not in pool:
            pool = [label, 'to the left of it', 'to the right of it', 'behind it']
        out.append(
            PlannedFact(
                construct='allocentric_encoding',
                status='thin',
                query_step=step_idx,
                encoding_step=step_idx,
                queried_object_id=tgt,
                reference_object_id=src,
                answer_label=label,
                answer_source=[
                    f"steps[{step_idx}].edges_allocentric[source={src},target={tgt}].angle_relation"
                ],
                image_paths=_img(step),
                options_pool=pool[:4],
                distractor_seeds=['opposite_relation', 'viewer_frame_answer'],
                extra={
                    'object_type': object_type_from_id(tgt, visible),
                    'reference_object': object_type_from_id(src, visible),
                },
            )
        )
        if len(out) >= max_items:
            return out
    return out


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
        # Keep head and mark middle omitted — still a short plan cue, not full memory
        head = labels[: max_tokens - 1]
        head.append('…')
        return head
    return labels


def _region_label(row: dict) -> str:
    return str(row.get('region_type') or row.get('region_id') or 'a room')


def plan_route_knowledge(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """
    Route planning for a short walked source→goal segment.

    Not full-episode turn recall (0→last). Uses consecutive region-trajectory
    change points so the plan is a local A→B route humans can reason about;
    egomotion between those steps is optional evidence, compressed if long.
    """
    traj = episode.get('region_trajectory') or []
    turns = (episode.get('route') or {}).get('turns') or []
    if not turns:
        # Fall back to agent_actions
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
        # Prefer short/medium segments; skip huge jumps (still allow compressed)
        if (t1 - t0) > 40 and len(segment) > 20:
            continue

        collapsed = _collapse_action_labels(segment, max_tokens=8)
        answer = ' → '.join(collapsed)
        source = _region_label(start)
        goal = _region_label(end)

        # Distractors: reverse, swap ends of compressed list, decoy
        rev = ' → '.join(reversed(collapsed))
        swapped = collapsed[:]
        if len(swapped) >= 2:
            swapped[0], swapped[-1] = swapped[-1], swapped[0]
        swap_s = ' → '.join(swapped)
        decoy = 'move_ahead ×2 → rotate_left 90°'
        pool = []
        for p in (answer, rev, swap_s, decoy):
            if p not in pool:
                pool.append(p)
        while len(pool) < 4:
            pool.append(f'decoy_route_{len(pool)}')

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
                distractor_seeds=['reversed_sequence', 'swapped_two_turns'],
                extra={
                    'source': source,
                    'goal': goal,
                    'source_region_id': start.get('region_id'),
                    'goal_region_id': end.get('region_id'),
                    'turn_labels': collapsed,
                    'object_type': goal,
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
    from collections import deque

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


def plan_survey_knowledge(episode: dict, max_items: int = 2) -> list[PlannedFact]:
    """
    Layout-based route *planning* (source→goal), not memorizing walked turns.

    Uses world_layout connectivity; images (if any) are views in source/goal
    regions when available. Egomotion is not the answer surface.
    """
    layout = episode.get('world_layout') or {}
    conn = layout.get('connectivity') or []
    regions = layout.get('regions') or []
    if not conn or len(regions) < 2:
        return [
            PlannedFact(
                construct='survey_knowledge',
                status='unsupported',
                reason='weak_or_missing_layout_connectivity',
            )
        ]

    id_to_label = {
        r.get('region_id'): (r.get('label') or r.get('region_id'))
        for r in regions
        if r.get('region_id')
    }

    # Prefer planning between regions the agent actually visited (grounded views)
    traj = episode.get('region_trajectory') or []
    visited = []
    for row in traj:
        rid = row.get('region_id')
        if rid and rid not in visited:
            visited.append(rid)
    if len(visited) < 2:
        visited = list(id_to_label.keys())[:4]

    out: list[PlannedFact] = []
    # Pairs: non-adjacent in visit order if possible (survey = may need map, not just last hop)
    pairs = []
    for i, a in enumerate(visited):
        for b in visited[i + 1 :]:
            if a != b:
                pairs.append((a, b))
    if not pairs and len(id_to_label) >= 2:
        ids = list(id_to_label.keys())
        pairs = [(ids[0], ids[1])]

    for a, b in pairs:
        result = _bfs_region_path(conn, a, b)
        if not result:
            continue
        path_regions, passages = result
        if len(path_regions) < 2:
            continue
        source = id_to_label.get(a, a)
        goal = id_to_label.get(b, b)
        # Answer: first passage / hop on the planned path
        first_pass = passages[0] if passages else None
        if first_pass:
            answer = f'use {first_pass} toward {id_to_label.get(path_regions[1], path_regions[1])}'
        else:
            answer = f'go to {id_to_label.get(path_regions[1], path_regions[1])}'

        # Distractors from other connectivity edges
        decoys = []
        for c in conn:
            pid = c.get('passage_id')
            to = c.get('to_region')
            if not pid or pid == first_pass:
                continue
            decoys.append(f'use {pid} toward {id_to_label.get(to, to)}')
            if len(decoys) >= 3:
                break
        while len(decoys) < 3:
            decoys.append(f'decoy_passage_{len(decoys)}')
        pool = [answer] + decoys
        seen: set[str] = set()
        uniq = []
        for p in pool:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

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
                status='thin',
                query_step=int(traj[-1]['timestep']) if traj else None,
                encoding_step=int(traj[0]['timestep']) if traj else None,
                answer_label=answer,
                answer_source=[
                    'world_layout.connectivity',
                    f'plan_path[{a}→{b}]={path_regions}',
                ],
                image_paths=dedup[:2],
                options_pool=uniq[:4],
                distractor_seeds=['known_route_answer', 'dead_end_path'],
                extra={
                    'source': source,
                    'goal': goal,
                    'path_regions': path_regions,
                    'passages': passages,
                    'object_type': goal,
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
                reason='no_plannable_source_goal_pair_in_layout',
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
) -> list[PlannedFact]:
    keys = constructs or list(PLANNERS.keys())
    facts: list[PlannedFact] = []
    for key in keys:
        fn = PLANNERS.get(key)
        if fn is None:
            continue
        facts.extend(fn(episode, max_items=max_per_construct))
    return facts
