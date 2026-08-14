"""Build candidate items from PlannedFact (templates + verbose preamble)."""

from __future__ import annotations

from typing import Optional

from cm_benchmark.generation.constructs import (
    OPPOSITE,
    ORTHOGONAL,
    frame_sequence_cue,
    object_type_from_id,
    select_template,
    step_by_index,
)
from cm_benchmark.generation.episode_io import environment_of, scene_id_of
from cm_benchmark.generation.planner import PlannedFact
from cm_benchmark.generation.schema import (
    CONSTRUCT_CLASS,
    default_frame_of_reference,
    CandidateItem,
)

# Constructs that carry pose per *item frame*. The full episode trajectory stays
# in episode GT; items only mirror the poses of the frames they show.
_TRAJECTORY_HOOKS = frozenset(
    {
        'spatial_working_memory',
        'spatial_updating',
        'route_knowledge',
        'survey_knowledge',
    }
)
# Actions inside the item's encoding -> query window only. route_knowledge is
# excluded: its answer *is* the action sequence, so the raw list would leak it.
_ACTIONS_HOOKS = frozenset(
    {
        'spatial_working_memory',
        'spatial_updating',
    }
)


def _shuffle_options(correct: str, pool: list[str], seeds: list[str]) -> tuple[dict, dict, str]:
    """Return options dict, distractor_rationale, answer key (A/B/C/D).

    Prefers labels already realized in ``pool``. Seed *names* that are failure-mode
    tags (not scene labels) only help classify opposite/orthogonal ego directions.
    """
    labels: list[str] = []
    for p in pool:
        if p and p not in labels:
            labels.append(p)
    if correct and correct not in labels:
        labels.insert(0, correct)

    failure_mode_tags = {
        'opposite_direction',
        'orthogonal_direction',
        'current_view_answer',
        'adjacent_step_answer',
        'off_by_one_count',
        'original_location',
        'nearby_receptacle',
        'salient_decoy_location',
        'pre_move_bearing',
        'opposite_turn_bearing',
        'over_rotation_bearing',
        'opposite_relation',
        'wrong_reference_object',
        'viewer_frame_answer',
        'reversed_sequence',
        'swapped_two_turns',
        'plausible_but_unwalked_route',
        'known_route_answer',
        'dead_end_path',
        'longer_traversed_detour',
    }

    for seed in seeds:
        if seed in failure_mode_tags:
            if seed == 'opposite_direction' and correct in OPPOSITE:
                cand = OPPOSITE[correct]
                if cand not in labels:
                    labels.append(cand)
            elif seed == 'orthogonal_direction' and correct in ORTHOGONAL:
                cand = ORTHOGONAL[correct]
                if cand not in labels:
                    labels.append(cand)
            continue
        if seed and seed not in labels:
            labels.append(seed)

    labels = labels[:4]
    if correct not in labels:
        if labels:
            labels[0] = correct
        else:
            labels = [correct]

    keys = ['A', 'B', 'C', 'D'][: len(labels)]
    idx = sum(ord(c) for c in correct) % len(labels)
    ordered = labels[idx:] + labels[:idx]
    options = {keys[i]: ordered[i] for i in range(len(ordered))}
    answer_key = next(k for k, v in options.items() if v == correct)

    # Map concrete distractor labels back to failure-mode names when planner
    # stored them as (label -> mode) in seeds via "mode::label" OR when the
    # label matches opposite/orthogonal of the correct answer.
    seed_label_to_mode: dict[str, str] = {}
    for seed in seeds:
        if '::' in seed:
            mode, lab = seed.split('::', 1)
            seed_label_to_mode[lab] = mode

    rationale = {}
    for k, v in options.items():
        if k == answer_key:
            rationale[k] = 'correct'
        elif v in seed_label_to_mode:
            rationale[k] = seed_label_to_mode[v]
        elif correct in OPPOSITE and v == OPPOSITE.get(correct):
            rationale[k] = 'opposite_direction'
        elif correct in ORTHOGONAL and v == ORTHOGONAL.get(correct):
            rationale[k] = 'orthogonal_direction'
        else:
            rationale[k] = 'diagnostic_decoy'
    return options, rationale, answer_key


def _usable_display_name(planned: Optional[str], obj_id: Optional[str]) -> str:
    """Prefer a planned category only when it is not a placeholder like Undefined."""
    if planned is not None and str(planned).strip():
        probe = object_type_from_id('_', {'_': {'category': planned}})
        if probe == str(planned).strip():
            return probe
    if obj_id:
        return object_type_from_id(obj_id)
    return 'object'


def _core_question(fact: PlannedFact) -> str:
    extra = fact.extra or {}
    mode = extra.get('template_mode')
    tmpl_index = int(extra.get('template_index') or 0)
    tmpl = select_template(fact.construct, template_mode=mode, index=tmpl_index)

    object_type = _usable_display_name(extra.get('object_type'), fact.queried_object_id)
    reference = _usable_display_name(
        extra.get('reference_object') or extra.get('reference_entity'),
        fact.reference_object_id,
    )
    source = extra.get('source') or extra.get('A') or 'the start'
    goal = extra.get('goal') or extra.get('B') or 'the goal'
    k = extra.get('k')
    if k is None and fact.query_step is not None and fact.encoding_step is not None:
        k = max(1, int(fact.query_step) - int(fact.encoding_step))
    k = max(1, int(k or 1))
    object_category = extra.get('object_category') or object_type

    body = tmpl.format(
        object_type=object_type,
        object=object_type,
        object_category=object_category,
        objects=object_category,
        reference_object=reference,
        reference_entity=reference,
        source=source,
        goal=goal,
        A=source,
        B=goal,
        k=k,
    )
    n_images = len(fact.image_paths or [])
    cue = frame_sequence_cue(n_images)
    return f'{cue}{body}' if cue else body


def build_verbose_preamble(episode: dict, fact: PlannedFact) -> str:
    """GT-grounded scene description that must not leak the answer label."""
    step_idx = fact.encoding_step if fact.encoding_step is not None else fact.query_step
    step = step_by_index(episode, step_idx) if step_idx is not None else None
    if step is None and episode.get('steps'):
        step = episode['steps'][0]

    parts = []
    n_images = len(fact.image_paths or [])
    extra = fact.extra or {}
    if n_images > 1:
        parts.append(frame_sequence_cue(n_images).strip())
        if fact.construct == 'spatial_updating':
            k = extra.get('k') or max(
                1, (fact.query_step or 1) - (fact.encoding_step or 0)
            )
            parts.append(
                f'Encode the object in the first image, then follow {k} navigation '
                "step(s). Answer the object's bearing from your pose in the last image."
            )
        elif fact.construct == 'spatial_working_memory':
            k = extra.get('k') or max(
                1, (fact.query_step or 1) - (fact.encoding_step or 0)
            )
            parts.append(
                f'The answer refers to the view from {k} steps before the last image, '
                'when the object was still visible.'
            )
        elif fact.construct == 'invisible_displacement':
            parts.append(
                'The object moves while hidden; answer its location after the last image.'
            )
        elif fact.construct == 'route_knowledge':
            src = extra.get('source', 'the start')
            goal = extra.get('goal', 'the goal')
            parts.append(
                f'Retrace the walked route from {src} to {goal}. '
                'The images show views near the start and goal.'
            )
        elif fact.construct == 'survey_knowledge':
            src = extra.get('source', 'the start')
            goal = extra.get('goal', 'the goal')
            parts.append(
                f'Use the layout to find a connection from {src} to {goal} '
                'that was not walked as an experienced route.'
            )

    room = None
    if step:
        room = step.get('current_room_type') or step.get('current_room')
    if room:
        parts.append(f'You are looking around a space identified as {room}.')

    visible = (step or {}).get('visible_objects') or {}
    queried = fact.queried_object_id
    answer = (fact.answer_label or '').lower()

    others = []
    for oid, odata in visible.items():
        if oid == queried:
            continue
        cat = object_type_from_id(oid, {oid: odata})
        rel_bits = []
        for edge in step.get('edges_egocentric') or []:
            if edge.get('target') != oid:
                continue
            ar = edge.get('angle_relation') or []
            from cm_benchmark.generation.constructs import angle_relation_to_ego_label

            lab = angle_relation_to_ego_label(ar)
            if lab and lab.lower() == answer:
                continue
            if ar and ar[0]:
                rel_bits.append(f'to your {ar[0]}')
            elif ar and ar[2]:
                z = 'ahead' if ar[2] == 'front' else ar[2]
                rel_bits.append(z + ' of you' if z == 'ahead' else f'{z} you')
            break
        if rel_bits:
            others.append(f'a {cat} ({rel_bits[0]})')
        else:
            others.append(f'a {cat}')

    if others:
        if len(others) == 1:
            parts.append(f'Also visible is {others[0]}.')
        else:
            parts.append(
                'Among other things, you can see '
                + ', '.join(others[:-1])
                + f', and {others[-1]}.'
            )

    if queried:
        qtype = _usable_display_name(extra.get('object_type'), queried)
        if queried in visible:
            parts.append(f'There is also a {qtype} in the scene.')
        else:
            parts.append(f'You previously noticed a {qtype}.')

    non_vis = (step or {}).get('non_visible_objects') or {}
    nv_cats = []
    for oid, odata in list(non_vis.items())[:4]:
        if oid == queried:
            continue
        nv_cats.append(object_type_from_id(oid, {oid: odata}))
    if nv_cats:
        parts.append(
            'Some objects are no longer in view, including '
            + ', '.join(nv_cats)
            + '.'
        )

    preamble = ' '.join(parts).strip()
    if fact.answer_label and fact.answer_label.lower() in preamble.lower():
        preamble = preamble.replace(fact.answer_label, '[…]')
        preamble = preamble.replace(fact.answer_label.lower(), '[…]')
    return preamble or 'You observe several objects in the environment.'


def build_question(episode: dict, fact: PlannedFact, style: str) -> str:
    core = _core_question(fact)
    if style == 'verbose':
        preamble = build_verbose_preamble(episode, fact)
        return f'{preamble} {core}'
    return core


def _frame_of_reference_for_fact(fact: PlannedFact) -> str:
    extra = fact.extra or {}
    if extra.get('frame_of_reference'):
        return str(extra['frame_of_reference'])
    return default_frame_of_reference(fact.construct)


def _step_of(entry) -> Optional[int]:
    if not isinstance(entry, dict):
        return None
    try:
        return int(entry['step'])
    except (KeyError, TypeError, ValueError):
        return None


def _frame_poses(episode: dict, fact: PlannedFact) -> list[dict]:
    """Pose per frame shown by the item, in image order."""
    traj = [e for e in (episode.get('agent_trajectory') or []) if isinstance(e, dict)]
    by_path: dict[str, dict] = {}
    for entry in traj:
        path = entry.get('image_path')
        if path and path not in by_path:
            by_path[path] = entry

    poses = [dict(by_path[p]) for p in (fact.image_paths or []) if p in by_path]
    if poses:
        return poses

    wanted = {int(s) for s in (fact.encoding_step, fact.query_step) if s is not None}
    return [dict(e) for e in traj if _step_of(e) in wanted]


def _window_actions(episode: dict, fact: PlannedFact) -> list:
    """Actions performed between the encoding frame and the query frame."""
    if fact.query_step is None:
        return []
    t1 = int(fact.query_step)
    t0 = int(fact.encoding_step) if fact.encoding_step is not None else t1 - 1
    out = []
    for entry in episode.get('agent_actions') or []:
        step = _step_of(entry)
        if step is not None and t0 < step <= t1:
            out.append(dict(entry))
    return out


def _attach_schema_hooks(episode: dict, fact: PlannedFact, item: CandidateItem) -> None:
    if fact.construct in _TRAJECTORY_HOOKS:
        item.agent_trajectory = _frame_poses(episode, fact) or None
    if fact.construct in _ACTIONS_HOOKS:
        item.agent_actions = _window_actions(episode, fact) or None
    if fact.construct == 'invisible_displacement' and fact.displacement_event:
        item.displacement_event = dict(fact.displacement_event)


def fact_to_items(
    episode: dict,
    fact: PlannedFact,
    *,
    base_id: str,
    styles: tuple[str, ...] = ('concise', 'verbose'),
) -> list[CandidateItem]:
    scene = scene_id_of(episode)
    env = environment_of(episode)
    fo_r = _frame_of_reference_for_fact(fact)

    if fact.status == 'unsupported':
        return [
            CandidateItem.unsupported(
                item_id=f'{base_id}_{fact.construct}_unsupported',
                construct=fact.construct,
                scene_id=scene,
                environment=env,
                reason=fact.reason or 'unsupported',
            )
        ]

    if not fact.answer_label:
        return [
            CandidateItem.unsupported(
                item_id=f'{base_id}_{fact.construct}_blank',
                construct=fact.construct,
                scene_id=scene,
                environment=env,
                reason=fact.reason or 'no_answer_label',
            )
        ]

    options, rationale, answer_key = _shuffle_options(
        fact.answer_label, fact.options_pool or [], fact.distractor_seeds or []
    )

    items: list[CandidateItem] = []
    ids = []
    for style in styles:
        iid = f'{base_id}_{fact.construct}_{fact.query_step}_{fact.queried_object_id}_{style}'
        iid = iid.replace('|', '_').replace(' ', '_')
        ids.append(iid)

    for i, style in enumerate(styles):
        paired = ids[1 - i] if len(ids) == 2 else None
        item = CandidateItem(
            item_id=ids[i],
            construct=fact.construct,
            class_=CONSTRUCT_CLASS[fact.construct],
            frame_of_reference=fo_r,
            environment=env,
            scene_id=scene,
            image_paths=list(fact.image_paths or []),
            question=build_question(episode, fact, style),
            options=options,
            answer=answer_key,
            answer_source=list(fact.answer_source or []),
            queried_object_id=fact.queried_object_id,
            distractor_rationale=rationale,
            status=fact.status,
            query_step=fact.query_step,
            encoding_step=fact.encoding_step,
            question_style=style,
            paired_item_id=paired,
            displacement_event=fact.displacement_event,
            difficulty=(fact.extra or {}).get('k'),
        )
        _attach_schema_hooks(episode, fact, item)
        items.append(item)
    return items


def build_items_from_facts(
    episode: dict,
    facts: list[PlannedFact],
    *,
    episode_tag: str = 'ep',
    styles: tuple[str, ...] = ('concise', 'verbose'),
) -> list[CandidateItem]:
    items: list[CandidateItem] = []
    for i, fact in enumerate(facts):
        items.extend(
            fact_to_items(episode, fact, base_id=f'{episode_tag}_{i:03d}', styles=styles)
        )
    return items
