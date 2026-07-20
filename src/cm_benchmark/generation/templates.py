"""Build candidate items from PlannedFact (templates + verbose preamble)."""

from __future__ import annotations

from typing import Optional

from cm_benchmark.generation.constructs import (
    CONSTRUCT_TEMPLATES,
    OPPOSITE,
    ORTHOGONAL,
    frame_sequence_cue,
    object_type_from_id,
    step_by_index,
)
from cm_benchmark.generation.episode_io import environment_of, scene_id_of
from cm_benchmark.generation.planner import PlannedFact
from cm_benchmark.generation.schema import CONSTRUCT_CLASS, CONSTRUCT_FOR, CandidateItem


def _shuffle_options(correct: str, pool: list[str], seeds: list[str]) -> tuple[dict, dict, str]:
    """Return options dict, distractor_rationale, answer key (A/B/C/D)."""
    labels = []
    for p in pool:
        if p and p not in labels:
            labels.append(p)
    # Fill from opposite/orthogonal of correct when needed
    if correct and correct not in labels:
        labels.insert(0, correct)
    for seed in seeds:
        if seed in ('opposite_direction',) and correct in OPPOSITE:
            cand = OPPOSITE[correct]
            if cand not in labels:
                labels.append(cand)
        if seed in ('orthogonal_direction',) and correct in ORTHOGONAL:
            cand = ORTHOGONAL[correct]
            if cand not in labels:
                labels.append(cand)
        if seed and seed not in labels and seed in (
            'ahead of you',
            'to your right',
            'behind you',
            'to your left',
        ):
            labels.append(seed)
    for filler in (
        'ahead of you',
        'to your right',
        'behind you',
        'to your left',
        'on the floor',
        'on/in the Shelf',
        'on/in the Sofa',
        'on/in the Table',
    ):
        if len(labels) >= 4:
            break
        if filler not in labels:
            labels.append(filler)
    labels = labels[:4]
    if correct not in labels:
        labels[0] = correct
    # Stable order: put correct at a deterministic index from hash of label
    keys = ['A', 'B', 'C', 'D']
    # Rotate so correct is not always A
    idx = sum(ord(c) for c in correct) % len(labels)
    ordered = labels[idx:] + labels[:idx]
    options = {keys[i]: ordered[i] for i in range(len(ordered))}
    answer_key = next(k for k, v in options.items() if v == correct)
    rationale = {}
    for k, v in options.items():
        if k == answer_key:
            rationale[k] = 'correct'
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
    templates = CONSTRUCT_TEMPLATES.get(fact.construct) or ['(no template)']
    tmpl = templates[0]
    extra = fact.extra or {}
    object_type = _usable_display_name(extra.get('object_type'), fact.queried_object_id)
    reference = _usable_display_name(
        extra.get('reference_object'), fact.reference_object_id
    )
    body = tmpl.format(
        object_type=object_type,
        object=object_type,
        reference_object=reference,
        reference_entity=reference,
        source=extra.get('source', 'the start'),
        goal=extra.get('goal', 'the goal'),
        A='A',
        B='B',
        k=max(1, (fact.query_step or 1) - (fact.encoding_step or 0)),
        objects=object_type,
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
    if n_images > 1:
        parts.append(frame_sequence_cue(n_images).strip())
        if fact.construct == 'spatial_updating':
            parts.append(
                'Use the earlier image to encode the object, then answer from your pose in the last image.'
            )
        elif fact.construct == 'spatial_working_memory':
            parts.append(
                'The answer refers to an earlier view when the object was visible, not only the last image.'
            )
        elif fact.construct == 'invisible_displacement':
            parts.append(
                'The object moves while hidden; answer its location after the last image.'
            )
        elif fact.construct == 'route_knowledge':
            src = (fact.extra or {}).get('source', 'the start')
            goal = (fact.extra or {}).get('goal', 'the goal')
            parts.append(
                f'Plan a short route from {src} to {goal} (not the whole episode). '
                'The images show views near the start and goal.'
            )
        elif fact.construct == 'survey_knowledge':
            src = (fact.extra or {}).get('source', 'the start')
            goal = (fact.extra or {}).get('goal', 'the goal')
            parts.append(
                f'Use the layout to plan from {src} to {goal}; '
                'do not rely on memorizing every walked step.'
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
        # Optional non-answer relation for other objects only
        rel_bits = []
        for edge in step.get('edges_egocentric') or []:
            if edge.get('target') != oid:
                continue
            ar = edge.get('angle_relation') or []
            # Skip if this text would equal the answer (should not for other objs usually)
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
        # Mention other objects first
        if len(others) == 1:
            parts.append(f'Also visible is {others[0]}.')
        else:
            parts.append(
                'Among other things, you can see '
                + ', '.join(others[:-1])
                + f', and {others[-1]}.'
            )

    # Mention queried object late without stating its answer relation
    if queried:
        qtype = _usable_display_name((fact.extra or {}).get('object_type'), queried)
        if queried in visible:
            parts.append(f'There is also a {qtype} in the scene.')
        else:
            parts.append(f'You previously noticed a {qtype}.')

    # Non-visible competitors (categories only)
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
    # Safety: strip answer substring if it slipped in
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


def fact_to_items(
    episode: dict,
    fact: PlannedFact,
    *,
    base_id: str,
    styles: tuple[str, ...] = ('concise', 'verbose'),
) -> list[CandidateItem]:
    scene = scene_id_of(episode)
    env = environment_of(episode)

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
        items.append(
            CandidateItem(
                item_id=ids[i],
                construct=fact.construct,
                class_=CONSTRUCT_CLASS[fact.construct],
                frame_of_reference=CONSTRUCT_FOR[fact.construct],
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
            )
        )
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
