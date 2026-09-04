"""Build candidate items from PlannedFact (templates + verbose preamble)."""

from __future__ import annotations

from typing import Optional

from cm_benchmark.generation.constructs import (
    OPPOSITE,
    ORTHOGONAL,
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
        'survey_based_route_planning',
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
# Taxonomy shared_rules exception: verbose may name other static scene objects
# WITHOUT pairing them to direction/distance/relation (naming without relating).
_VERBOSE_SCENE_DETAIL_CONSTRUCTS = frozenset(
    {
        'spatial_working_memory',
        'invisible_displacement',
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
        # "mode::label" seeds are for distractor_rationale only, not option text.
        if '::' in seed:
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
    # Precomputed by planner: '' if category unique, phrase if duplicate+landmark,
    # or omitted (treated as '') for constructs without referring expressions.
    disambiguator = extra.get('disambiguator') or ''
    reference = _usable_display_name(
        extra.get('reference_object') or extra.get('reference_entity'),
        fact.reference_object_id,
    )
    source = extra.get('source') or extra.get('A') or 'the start'
    goal = extra.get('goal') or extra.get('B') or 'the goal'
    landmark_a = extra.get('A') or source
    landmark_b = extra.get('B') or goal
    landmark_c = extra.get('C') or object_type
    k = extra.get('k')
    if k is None and fact.query_step is not None and fact.encoding_step is not None:
        k = max(1, int(fact.query_step) - int(fact.encoding_step))
    k = max(1, int(k or 1))
    object_category = extra.get('object_category') or object_type
    new_location = extra.get('new_location') or extra.get('to_receptacle') or 'a surface'
    other_object_type = extra.get('other_object_type') or 'another object'
    condition = extra.get('condition') or 'a passage is closed'
    relation = extra.get('relation') or 'left'

    body = tmpl.format(
        object_type=object_type,
        object=object_type,
        object_category=object_category,
        objects=object_category,
        other_object_type=other_object_type,
        reference_object=reference,
        reference_entity=reference,
        source=source,
        goal=goal,
        A=landmark_a,
        B=landmark_b,
        C=landmark_c,
        k=k,
        new_location=new_location,
        condition=condition,
        relation=relation,
        disambiguator=disambiguator,
    )
    # Online sequential protocol: no multi-image "time order" cue; templates
    # already situate "now" / "{k} steps ago" relative to the live nav stream.
    return body


def build_verbose_preamble(episode: dict, fact: PlannedFact) -> str:
    """GT-grounded scene description that must not leak the answer label.

    Per taxonomy shared_rules: only SWM / invisible_displacement verbose may
    name other static scene objects, and never with direction/distance/relation.
    """
    step_idx = fact.encoding_step if fact.encoding_step is not None else fact.query_step
    step = step_by_index(episode, step_idx) if step_idx is not None else None
    if step is None and episode.get('steps'):
        step = episode['steps'][0]

    parts = []
    extra = fact.extra or {}

    # Online protocol cues that are NOT already in the core template.
    # SWM / spatial_updating embed {k} in CONSTRUCT_TEMPLATES — do not restate.
    if fact.construct == 'invisible_displacement':
        mode = extra.get('template_mode')
        if mode == 'swap':
            parts.append(
                'Two objects exchanged places while hidden. The question names '
                'the partner object you saw earlier — not a receptacle.'
            )
        else:
            parts.append(
                'The object relocated while hidden. Use the destination named '
                'in the question and report its bearing from your current pose.'
            )
    elif fact.construct == 'route_knowledge':
        src = extra.get('source', 'the start')
        goal = extra.get('goal', 'the goal')
        parts.append(
            f'Retrace the walked route from {src} to {goal} based on the path '
            'you have followed so far. Choose the matching turn sequence.'
        )
    elif fact.construct == 'survey_based_route_planning':
        src = extra.get('source', 'the start')
        goal = extra.get('goal', 'the goal')
        mode = extra.get('template_mode') or 'direction_distance'
        if mode == 'conditional_detour':
            cond = extra.get('condition') or 'a passage is closed'
            parts.append(
                f'Use the layout to decide the first heading from {src} toward {goal} '
                f'under the recorded condition ({cond}). This is not a turn sequence.'
            )
        else:
            parts.append(
                f'Use the layout to judge direction and distance of {goal} relative '
                f'to {src}. The connecting path was never walked.'
            )
    elif fact.construct == 'perspective_taking':
        parts.append(
            f"Adopt an imagined viewpoint at {extra.get('A', 'landmark A')}, "
            f"facing {extra.get('B', 'landmark B')}, then locate "
            f"{extra.get('C', 'landmark C')}."
        )

    room = None
    if step:
        room = step.get('current_room_type') or step.get('current_room')
    if room:
        parts.append(f'You are looking around a space identified as {room}.')

    visible = (step or {}).get('visible_objects') or {}
    queried = fact.queried_object_id
    allow_scene_detail = fact.construct in _VERBOSE_SCENE_DETAIL_CONSTRUCTS

    # Naming without relating: type names only — never edges_egocentric / bearings.
    # Union encode + query (and any shown intermediate) so multi-frame items still
    # get scene detail when the encoding view is sparse.
    if allow_scene_detail:
        detail_steps = []
        for s in (fact.encoding_step, fact.query_step):
            if s is None:
                continue
            st = step_by_index(episode, int(s))
            if st is not None:
                detail_steps.append(st)
        if not detail_steps and step is not None:
            detail_steps = [step]

        others: list[str] = []
        seen_types: set[str] = set()
        for st in detail_steps:
            for oid, odata in (st.get('visible_objects') or {}).items():
                if oid == queried:
                    continue
                typ = object_type_from_id(oid, {oid: odata})
                key = typ.lower()
                if key in seen_types:
                    continue
                seen_types.add(key)
                others.append(f'a {typ}')

        if others:
            if len(others) == 1:
                parts.append(f'Also visible is {others[0]}.')
            else:
                parts.append(
                    'Among other things, you can see '
                    + ', '.join(others[:-1])
                    + f', and {others[-1]}.'
                )

        nv_cats: list[str] = []
        nv_seen: set[str] = set()
        for st in detail_steps:
            for oid, odata in list((st.get('non_visible_objects') or {}).items())[:4]:
                if oid == queried:
                    continue
                typ = object_type_from_id(oid, {oid: odata})
                key = typ.lower()
                if key in nv_seen:
                    continue
                nv_seen.add(key)
                nv_cats.append(typ)
                if len(nv_cats) >= 4:
                    break
            if len(nv_cats) >= 4:
                break
        if nv_cats:
            parts.append(
                'Some objects are no longer in view, including '
                + ', '.join(nv_cats)
                + '.'
            )

    if queried:
        qtype = _usable_display_name(extra.get('object_type'), queried)
        if queried in visible:
            parts.append(f'There is also a {qtype} in the scene.')
        else:
            parts.append(f'You previously noticed a {qtype}.')

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


def _compact_pose(entry: dict) -> Optional[dict]:
    """Item-level pose: step + xz + yaw only (drop pitch/roll, y, image_path)."""
    step = _step_of(entry)
    if 'x' in entry and 'z' in entry and 'heading' in entry:
        try:
            out = {
                'x': float(entry['x']),
                'z': float(entry['z']),
                'heading': float(entry['heading']),
            }
        except (TypeError, ValueError):
            return None
        if step is not None:
            out['step'] = step
        return out

    pos = entry.get('position') or entry.get('pos')
    rot = entry.get('rotation') or entry.get('rot')
    if pos is None or rot is None:
        return None
    try:
        x = float(pos[0])
        z = float(pos[2])
        heading = float(rot[1] if len(rot) > 1 else rot[0])
    except (TypeError, ValueError, IndexError):
        return None
    out = {'x': x, 'z': z, 'heading': heading}
    if step is not None:
        out['step'] = step
    return out


def _compact_action(entry: dict) -> Optional[dict]:
    """Item-level action: step + label (+ degrees when present)."""
    action = entry.get('action')
    if action is None:
        return None
    out: dict = {'action': action}
    step = _step_of(entry)
    if step is not None:
        out['step'] = step
    if entry.get('degrees') is not None:
        out['degrees'] = entry['degrees']
    return out


def _frame_poses(episode: dict, fact: PlannedFact) -> list[dict]:
    """Compact pose per frame shown by the item, in image order."""
    traj = [e for e in (episode.get('agent_trajectory') or []) if isinstance(e, dict)]
    by_path: dict[str, dict] = {}
    for entry in traj:
        path = entry.get('image_path')
        if path and path not in by_path:
            by_path[path] = entry

    poses: list[dict] = []
    for path in fact.image_paths or []:
        entry = by_path.get(path)
        if entry is None:
            continue
        compact = _compact_pose(entry)
        if compact is not None:
            poses.append(compact)
    if poses:
        return poses

    wanted = {int(s) for s in (fact.encoding_step, fact.query_step) if s is not None}
    out = []
    for e in traj:
        if _step_of(e) not in wanted:
            continue
        compact = _compact_pose(e)
        if compact is not None:
            out.append(compact)
    return out


def _window_actions(episode: dict, fact: PlannedFact) -> list:
    """Actions between encoding and query frames (compact; no pose echo)."""
    if fact.query_step is None:
        return []
    t1 = int(fact.query_step)
    t0 = int(fact.encoding_step) if fact.encoding_step is not None else t1 - 1
    out = []
    for entry in episode.get('agent_actions') or []:
        if not isinstance(entry, dict):
            continue
        step = _step_of(entry)
        if step is not None and t0 < step <= t1:
            compact = _compact_action(entry)
            if compact is not None:
                out.append(compact)
    return out


def _slide_context(fact: PlannedFact) -> dict:
    """Compact construct-specific fields for slides / review (not scored)."""
    extra = fact.extra or {}
    keys = (
        'A',
        'B',
        'C',
        'source',
        'goal',
        'object_type',
        'object_category',
        'template_mode',
        'new_location',
        'other_object_type',
        'disambiguator',
        'k',
        'load_n_objects',
        'condition',
        'direction',
        'distance_label',
        'hop_count',
        'min_hop_count',
    )
    ctx = {k: extra[k] for k in keys if extra.get(k) is not None and extra.get(k) != ''}
    return ctx


def infer_image_roles(fact: PlannedFact) -> list[str]:
    """Human-readable captions aligned with ``fact.image_paths`` for slides."""
    extra = fact.extra or {}
    paths = list(fact.image_paths or [])
    n = len(paths)
    if n == 0:
        return []
    preset = extra.get('image_roles')
    if isinstance(preset, list) and len(preset) == n:
        return [str(r) for r in preset]

    c = fact.construct
    ot = extra.get('object_type') or 'object'
    if c == 'perspective_taking':
        tags = [
            f"A · stand here ({extra.get('A') or 'landmark'})",
            f"B · face toward ({extra.get('B') or 'landmark'})",
            f"C · locate ({extra.get('C') or ot})",
        ]
        return [tags[i] if i < len(tags) else f'view {i}' for i in range(n)]
    if c in ('route_knowledge', 'survey_based_route_planning'):
        src = extra.get('source') or 'source'
        goal = extra.get('goal') or 'goal'
        if n == 1:
            return [f'source/goal · {src} → {goal}']
        roles = [f'source · {src}']
        if n >= 2:
            roles.append(f'goal · {goal}')
        while len(roles) < n:
            roles.append(f'view {len(roles)}')
        return roles
    if c == 'egocentric_encoding':
        return [f'current view · find {ot}']
    if c == 'allocentric_encoding':
        ref = extra.get('reference_object') or extra.get('B') or 'reference'
        return [f'current view · {ot} relative to {ref}']
    if c in (
        'spatial_working_memory',
        'spatial_updating',
        'invisible_displacement',
    ):
        if n == 1:
            return [f'encode/query · {ot}']
        roles = [f'encode · last clear view of {ot}']
        for i in range(1, n - 1):
            roles.append(f'navigation · frame {i}')
        roles.append('query · now')
        return roles
    return [f'frame {i}' for i in range(n)]


def slide_readout(construct: str, context: Optional[dict] = None) -> str:
    """One-line how-to-read the images for a construct (slide caption)."""
    ctx = context or {}
    if construct == 'perspective_taking':
        a, b, c = ctx.get('A'), ctx.get('B'), ctx.get('C')
        return (
            f'Imagined pose: stand at {a}, face {b}, report where {c} is — '
            'not the camera’s egocentric answer.'
        )
    if construct == 'route_knowledge':
        return (
            f"Landmark views of source ({ctx.get('source')}) and goal "
            f"({ctx.get('goal')}). The answer is the walked turn sequence, "
            'not something readable from these two stills alone.'
        )
    if construct == 'survey_based_route_planning':
        mode = ctx.get('template_mode') or 'direction_distance'
        if mode == 'conditional_detour':
            return (
                f"Layout views of {ctx.get('source')} → {ctx.get('goal')}. "
                f"Answer uses the recorded closure ({ctx.get('condition')}); "
                'not a turn sequence.'
            )
        return (
            f"Layout views of source ({ctx.get('source')}) and goal "
            f"({ctx.get('goal')}). Judge allocentric direction/distance on an "
            'untraversed link — not egocentric from the camera.'
        )
    if construct == 'spatial_working_memory':
        if ctx.get('template_mode') == 'recall_count':
            return (
                'Ordered navigation frames. Count how many distinct instances of '
                f"{ctx.get('object_category') or ctx.get('object_type')} you saw "
                'along the stream up to now.'
            )
        return (
            'Ordered navigation: encode the object in the first frame, then recall '
            'its earlier bearing at query (object is gone from the final view).'
        )
    if construct == 'spatial_updating':
        return (
            'Ordered navigation: encode early, track your own motion through '
            'intermediate frames, report the updated bearing at the final pose.'
        )
    if construct == 'invisible_displacement':
        return (
            'Encode while visible, then the object moves while hidden; query when '
            'still absent — use the destination cue in the question.'
        )
    if construct == 'egocentric_encoding':
        return 'Single current view: report the object’s bearing in the camera frame.'
    if construct == 'allocentric_encoding':
        return 'Single view: object–object relation in a trusted object-centered frame.'
    return 'Raw images shown to the model (same as evaluation input).'


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
            image_roles=infer_image_roles(fact) or None,
            context=_slide_context(fact) or None,
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
