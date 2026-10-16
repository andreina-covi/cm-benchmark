"""Build candidate items from PlannedFact (templates + verbose preamble)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cm_benchmark.generation.config import get_generation_config
from cm_benchmark.generation.disambiguation import draw_marked_frames
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
# Actions inside the item's encoding -> query window only. Class-4 is
# excluded: the model must produce the action sequence; attaching the
# collected list would leak the answer.
_ACTIONS_HOOKS = frozenset(
    {
        'spatial_working_memory',
        'spatial_updating',
    }
)
# Free-form collected-format actions; scored by graph reach, not MCQ.
_FREEFORM_ACTION_CONSTRUCTS = frozenset(
    {
        'route_knowledge',
        'survey_based_route_planning',
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

    labels = labels[: get_generation_config().options.n_options]
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
        if probe and probe != '_':
            return probe
    if obj_id:
        return object_type_from_id(obj_id)
    return 'object'


def _core_question(fact: PlannedFact) -> str:
    """Fill the construct's template with the fact's names, delay and cues."""
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
            f'Retrace the walked route from the {src} to the {goal}.'
        )
    elif fact.construct == 'survey_based_route_planning':
        src = extra.get('source', 'the start')
        goal = extra.get('goal', 'the goal')
        parts.append(
            f'Use the layout to plan a walk from the {src} to the {goal}. '
            'The connecting path was never walked; evidence comes from a '
            'doorway glimpse.'
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
            cap = get_generation_config().templates.verbose_max_nonvisible_names
            for oid, odata in list((st.get('non_visible_objects') or {}).items())[:cap]:
                if oid == queried:
                    continue
                typ = object_type_from_id(oid, {oid: odata})
                key = typ.lower()
                if key in nv_seen:
                    continue
                nv_seen.add(key)
                nv_cats.append(typ)
                if len(nv_cats) >= cap:
                    break
            if len(nv_cats) >= cap:
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
    """Question text for one style: concise, or preamble + concise."""
    core = _core_question(fact)
    if style == 'verbose':
        preamble = build_verbose_preamble(episode, fact)
        return f'{preamble} {core}'
    return core


def _frame_of_reference_for_fact(fact: PlannedFact) -> str:
    """The fact's frame of reference, else the construct default."""
    extra = fact.extra or {}
    if extra.get('frame_of_reference'):
        return str(extra['frame_of_reference'])
    return default_frame_of_reference(fact.construct)


def _step_of(entry) -> Optional[int]:
    """Integer ``step`` of a trajectory / action row, or None."""
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
    """Compact construct-specific fields for slides, review, and class-4 scoring.

    Source, goal, start pose, and the reference action sequence are stored on
    the item. Traversed / viewed edge dumps and path-node lists are not: the
    scorer reloads reachable positions and edges from the SPOC nav-graph file.
    """
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
        'template_index',
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
        'answer_format',
        'scoring',
        'graph_scope',
        'action_sequence',
        'start_heading_deg',
        'start_pos',
        'goal_pos',
        'source_landmark_id',
        'goal_landmark_id',
        'source_mark',
        'goal_mark',
        'source_node',
        'goal_node',
        'view_radius_m',
        'disambiguation',
        'disambiguation_landmark',
        'marks',
    )
    ctx = {k: extra[k] for k in keys if extra.get(k) is not None and extra.get(k) != ''}
    return ctx


def _item_context(episode: dict, fact: PlannedFact) -> Optional[dict]:
    """Stored item context. Class-4 keeps the reference, not edge dumps."""
    ctx = _slide_context(fact)
    if fact.construct in _FREEFORM_ACTION_CONSTRUCTS:
        seq = list(ctx.get('action_sequence') or [])
        ctx['reference_length'] = len(seq)
        path = episode.get('nav_graph_path')
        if path:
            ctx['nav_graph_path'] = str(path)
    return ctx or None


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
        if c == 'survey_based_route_planning':
            if n == 1:
                return [f'source/goal sighted · {src} → {goal}']
            roles = [f'source sighted · {src}']
            if n >= 2:
                roles.append(f'goal sighted · {goal}')
        else:
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


def class4_slide_panel(*args, **kwargs):
    """Re-export for tests / callers that already import templates."""
    from cm_benchmark.generation.slide_copy import class4_slide_panel as _panel

    return _panel(*args, **kwargs)


def _attach_schema_hooks(episode: dict, fact: PlannedFact, item: CandidateItem) -> None:
    """Copy per-frame poses, window actions and displacement events onto the item."""
    if fact.construct in _TRAJECTORY_HOOKS:
        item.agent_trajectory = _frame_poses(episode, fact) or None
    if fact.construct in _ACTIONS_HOOKS:
        item.agent_actions = _window_actions(episode, fact) or None
    if fact.construct == 'invisible_displacement' and fact.displacement_event:
        item.displacement_event = dict(fact.displacement_event)


def _stable_item_id(base_id: str, fact: PlannedFact, style: str) -> str:
    """Id without the fact index, so proximity settings share the same key."""
    extra = fact.extra or {}
    parts: list = [base_id, fact.construct]
    if extra.get('template_mode'):
        parts.append(extra['template_mode'])
    if fact.encoding_step is not None:
        parts.append(fact.encoding_step)
    if fact.query_step is not None:
        parts.append(fact.query_step)
    if fact.queried_object_id:
        parts.append(fact.queried_object_id)
    elif extra.get('object_category'):
        parts.append(extra['object_category'])
    for key in (
        'landmark_a_id',
        'landmark_b_id',
        'landmark_c_id',
        'source_landmark_id',
        'goal_landmark_id',
        'swap_partner_id',
        'floor_anchor_id',
        'destination_landmark_id',
    ):
        val = extra.get(key)
        if val:
            parts.append(val)
    parts.append(style)
    return '_'.join(str(p) for p in parts).replace('|', '_').replace(' ', '_')


def _image_paths_for_item(
    episode: dict,
    fact: PlannedFact,
    item_id: str,
    marked_root: Optional[Path],
) -> list[str]:
    """Original frames, or marked copies under ``marked_root/<item_id>/``."""
    paths = list(fact.image_paths or [])
    extra = fact.extra or {}
    marks = extra.get('marks') or {}
    if not marks or marked_root is None or not paths:
        return paths
    steps = [int(s) for s in (extra.get('shown_frame_steps') or [])]
    if len(steps) != len(paths):
        from cm_benchmark.generation.planner import _step_for_image_path

        steps = [
            si if (si := _step_for_image_path(episode, p)) is not None else 0
            for p in paths
        ]
    dest = Path(marked_root) / item_id
    return draw_marked_frames(episode, paths, steps, marks, dest)


def fact_to_items(
    episode: dict,
    fact: PlannedFact,
    *,
    base_id: str,
    styles: tuple[str, ...] = ('concise', 'verbose'),
    marked_root: Optional[Path] = None,
) -> list[CandidateItem]:
    """One item per question style for a fact (or one unsupported placeholder)."""
    scene = scene_id_of(episode)
    env = environment_of(episode)
    fo_r = _frame_of_reference_for_fact(fact)
    extra = fact.extra or {}

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

    if fact.construct in _FREEFORM_ACTION_CONSTRUCTS:
        options, rationale, answer_key = {}, {}, fact.answer_label
    else:
        options, rationale, answer_key = _shuffle_options(
            fact.answer_label, fact.options_pool or [], fact.distractor_seeds or []
        )

    items: list[CandidateItem] = []
    ids = [_stable_item_id(base_id, fact, style) for style in styles]

    for i, style in enumerate(styles):
        paired = ids[1 - i] if len(ids) == 2 else None
        image_paths = _image_paths_for_item(episode, fact, ids[i], marked_root)
        item = CandidateItem(
            item_id=ids[i],
            construct=fact.construct,
            class_=CONSTRUCT_CLASS[fact.construct],
            frame_of_reference=fo_r,
            environment=env,
            scene_id=scene,
            image_paths=image_paths,
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
            difficulty=extra.get('k'),
            image_roles=infer_image_roles(fact) or None,
            context=_item_context(episode, fact),
            disambiguation=extra.get('disambiguation') or 'none',
            disambiguation_landmark=extra.get('disambiguation_landmark'),
            marks=dict(extra.get('marks') or {}),
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
    marked_root: Optional[Path] = None,
) -> list[CandidateItem]:
    """Items for every fact. ``item_id`` is stable across proximity settings."""
    items: list[CandidateItem] = []
    for fact in facts:
        items.extend(
            fact_to_items(
                episode,
                fact,
                base_id=episode_tag,
                styles=styles,
                marked_root=marked_root,
            )
        )
    return items
