"""[CODE] deterministic scorers. An LLM must never judge correctness."""

from __future__ import annotations

from typing import Any, Optional

from cm_benchmark.generation.nav_graph import (
    load_episode_nav_graph,
    score_route_action_sequence,
    score_survey_action_sequence,
    snap_trajectory_to_graph,
    traversed_edges_from_snapped,
    viewed_edges_from_trajectory,
)


def _item_extra(item: Optional[dict[str, Any]]) -> dict:
    rec = item or {}
    extra = rec.get('context') or rec.get('extra') or {}
    return extra if isinstance(extra, dict) else {}


def _nav_endpoints(item: Optional[dict[str, Any]]) -> tuple[Any, Any, float, Any, Any]:
    extra = _item_extra(item)
    rec = item or {}
    source_node = extra.get('source_node') or rec.get('source_node')
    goal_node = extra.get('goal_node') or rec.get('goal_node')
    heading = extra.get('start_heading_deg', rec.get('start_heading_deg', 0.0))
    start_pos = extra.get('start_pos') or rec.get('start_pos')
    goal_pos = extra.get('goal_pos') or rec.get('goal_pos')
    return source_node, goal_node, float(heading or 0.0), start_pos, goal_pos


def _item_edges(item: Optional[dict[str, Any]], key: str) -> list:
    extra = _item_extra(item)
    rec = item or {}
    edges = extra.get(key)
    if edges is None:
        edges = rec.get(key) or []
    return edges


def _graph_for_item(item, graph, episode):
    """Use the caller graph, else load episode_start nodes and edges from the SPOC file."""
    if graph is not None:
        return graph
    if episode is not None:
        return load_episode_nav_graph(episode)
    path = _item_extra(item).get('nav_graph_path')
    if not path:
        return None
    return load_episode_nav_graph({'nav_graph_path': path})


def _rebuild_edges(item, graph, episode, key: str) -> list:
    """Traversed / viewed edges from the episode trajectory and the SPOC graph.

    Slim items do not store those dumps. ``view_radius_m`` on the item keeps
    the survey radius that generation calibrated.
    """
    if episode is None or graph is None:
        return []
    snapped = snap_trajectory_to_graph(graph, episode.get('agent_trajectory') or [])
    traversed = traversed_edges_from_snapped(snapped)
    if key == 'traversed_edges':
        return [list(edge) for edge in traversed]
    radius = _item_extra(item).get('view_radius_m')
    if radius is None:
        return []
    viewed, _nodes = viewed_edges_from_trajectory(
        graph,
        episode.get('agent_trajectory') or [],
        float(radius),
        traversed_edges=traversed,
    )
    return [list(edge) for edge in viewed]


def _edges_for_score(item, graph, episode, key: str) -> list:
    stored = _item_edges(item, key)
    if stored:
        return stored
    return _rebuild_edges(item, graph, episode, key)


def _reference_actions(item: Optional[dict[str, Any]]):
    """Stored gold action sequence for nDTW, when present."""
    extra = _item_extra(item)
    rec = item or {}
    seq = extra.get('action_sequence')
    if seq is None:
        seq = rec.get('action_sequence')
    return seq


def score_route_knowledge(
    item: Optional[dict[str, Any]],
    graph,
    raw_actions,
    *,
    episode: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict:
    """Score a route_knowledge reply: success, SPL, nDTW/SDTW; validity diagnostic.

    When the item has no ``traversed_edges``, pass ``episode`` so the walked
    edges are rebuilt from the trajectory and the SPOC graph.
    """
    source_node, goal_node, heading, start_pos, goal_pos = _nav_endpoints(item)
    graph = _graph_for_item(item, graph, episode)
    kwargs.setdefault('reference_actions', _reference_actions(item))
    return score_route_action_sequence(
        graph,
        source_node,
        goal_node,
        raw_actions,
        _edges_for_score(item, graph, episode, 'traversed_edges'),
        start_heading_deg=heading,
        start_pos=start_pos,
        goal_pos=goal_pos,
        **kwargs,
    )


def score_survey_based_route_planning(
    item: Optional[dict[str, Any]],
    graph,
    raw_actions,
    *,
    episode: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict:
    """Score a survey reply: success (headline) and SPL; validity is diagnostic.

    When the item omits edge dumps, pass ``episode`` to rebuild them from the
    trajectory, ``view_radius_m``, and the SPOC graph.
    """
    source_node, goal_node, heading, start_pos, goal_pos = _nav_endpoints(item)
    graph = _graph_for_item(item, graph, episode)
    kwargs.setdefault('reference_actions', _reference_actions(item))
    return score_survey_action_sequence(
        graph,
        source_node,
        goal_node,
        raw_actions,
        _edges_for_score(item, graph, episode, 'viewed_edges'),
        _edges_for_score(item, graph, episode, 'traversed_edges'),
        start_heading_deg=heading,
        start_pos=start_pos,
        goal_pos=goal_pos,
        **kwargs,
    )


def score_mcq(item: dict[str, Any], choice: Optional[str]) -> dict:
    """Exact option-letter match, plus the failure mode of the chosen option.

    ``chosen_mode`` is the item's ``distractor_rationale`` tag for the chosen
    letter (``correct``, ``original_location``, ...), or None when the reply is
    not one of the option letters.
    """
    key = str(choice or '').strip().upper()
    options = item.get('options') or {}
    if key not in options:
        return {'correct': False, 'chosen_mode': None}
    return {
        'correct': key == item.get('answer'),
        'chosen_mode': (item.get('distractor_rationale') or {}).get(key),
    }


def distractor_choice_rates(items: list[dict[str, Any]], choices: dict[str, Optional[str]]) -> dict:
    """Per construct, how wrong answers split over distractor failure modes.

    For invisible displacement, ``original_location`` is the perseveration
    (A-not-B) error: the model answers with where the object was, not where it
    was moved. ``chance`` is the share one tag would get if a wrong answer were
    picked uniformly among the item's wrong options, for comparison.
    """
    out: dict[str, dict] = {}
    for item in items:
        iid = item.get('item_id')
        if iid not in choices or not item.get('options'):
            continue
        res = score_mcq(item, choices[iid])
        row = out.setdefault(
            item.get('construct'),
            {'n': 0, 'n_correct': 0, 'n_wrong': 0, 'modes': {}, 'chance': {}},
        )
        row['n'] += 1
        if res['correct']:
            row['n_correct'] += 1
            continue
        row['n_wrong'] += 1
        n_wrong_opts = max(1, len(item['options']) - 1)
        for mode in (item.get('distractor_rationale') or {}).values():
            if mode == 'correct':
                continue
            row['chance'][mode] = row['chance'].get(mode, 0.0) + 1.0 / n_wrong_opts
        mode = res['chosen_mode'] or 'invalid_reply'
        row['modes'][mode] = row['modes'].get(mode, 0) + 1
    for row in out.values():
        n = row['n_wrong']
        row['rates'] = {m: c / n for m, c in row['modes'].items()} if n else {}
        row['chance'] = {m: c / n for m, c in row['chance'].items()} if n else {}
    return out


ROUTE_MC_CONSTRUCTS = ('route_knowledge', 'survey_based_route_planning')


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials (95% by default)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _acc_row(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {'n': n, 'n_correct': k, 'accuracy': k / n if n else None, 'ci95': [lo, hi]}


def route_mc_metrics(items: list[dict[str, Any]], choices: dict[str, Optional[str]]) -> dict:
    """Class-4 multiple-choice metrics per construct, from option letters.

    ``choices`` maps ``item_id`` to the reply letter. Per construct:
    accuracy with a 95% Wilson interval, chance (1 / options), the share of
    replies that are not an option letter, accuracy by route length
    (``n_route_steps``), by question style and by turn naming
    (``fully_named``: every mid turn names a landmark, ``partly_named``: some
    turns are ``walk ahead``; higher accuracy on fully named routes means the
    model uses the landmarks), how wrong answers split over
    distractor tags against chance (the single changes ``mirrored_turn`` /
    ``swapped_landmarks`` and their combined option, each its own row), and
    ``consensus_shortcut``: the
    accuracy a text-only "pick the option closest to the others" rule gets on
    the same items (from ``context.consensus_heuristic_agrees``; about 0.25 in
    the 2x2 design). Free-action items (no options) are left out; see
    ``free_action_descriptives``.
    """
    picked = [
        it for it in items
        if it.get('construct') in ROUTE_MC_CONSTRUCTS
        and it.get('options')
        and it.get('item_id') in choices
    ]
    out: dict[str, dict] = {}
    for construct in ROUTE_MC_CONSTRUCTS:
        rows = [it for it in picked if it.get('construct') == construct]
        if not rows:
            continue
        results = [(it, score_mcq(it, choices[it['item_id']])) for it in rows]
        n = len(results)
        k = sum(1 for _it, r in results if r['correct'])
        invalid = sum(1 for _it, r in results if r['chosen_mode'] is None)
        chance = sum(1.0 / len(it['options']) for it in rows) / n
        by_steps: dict = {}
        by_style: dict = {}
        by_naming: dict = {}
        for it, r in results:
            ctx = _item_extra(it)
            steps = ctx.get('n_route_steps')
            named, total = ctx.get('named_turns'), ctx.get('total_turns')
            naming = (
                None if named is None or not total
                else 'fully_named' if named == total else 'partly_named'
            )
            for table, key in (
                (by_steps, steps),
                (by_style, it.get('question_style')),
                (by_naming, naming),
            ):
                cell = table.setdefault(key, [0, 0])
                cell[0] += int(r['correct'])
                cell[1] += 1
        shortcut = 0.0
        for it in rows:
            agrees = _item_extra(it).get('consensus_heuristic_agrees')
            shortcut += 1.0 / len(it['options']) if agrees is None else float(bool(agrees))
        out[construct] = {
            **_acc_row(k, n),
            'chance': chance,
            'invalid_rate': invalid / n,
            'by_n_route_steps': {s: _acc_row(c, t) for s, (c, t) in sorted(by_steps.items(), key=lambda kv: str(kv[0]))},
            'by_question_style': {s: _acc_row(c, t) for s, (c, t) in sorted(by_style.items(), key=lambda kv: str(kv[0]))},
            'by_naming': {s: _acc_row(c, t) for s, (c, t) in sorted(by_naming.items(), key=lambda kv: str(kv[0]))},
            'distractors': distractor_choice_rates(rows, choices).get(construct, {}),
            'consensus_shortcut': shortcut / n,
        }
    return out


ROTATION_CONSTRUCTS = ('invisible_displacement', 'spatial_updating')
ROTATION_BIN_EDGES_DEG = (180.0, 360.0)


def rotation_bin(deg: Optional[float], edges=ROTATION_BIN_EDGES_DEG) -> Optional[str]:
    """Group label for ``rotation_since_last_seen``: ``<180``, ``180-360``, ``>360``."""
    if deg is None:
        return None
    edges = [float(e) for e in edges]
    if float(deg) < edges[0]:
        return f'<{edges[0]:g}'
    for lo, hi in zip(edges, edges[1:]):
        if float(deg) <= hi:
            return f'{lo:g}-{hi:g}'
    return f'>{edges[-1]:g}'


def rotation_metrics(
    items: list[dict[str, Any]],
    choices: dict[str, Optional[str]],
    edges=ROTATION_BIN_EDGES_DEG,
) -> dict:
    """Accuracy by ``rotation_since_last_seen`` group, per construct.

    ``rotation_since_last_seen`` is the agent's total turning (left and right both
    add) between the last frame showing the cue the viewer must track and the
    query: the object for spatial updating, the named destination landmark (or
    swap partner) for invisible displacement. Accuracy that falls as rotation
    grows is a finding about updating load, not an item defect. Items without the
    field are left out.
    """
    out: dict[str, dict] = {}
    for construct in ROTATION_CONSTRUCTS:
        table: dict[str, list[int]] = {}
        for it in items:
            if it.get('construct') != construct or it.get('item_id') not in choices:
                continue
            group = rotation_bin(_item_extra(it).get('rotation_since_last_seen'), edges)
            if group is None:
                continue
            cell = table.setdefault(group, [0, 0])
            cell[0] += int(score_mcq(it, choices[it['item_id']])['correct'])
            cell[1] += 1
        if table:
            order = [rotation_bin(e - 1.0, edges) for e in edges] + [rotation_bin(edges[-1] + 1.0, edges)]
            out[construct] = {g: _acc_row(*table[g]) for g in order if g in table}
    return out


HEADING_ROTATION_BIN_EDGES_DEG = (360.0, 540.0)
FRAMES_SHOWN_BIN_EDGES = (8, 16, 24)


def frames_shown_bin(n: Any, edges=FRAMES_SHOWN_BIN_EDGES) -> Optional[str]:
    """Group label for ``n_frames_shown``: ``1-8``, ``9-16``, ``17-24``, ``25+``."""
    if n is None:
        return None
    lo = 1
    for hi in edges:
        if int(n) <= hi:
            return f'{lo}-{hi}'
        lo = hi + 1
    return f'{lo}+'


def review_groups(ctx: dict) -> dict[str, Optional[str]]:
    """Subgroups an item's results are split by (model scores and human review).

    ``template_mode`` (SWM heading change / count) and ``condition`` (EE
    single_frame / in_sequence) separate tasks that share a construct; heading
    items add ``n_turn_events`` and ``total_rotation_deg`` groups, and sequence
    items an ``n_frames_shown`` group.
    """
    heading = ctx.get('template_mode') == 'heading_change'
    seq = heading or ctx.get('condition') == 'in_sequence'
    return {
        'template_mode': ctx.get('template_mode'),
        'condition': ctx.get('condition'),
        'n_turn_events': turn_event_group(ctx.get('n_turn_events')) if heading else None,
        'total_rotation_deg': (
            rotation_bin(ctx.get('total_rotation_deg'), HEADING_ROTATION_BIN_EDGES_DEG) if heading else None
        ),
        'n_frames_shown': frames_shown_bin(ctx.get('n_frames_shown')) if seq else None,
    }


def turn_event_group(n: Any) -> Optional[str]:
    """``2`` / ``3`` / ``4+`` for a heading item's ``n_turn_events``."""
    if n is None:
        return None
    return '4+' if int(n) >= 4 else str(int(n))


def swm_metrics(
    items: list[dict[str, Any]],
    choices: dict[str, Optional[str]],
    rotation_edges=HEADING_ROTATION_BIN_EDGES_DEG,
) -> dict:
    """Spatial working memory accuracy, per template.

    ``heading_change`` (the main template): accuracy overall, by turn-event group
    (``n_turn_events``: 2 / 3 / 4+) and by ``total_rotation_deg`` group.
    ``recall_count`` (secondary): accuracy next to ``majority_baseline``, the
    accuracy of always answering the most common gold count (about 0.8 for
    "2" on Round 2), so a model at that level has not shown it counts.
    """
    out: dict[str, dict] = {}
    for mode in ('heading_change', 'recall_count'):
        rows = [
            it for it in items
            if it.get('construct') == 'spatial_working_memory'
            and _item_extra(it).get('template_mode') == mode
            and it.get('item_id') in choices
        ]
        if not rows:
            continue
        correct = {it['item_id']: score_mcq(it, choices[it['item_id']])['correct'] for it in rows}
        k = sum(correct.values())
        entry: dict[str, Any] = _acc_row(k, len(rows))
        if mode == 'heading_change':
            by_events: dict[str, list[int]] = {}
            by_rot: dict[str, list[int]] = {}
            for it in rows:
                ctx = _item_extra(it)
                for table, key in (
                    (by_events, turn_event_group(ctx.get('n_turn_events'))),
                    (by_rot, rotation_bin(ctx.get('total_rotation_deg'), rotation_edges)),
                ):
                    if key is None:
                        continue
                    cell = table.setdefault(key, [0, 0])
                    cell[0] += int(correct[it['item_id']])
                    cell[1] += 1
            rot_order = [rotation_bin(e - 1.0, rotation_edges) for e in rotation_edges] + [
                rotation_bin(rotation_edges[-1] + 1.0, rotation_edges)
            ]
            entry['by_n_turn_events'] = {g: _acc_row(*by_events[g]) for g in ('2', '3', '4+') if g in by_events}
            entry['by_total_rotation_deg'] = {g: _acc_row(*by_rot[g]) for g in rot_order if g in by_rot}
        else:
            golds: dict[str, int] = {}
            for it in rows:
                gold = (it.get('options') or {}).get(it.get('answer'))
                golds[str(gold)] = golds.get(str(gold), 0) + 1
            top, top_n = max(golds.items(), key=lambda kv: (kv[1], kv[0]))
            entry['majority_answer'] = top
            entry['majority_baseline'] = top_n / len(rows)
        out[mode] = entry
    return out


def free_action_descriptives(
    items: list[dict[str, Any]], scores: dict[str, dict[str, Any]]
) -> dict:
    """Descriptive summary of the free-action diagnostic subset, per construct.

    ``scores`` maps ``item_id`` to a ``score_route_knowledge`` /
    ``score_survey_based_route_planning`` result. Only items with
    ``context.diagnostic_subset == 'free_action'`` count. The subset is small
    (about one per house), so this reports counts and means only, no
    intervals or tests.
    """
    out: dict[str, dict] = {}
    for it in items:
        if _item_extra(it).get('diagnostic_subset') != 'free_action':
            continue
        res = scores.get(it.get('item_id'))
        if res is None:
            continue
        row = out.setdefault(it.get('construct'), {'n': 0, 'n_success': 0, 'spl': [], 'sdtw': []})
        row['n'] += 1
        row['n_success'] += int(bool(res.get('success')))
        if res.get('spl') is not None:
            row['spl'].append(float(res['spl']))
        if res.get('sdtw') is not None:
            row['sdtw'].append(float(res['sdtw']))
    for row in out.values():
        spl, sdtw = row.pop('spl'), row.pop('sdtw')
        row['success_rate'] = row['n_success'] / row['n'] if row['n'] else None
        row['mean_spl'] = sum(spl) / len(spl) if spl else None
        row['mean_sdtw'] = sum(sdtw) / len(sdtw) if sdtw else None
    return out
