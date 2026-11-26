"""SWM = integration across frames: heading change and count with re-identification."""

import pytest

from cm_benchmark.generation.constructs import HEADING_OPTIONS
from cm_benchmark.generation.planner import (
    _count_options,
    _wrap180,
    heading_distractors,
    heading_label,
    heading_turn_events,
)


def _rows(spec):
    """``(step, yaw, x, z)`` rows from a list of ('R'|'L'|'M') actions, 45 deg / 0.5 m each."""
    rows = [(0, 0.0, 0.0, 0.0)]
    yaw, x = 0.0, 0.0
    for i, a in enumerate(spec, start=1):
        if a == 'R':
            yaw += 45.0
        elif a == 'L':
            yaw -= 45.0
        else:
            x += 0.5
        rows.append((i, yaw % 360.0, x, 0.0))
    return rows


def test_wrap180_range():
    assert _wrap180(0) == 0
    assert _wrap180(180) == 180
    assert _wrap180(-180) == 180
    assert _wrap180(270) == -90
    assert _wrap180(-270) == 90
    assert _wrap180(405) == 45


@pytest.mark.near_cardinal
def test_heading_label_cones_reject_off_axis():
    # near_cardinal_max_deg is 20 in the default config.
    assert heading_label(0) == 'same'
    assert heading_label(90) == 'right'
    assert heading_label(-90) == 'left'
    assert heading_label(180) == 'around'
    assert heading_label(-180) == 'around'
    assert heading_label(45) is None
    assert heading_label(-135) is None
    assert heading_label(19) == 'same'
    assert heading_label(21) is None


def test_turn_events_merge_and_need_55_deg():
    # Two 45 deg steps back to back = one 90 deg event; a lone 45 is a correction.
    net, total, events = heading_turn_events(_rows('RR' + 'MMM' + 'L' + 'MMM'))
    assert net == 45.0 and total == 135.0
    assert [e['net_deg'] for e in events] == [90.0]
    # Turns closer than turn_merge_m (1.0 m) of travel merge: R, 0.5 m, R = one event.
    _n, _t, events = heading_turn_events(_rows('R' + 'M' + 'R' + 'MMM'))
    assert [e['net_deg'] for e in events] == [90.0]
    # A glance (out and back) merges to net 0: not an event, still in the total.
    net, total, events = heading_turn_events(_rows('RL' + 'MMM'))
    assert events == [] and net == 0.0 and total == 90.0


def test_two_turn_rule_counts_separate_events():
    _n, _t, one = heading_turn_events(_rows('MM' + 'RR' + 'MMM'))
    _n, _t, two = heading_turn_events(_rows('RR' + 'MMM' + 'RR' + 'MMM'))
    assert len(one) == 1 and len(two) == 2


@pytest.mark.near_cardinal
def test_heading_distractor_tags():
    # right, walk, right, walk, left: net right; before the last event: around;
    # mirrored: left; last event alone: left (already used) -> remaining label.
    rows = _rows('RR' + 'MMM' + 'RR' + 'MMM' + 'LL' + 'MMM')
    net, _t, events = heading_turn_events(rows)
    assert heading_label(net) == 'right' and len(events) == 3
    tags = heading_distractors('right', rows, events)
    assert tags == [('partial_sum', 'around'), ('mirrored', 'left'), ('remaining_label', 'same')]
    # Around has no mirror; the last turn alone is "right".
    rows = _rows('RR' + 'MMM' + 'RR' + 'MMM')
    tags = heading_distractors('around', rows, heading_turn_events(rows)[2])
    assert ('partial_sum', 'right') in tags
    assert [t for t, _l in tags].count('mirrored') == 0
    assert sorted(lab for _t, lab in tags) == ['left', 'right', 'same']
    # Last turn alone differs from the partial sum: left, walk, right -> same.
    rows = _rows('LL' + 'MMM' + 'RR' + 'MMM')
    tags = dict((lab, t) for t, lab in heading_distractors('same', rows, heading_turn_events(rows)[2]))
    assert tags == {'left': 'partial_sum', 'right': 'last_turn_only', 'around': 'remaining_label'}


def test_count_options_double_counted_tag():
    pool, seeds = _count_options(2, 3)
    assert pool == ['2', '3', '1', '4']
    # n + 1 = 3 is already the double count, so the last slot is n + 2.
    assert seeds == ['double_counted::3', 'off_by_one_count::1', 'off_by_two_count::4']
    # double_counted equal to the gold answer is not offered.
    pool, seeds = _count_options(3, 3)
    assert pool == ['3', '2', '4', '5']
    assert all(not s.startswith('double_counted') for s in seeds)
    assert seeds[-1] == 'off_by_two_count::5'


def _heading_episode(spec='RR' + 'MMM' + 'RR' + 'MMM' + 'LL' + 'MMM'):
    """Walk from ``spec``; default: right 90, 1.5 m, right 90, 1.5 m, left 90, 1.5 m."""
    rows = _rows(spec)
    steps, traj, acts = [], [], []
    prev = None
    for si, yaw, x, z in rows:
        pos, rot = [x, 0.9, z], [0.0, yaw, 0.0]
        steps.append({
            'step': si,
            'image_path': f'/img_{si}.png',
            'agent': {'position': pos, 'rotation': rot},
            'visible_objects': {},
            'non_visible_objects': {},
            'edges_egocentric': [],
        })
        traj.append({'step': si, 'position': pos, 'rotation': rot, 'image_path': f'/img_{si}.png'})
        if prev is not None:
            a = 'RotateRight' if yaw != prev[1] and _wrap180(yaw - prev[1]) > 0 else (
                'RotateLeft' if yaw != prev[1] else 'MoveAhead')
            acts.append({'step': si, 'action': a, 'degrees': 45 if yaw != prev[1] else None})
        prev = (si, yaw)
    return {
        'scene_id': 'house_test',
        'steps': steps,
        'agent_trajectory': traj,
        'agent_actions': acts,
        'displacement_events': [],
        'object_catalog': {},
    }


@pytest.mark.near_cardinal
@pytest.mark.label_margin
@pytest.mark.frame_fraction_floor
@pytest.mark.anchor_sector
@pytest.mark.label_change
def test_heading_change_items_with_default_config():
    from cm_benchmark.generation.pipeline import generate_items_for_episode

    items = generate_items_for_episode(
        _heading_episode(), constructs=['spatial_working_memory'], max_per_construct=20
    )
    heading = [
        it for it in items
        if it.get('status') == 'ok' and (it.get('context') or {}).get('template_mode') == 'heading_change'
    ]
    assert heading, 'expected heading-change items'
    styles = {it['question_style'] for it in heading}
    assert styles == {'concise', 'verbose'}
    allowed = set(HEADING_OPTIONS.values())
    for it in heading:
        ctx = it['context']
        assert ctx['n_turn_events'] >= 2
        assert ctx['n_frames_shown'] == len(it['image_paths'])
        assert ctx['total_rotation_deg'] >= 180.0
        assert set(it['options'].values()) == allowed
        tags = set(it['distractor_rationale'].values()) - {'correct'}
        assert tags <= {'partial_sum', 'mirrored', 'last_turn_only', 'remaining_label'}
        q = it['question']
        assert 'Compared with frame' in q and 'which way are you facing' in q
        # Invariant 4: the question never states a move or a direction.
        for word in ('left', 'right', 'around', 'turn', 'rotate'):
            assert word not in q.lower()
    # Every gold answer is the net yaw change between the two frames.
    rows = _rows('RR' + 'MMM' + 'RR' + 'MMM' + 'LL' + 'MMM')
    for it in heading:
        span = [r for r in rows if it['encoding_step'] <= r[0] <= it['query_step']]
        net = heading_turn_events(span)[0]
        assert it['options'][it['answer']] == HEADING_OPTIONS[heading_label(net)]
    golds = {it['options'][it['answer']] for it in heading}
    assert len(golds) >= 2, golds


def _heading_facts(episode, max_items=20):
    from cm_benchmark.generation.planner import _plan_heading_change

    return _plan_heading_change(episode, max_items)


@pytest.mark.near_cardinal
def test_heading_rejects_shared_view_and_stores_displacement():
    ep = _heading_episode()
    facts = _heading_facts(ep)
    assert facts
    for f in facts:
        assert f.extra['displacement_m'] is not None and f.extra['displacement_m'] >= 0
    # One object clearly seen in every frame: frames i and n always share it.
    for st in ep['steps']:
        st['visible_objects'] = {'Sofa|1': {}}
    assert _heading_facts(ep) == []
    # Seen only in the first and last frames: spans from step 0 to the end are dropped.
    last = ep['steps'][-1]['step']
    for st in ep['steps']:
        st['visible_objects'] = {'Sofa|1': {}} if st['step'] in (0, last) else {}
    assert all(not (f.encoding_step == 0 and f.query_step == last) for f in _heading_facts(ep))


@pytest.mark.near_cardinal
def test_heading_picks_spread_over_event_groups_and_house_cap():
    from cm_benchmark.generation.config import get_generation_config, use_generation_config
    from dataclasses import replace

    # Four right turns then a left: spans with 2, 3 and 4 events all get picked.
    facts = _heading_facts(_heading_episode('RRMMM' * 4 + 'LLMMM'))
    assert {f.extra['n_turn_events'] for f in facts} == {2, 3, 4}
    cfg = get_generation_config()
    capped = replace(cfg, spatial_working_memory=replace(cfg.spatial_working_memory, heading_max_per_house=1))
    with use_generation_config(capped):
        assert len(_heading_facts(_heading_episode())) == 1


def test_swm_metrics_groups_and_count_baseline():
    from cm_benchmark.evaluation.score import swm_metrics

    def heading(iid, n_events, total, ok):
        return {
            'item_id': iid, 'construct': 'spatial_working_memory', 'answer': 'A',
            'options': {'A': 'x', 'B': 'y'},
            'context': {'template_mode': 'heading_change', 'n_turn_events': n_events, 'total_rotation_deg': total},
        }, ('A' if ok else 'B')

    def count(iid, gold):
        return {
            'item_id': iid, 'construct': 'spatial_working_memory', 'answer': 'A',
            'options': {'A': gold, 'B': '9'}, 'context': {'template_mode': 'recall_count'},
        }, 'A'

    pairs = [heading('h1', 2, 270, True), heading('h2', 3, 450, False), heading('h3', 5, 720, True),
             count('c1', '2'), count('c2', '2'), count('c3', '3'), count('c4', '2')]
    items = [p[0] for p in pairs]
    choices = {p[0]['item_id']: p[1] for p in pairs}
    out = swm_metrics(items, choices)
    h = out['heading_change']
    assert h['n'] == 3 and h['n_correct'] == 2
    assert {g: r['n_correct'] for g, r in h['by_n_turn_events'].items()} == {'2': 1, '3': 0, '4+': 1}
    assert list(h['by_total_rotation_deg']) == ['<360', '360-540', '>540']
    c = out['recall_count']
    assert c['majority_answer'] == '2' and c['majority_baseline'] == 0.75 and c['accuracy'] == 1.0
