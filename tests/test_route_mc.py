"""Class-4 multiple choice: turn merging, labels, option text, distractors, metrics."""

import pytest

from cm_benchmark.evaluation.score import route_mc_metrics, wilson_interval
from cm_benchmark.generation.route_mc import (
    consensus_pick,
    format_route,
    point_box_gap_xz,
    route_distractors,
    turn_clusters,
    turn_label,
)

LABELS = (90.0, 135.0, 180.0)


def _poses(actions, step=0.2, rot=45.0):
    """Dead-reckoned xz poses (one per action plus the start), heading 0 = +z."""
    import math

    x = z = 0.0
    heading = 0.0
    poses = [(x, z)]
    for a in actions:
        if a == 'rotate_right':
            heading += rot
        elif a == 'rotate_left':
            heading -= rot
        else:
            x += step * math.sin(math.radians(heading))
            z += step * math.cos(math.radians(heading))
        poses.append((x, z))
    return poses


@pytest.mark.parametrize(
    'deg,label',
    [(0, 'straight'), (45, 'straight'), (-45, 'straight'), (90, 'right'), (-90, 'left'),
     (135, 'sharp_right'), (-135, 'sharp_left'), (180, 'around'), (-180, 'around'), (100, None)],
)
def test_turn_label_exact_45_degree_steps(deg, label):
    assert turn_label(deg, LABELS) == label


def test_sharp_turn_text_and_mirror_keeps_sharp():
    assert format_route('sharp_left', ['right'], ['Bed']) == (
        'Turn sharply left, walk to the Bed, turn right, walk to the end point'
    )
    correct, wrong = route_distractors('sharp_left', ['right'], ['Bed'], key='k')
    texts = [t for _tag, t in wrong]
    assert 'Turn sharply right, walk to the Bed, turn right, walk to the end point' in texts
    assert all(('sharply' in t) == ('sharply' in correct) for t in texts)


def test_turn_clusters_merge_close_turns_and_drop_trailing():
    m, r, l = 'move_ahead', 'rotate_right', 'rotate_left'
    # start turn left, 1.0 m, 45 right, 0.4 m, 45 right (merged: 90), 1.0 m, trailing turn
    acts = [l, l] + [m] * 5 + [r] + [m] * 2 + [r] + [m] * 5 + [l]
    start, mids = turn_clusters(acts, _poses(acts), rotation_deg=45.0, merge_m=1.0)
    assert start == -90.0
    assert [c['net_deg'] for c in mids] == [90.0]
    assert len(mids[0]['points']) == 4  # two runs, each with its before/after pose


def test_turn_clusters_keep_far_turns_apart():
    m, r = 'move_ahead', 'rotate_right'
    acts = [m] * 3 + [r, r] + [m] * 6 + [r, r] + [m] * 3
    _start, mids = turn_clusters(acts, _poses(acts), rotation_deg=45.0, merge_m=1.0)
    assert [c['net_deg'] for c in mids] == [90.0, 90.0]


def test_format_route_has_fixed_slots():
    assert format_route('straight', ['left'], ['Sofa']) == (
        'Walk to the Sofa, turn left, walk to the end point'
    )
    assert format_route('right', ['left', 'around'], ['Sofa', 'Bed']) == (
        'Turn right, walk to the Sofa, turn left, walk to the Bed, turn around, walk to the end point'
    )


def _parts_dist(a, b):
    return sum(x != y for x, y in zip(a.split(', '), b.split(', ')))


def test_distractors_2x2_mirror_and_swap_equidistant_and_seeded():
    names = ['Sofa', 'Bed', 'Desk']
    correct, wrong = route_distractors('straight', ['left', 'right', 'right'], names, key='k')
    tags = [tag for tag, _t in wrong]
    assert tags == ['mirrored_turn', 'swapped_landmarks', 'mirrored_turn+swapped_landmarks']
    texts = [correct, *(t for _tag, t in wrong)]
    assert len(set(texts)) == 4
    assert len({len(t.split(', ')) for t in texts}) == 1
    totals = [sum(_parts_dist(a, b) for b in texts) for a in texts]
    assert len(set(totals)) == 1  # nobody is the "average" option
    # X + Y is exactly X's change applied to Y
    x, y, xy = (t for _tag, t in wrong)
    assert _parts_dist(x, xy) == _parts_dist(correct, y)
    assert _parts_dist(y, xy) == _parts_dist(correct, x)
    assert route_distractors('straight', ['left', 'right', 'right'], names, key='k') == (correct, wrong)


def test_two_mirrors_when_no_swap_exists():
    correct, wrong = route_distractors('left', ['right'], ['Sofa'], key='k')
    assert [tag for tag, _t in wrong] == ['mirrored_turn', 'mirrored_turn', 'mirrored_turn+mirrored_turn']
    texts = [correct, *(t for _tag, t in wrong)]
    assert len(set(texts)) == 4


def test_no_option_set_without_two_independent_changes():
    assert route_distractors('straight', ['right'], ['Sofa'], key='k')[1] == []
    assert route_distractors('straight', ['around', 'around'], [None, None], key='k')[1] == []


def test_unnamed_turns_walk_ahead_and_swap_with_named():
    names = [None, 'Bed']
    assert format_route('straight', ['left', 'right'], names) == (
        'Walk ahead, turn left, walk to the Bed, turn right, walk to the end point'
    )
    _correct, wrong = route_distractors('straight', ['left', 'right'], names, key='k')
    swapped = dict(wrong)['swapped_landmarks']
    assert swapped == 'Walk to the Bed, turn left, walk ahead, turn right, walk to the end point'
    _c, wrong = route_distractors('straight', ['left', 'right'], [None, None], key='k')
    assert 'swapped_landmarks' not in dict(wrong)


def test_consensus_pick_breaks_ties_at_chance_and_box_gap():
    assert consensus_pick(['a, b, c', 'x, b, c', 'a, y, c', 'a, b, z'], 'k') == 0
    hits = 0
    for i in range(400):
        correct, wrong = route_distractors('straight', ['left', 'right'], [None, 'Bed'], key=f'k{i}')
        texts = [correct, *(t for _tag, t in wrong)]
        hits += consensus_pick(texts, f'k{i}') == 0
    assert 0.18 < hits / 400 < 0.32
    assert point_box_gap_xz((0.0, 0.0), (1.0, 0, 0.0), (2.0, 1, 1.0)) == pytest.approx(1.0)
    assert point_box_gap_xz((1.5, 0.5), (1.0, 0, 0.0), (2.0, 1, 1.0)) == 0.0


def _item(iid, answer, steps, style='concise', construct='route_knowledge', agrees=None):
    return {
        'item_id': iid,
        'construct': construct,
        'question_style': style,
        'options': {k: f'route {k}' for k in 'ABCD'},
        'answer': answer,
        'distractor_rationale': {
            k: ('correct' if k == answer else tag)
            for k, tag in zip('ABCD', ['mirrored_turn', 'swapped_landmarks', 'mirrored_turn+swapped_landmarks', 'x'])
        },
        'context': {
            'n_route_steps': steps,
            'consensus_heuristic_agrees': agrees,
            'named_turns': steps - 1 if style == 'concise' else 0,
            'total_turns': steps - 1,
        },
    }


def test_route_mc_metrics():
    items = [
        _item('1', 'A', 2, agrees=True),
        _item('2', 'A', 3, agrees=False),
        _item('3', 'D', 3, style='verbose'),
        _item('4', 'A', 2, construct='survey_based_route_planning'),
        {'item_id': '5', 'construct': 'egocentric_encoding', 'options': {'A': 'x'}, 'answer': 'A'},
    ]
    choices = {'1': 'A', '2': 'B', '3': 'zz', '4': 'C', '5': 'A'}
    m = route_mc_metrics(items, choices)
    rk = m['route_knowledge']
    assert (rk['n'], rk['n_correct']) == (3, 1)
    assert rk['chance'] == pytest.approx(0.25)
    assert rk['invalid_rate'] == pytest.approx(1 / 3)
    assert rk['by_n_route_steps'][2]['accuracy'] == 1.0
    assert rk['by_n_route_steps'][3]['accuracy'] == 0.0
    assert rk['distractors']['modes'] == {'swapped_landmarks': 1, 'invalid_reply': 1}
    assert rk['by_naming']['fully_named']['n'] == 2
    assert rk['by_naming']['partly_named']['accuracy'] == 0.0
    assert rk['consensus_shortcut'] == pytest.approx((1 + 0 + 0.25) / 3)
    assert m['survey_based_route_planning']['distractors']['modes'] == {
        'mirrored_turn+swapped_landmarks': 1
    }
    assert 'egocentric_encoding' not in m
    lo, hi = wilson_interval(1, 3)
    assert rk['ci95'] == [lo, hi] and lo < 1 / 3 < hi


def test_free_action_descriptives():
    from cm_benchmark.evaluation.score import free_action_descriptives

    items = [
        {'item_id': 'a', 'construct': 'route_knowledge', 'context': {'diagnostic_subset': 'free_action'}},
        {'item_id': 'b', 'construct': 'route_knowledge', 'context': {'diagnostic_subset': 'free_action'}},
        {'item_id': 'c', 'construct': 'survey_based_route_planning', 'context': {'diagnostic_subset': 'free_action'}},
        {'item_id': 'd', 'construct': 'route_knowledge', 'options': {'A': 'x'}},
    ]
    scores = {
        'a': {'success': True, 'spl': 0.5, 'sdtw': 0.4},
        'b': {'success': False, 'spl': 0.0, 'sdtw': 0.0},
        'c': {'success': True, 'spl': 1.0},
        'd': {'success': True, 'spl': 1.0},
    }
    out = free_action_descriptives(items, scores)
    assert out['route_knowledge'] == {
        'n': 2, 'n_success': 1, 'success_rate': 0.5, 'mean_spl': 0.25, 'mean_sdtw': 0.2,
    }
    assert out['survey_based_route_planning']['mean_sdtw'] is None
