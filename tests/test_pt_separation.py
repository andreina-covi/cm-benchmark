"""Perspective-taking: no combined-asset parts, no resting pairs, stable label."""

import pytest

from cm_benchmark.generation.geometry import near_cardinal, off_axis_deg
from cm_benchmark.generation.planner import (
    _asset_group,
    _pt_label_stable,
    _pt_objects_entangled,
)


def _box(cx, cz, hx, hz):
    return {'aabb_min': [cx - hx, 0.0, cz - hz], 'aabb_max': [cx + hx, 1.0, cz + hz]}


def test_asset_group_only_for_combined_ids():
    assert _asset_group('television|7|1|0') == 'television|7|1'
    assert _asset_group('television|7|1|0___1') == 'television|7|1'
    assert _asset_group('Chair|2|9') is None


def test_combined_asset_parts_are_entangled():
    episode = {'object_catalog': {}}
    assert _pt_objects_entangled(episode, ['television|7|1|0', 'Sofa|7|2', 'television|7|1|1'])
    assert not _pt_objects_entangled(episode, ['Chair|2|9', 'Chair|2|10', 'Sofa|7|2'])


def test_resting_object_is_entangled():
    episode = {'object_catalog': {'DiningTable|1|1': {'receptacle_object_ids': ['Bag|1|5']}}}
    assert _pt_objects_entangled(episode, ['DiningTable|1|1', 'Sofa|1|2', 'Bag|1|5'])
    assert _pt_objects_entangled(episode, ['Bag|1|5', 'Sofa|1|2', 'DiningTable|1|1'])


def _pos(x, z):
    return {'x': x, 'y': 0.0, 'z': z}


def test_label_flips_when_target_is_inside_standpoint_box():
    # A is a 2 m wide table; C sits on it, so C's side depends on where on A you stand.
    episode = {'object_catalog': {'A': _box(0, 0, 1.0, 0.5), 'C': _box(0.1, 0, 0.1, 0.1)}}
    assert not _pt_label_stable(episode, ['A', 'B', 'C'], _pos(0, 0), _pos(0, 5), _pos(0.1, 0), 'to your right')


def test_label_stable_for_separated_objects():
    episode = {'object_catalog': {'A': _box(0, 0, 0.3, 0.3), 'C': _box(-3, 1, 0.3, 0.3)}}
    # Facing +z from A, C at x = -3 is on one side for every sampled point.
    from cm_benchmark.generation.constructs import imagined_perspective_label

    label = imagined_perspective_label(_pos(0, 0), _pos(0, 5), _pos(-3, 1))
    assert label is not None
    assert _pt_label_stable(episode, ['A', 'B', 'C'], _pos(0, 0), _pos(0, 5), _pos(-3, 1), label)


def test_disparity_bins_are_45_degrees():
    from cm_benchmark.generation.planner import _pt_disparity_bin

    assert [_pt_disparity_bin(a) for a in (0, 44.9, 45, 90, 134, 135, 180)] == [0, 0, 1, 2, 2, 3, 3]
    assert _pt_disparity_bin(None) is None


def _fact(label, b, a_id='A', agrees=None):
    from cm_benchmark.generation.planner import PlannedFact

    return PlannedFact(
        construct='perspective_taking',
        status='ok',
        answer_label=label,
        extra={'disparity_bin': b, 'landmark_a_id': a_id, 'image_heuristic_agrees': agrees},
    )


def test_spread_takes_rare_label_and_bin_first_and_control_last():
    from cm_benchmark.generation.planner import _pt_spread

    facts = (
        [_fact('to your left', 1, f'L{i}') for i in range(5)]
        + [_fact('to your right', 0, f'R{i}') for i in range(2)]
        + [_fact('behind you', 3, 'B0')]
    )
    picked = _pt_spread(facts, 3)
    assert [f.answer_label for f in picked] == ['behind you', 'to your left', 'to your right']
    assert len(_pt_spread(facts, 100)) == len(facts)


def test_spread_caps_uses_of_one_object_as_a():
    from cm_benchmark.generation.planner import _pt_spread

    facts = [_fact('to your left', 1, 'Plant') for _ in range(4)] + [
        _fact('to your right', 2, 'Plant') for _ in range(3)
    ]
    for i, f in enumerate(facts):
        f.extra['landmark_b_id'] = f'B{i}'
    picked = _pt_spread(facts, 10)
    assert len(picked) == 2


def test_spread_balances_image_shortcut():
    from cm_benchmark.generation.planner import _pt_spread

    facts = [_fact('to your left', 1, f'X{i}', True) for i in range(4)] + [
        _fact('to your left', 1, 'Y', False)
    ]
    picked = _pt_spread(facts, 2)
    assert sorted(f.extra['image_heuristic_agrees'] for f in picked) == [False, True]


def test_label_flips_when_b_is_long():
    # A at origin, B a 6 m long counter across x; C is ~60° left of B's centre (left)
    # but only ~15° left of B's left end (ahead).
    episode = {
        'object_catalog': {
            'A': _box(0, 0, 0.1, 0.1),
            'B': _box(0, 3, 3.0, 0.2),
            'C': _box(-3.5, 2.0, 0.1, 0.1),
        }
    }
    from cm_benchmark.generation.constructs import imagined_perspective_label

    label = imagined_perspective_label(_pos(0, 0), _pos(0, 3), _pos(-3.5, 2.0))
    assert label is not None
    assert not _pt_label_stable(episode, ['A', 'B', 'C'], _pos(0, 0), _pos(0, 3), _pos(-3.5, 2.0), label)
    # A point-like B and a C square to the left (90 deg) give a stable answer.
    episode['object_catalog']['B'] = _box(0, 3, 0.05, 0.05)
    episode['object_catalog']['C'] = _box(-3.5, 0.0, 0.1, 0.1)
    left = imagined_perspective_label(_pos(0, 0), _pos(0, 3), _pos(-3.5, 0.0))
    assert _pt_label_stable(episode, ['A', 'B', 'C'], _pos(0, 0), _pos(0, 3), _pos(-3.5, 0.0), left)


def test_stand_point_inside_margin_rejects():
    # C 60 deg left at ~4 m: the centre is clear, but from the 0.5 m stand ring the
    # angle reaches the 45 deg edge's margin, so the triple is rejected.
    episode = {
        'object_catalog': {
            'A': _box(0, 0, 0.1, 0.1),
            'B': _box(0, 3, 0.05, 0.05),
            'C': _box(-3.5, 2.0, 0.1, 0.1),
        }
    }
    from cm_benchmark.generation.constructs import imagined_perspective_label

    label = imagined_perspective_label(_pos(0, 0), _pos(0, 3), _pos(-3.5, 2.0))
    assert label is not None
    assert not _pt_label_stable(episode, ['A', 'B', 'C'], _pos(0, 0), _pos(0, 3), _pos(-3.5, 2.0), label)


def test_image_heuristic_reads_marks():
    from cm_benchmark.generation.planner import _pt_image_heuristic

    episode = {
        'steps': [
            {
                'step': 0,
                'visible_objects': {
                    'A': {'mark_xy': [100, 50]},
                    'C': {'bbox': [200, 0, 300, 40]},
                },
            }
        ]
    }
    assert _pt_image_heuristic(episode, 0, 'A', 'C') == 'to your right'
    assert _pt_image_heuristic(episode, 0, 'C', 'A') == 'to your left'
    assert _pt_image_heuristic(episode, 0, 'A', 'Z') is None


def test_control_bin_capped_in_whole_set():
    from cm_benchmark.generation.generate_items import _trim_control_share

    def item(b):
        return {'context': {'disparity_bin': b}}

    chosen = [('s', item(0))] * 5 + [('s', item(2))] * 5
    _trim_control_share(chosen, 0.2)
    bins = [it['context']['disparity_bin'] for _n, it in chosen]
    assert bins.count(0) <= max(1, int(0.2 * len(bins)))
    assert bins.count(0) >= 1


def test_best_window_caps_frames_and_prefers_a_pair():
    from cm_benchmark.generation.planner import _pt_best_window

    ep = {'steps': [{'step': i, 'image_path': f'/i{i}.png'} for i in range(40)]}
    assert _pt_best_window(ep, [[1, 9], [9], [3, 9]], 8) == (9, 9, True)
    assert _pt_best_window(ep, [[1], [4], [6]], 8) == (1, 6, False)
    assert _pt_best_window(ep, [[1], [4], [30]], 8) is None
    # (10, 12) is shorter, but (20, 24) has B and C together on step 24.
    assert _pt_best_window(ep, [[10, 20], [11, 24], [12, 24]], 8) == (20, 24, True)
    assert _pt_best_window(ep, [[1], [], [6]], 8) is None
    # Shortest cover from step 1 ends at 3; extending to 5 adds the B+C frame.
    assert _pt_best_window(ep, [[1], [2, 5], [3, 5]], 8) == (1, 5, True)


def test_spread_takes_b_c_swap_once():
    from cm_benchmark.generation.planner import PlannedFact, _pt_spread

    def f(b, c, label):
        return PlannedFact(
            construct='perspective_taking',
            status='ok',
            answer_label=label,
            extra={'disparity_bin': 1, 'landmark_a_id': 'Sofa', 'landmark_b_id': b, 'landmark_c_id': c},
        )

    picked = _pt_spread([f('Trash', 'Wagon', 'ahead of you'), f('Wagon', 'Trash', 'to your left')], 5)
    assert len(picked) == 1


def test_a_standpoints_keep_only_the_side_facing_b():
    from cm_benchmark.generation.planner import _a_standpoints

    episode = {'object_catalog': {'A': _box(0, 0, 0.5, 0.5)}}
    pts = _a_standpoints(episode, 'A', _pos(0, 0), _pos(0, 5), 3, 0.5)
    assert pts[0] == _pos(0, 0)
    assert all(p['z'] > 0 for p in pts[1:])
    assert len(pts) == 4  # centre + three grid points on the +z edge


def test_frames_continuous_rejects_jumps():
    from cm_benchmark.generation.planner import _pt_frames_continuous

    def ep(poses):
        return {
            'steps': [
                {'step': i, 'image_path': f'/i{i}.png', 'agent': {'position': [x, 0.9, z], 'rotation': [0, h, 0]}}
                for i, (x, z, h) in enumerate(poses)
            ]
        }

    assert _pt_frames_continuous(ep([(0, 0, 0), (0, 0.2, 0), (0, 0.2, 45)]), [0, 1, 2])
    assert not _pt_frames_continuous(ep([(0, 0, 0), (0, 0, 90)]), [0, 1])
    assert not _pt_frames_continuous(ep([(0, 0, 0), (0, 1.0, 0)]), [0, 1])


@pytest.mark.near_cardinal
def test_near_cardinal_keeps_only_bearings_close_to_an_axis():
    # directions.near_cardinal_max_deg is 20: 19 deg off "ahead" passes, 23 deg does not.
    assert off_axis_deg(19.0) == 19.0 and off_axis_deg(-71.0) == 19.0
    assert near_cardinal(19.0) and near_cardinal(181.0) and near_cardinal(-90.0)
    assert not near_cardinal(23.0) and not near_cardinal(146.0)
    assert not near_cardinal(None)


def test_near_cardinal_off_when_null():
    # conftest sets near_cardinal_max_deg to None: every bearing passes.
    assert near_cardinal(35.0)
