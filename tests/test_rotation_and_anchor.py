"""rotation_since_last_seen (total turning) and the ID landmark-sector rule."""

from cm_benchmark.generation.planner import _landmark_ego_label, _total_turn_deg


def _episode(yaws):
    return {
        'steps': [
            {'step': i, 'agent': {'position': [0.0, 0.9, 0.0], 'rotation': [0.0, y, 0.0]}}
            for i, y in enumerate(yaws)
        ],
        'object_catalog': {
            # Agent faces +z at yaw 0; a box 3 m ahead and a box 3 m behind.
            'Ahead|1|1': {'aabb_min': [-0.2, 0.0, 2.8], 'aabb_max': [0.2, 1.0, 3.2]},
            'Behind|1|2': {'aabb_min': [-0.2, 0.0, -3.2], 'aabb_max': [0.2, 1.0, -2.8]},
        },
    }


def test_total_turn_adds_left_and_right_turns():
    # 45 right, 45 back, 45 left, wrap 315 -> 0 (45): 180 total, though net is 0.
    ep = _episode([0.0, 45.0, 0.0, 315.0, 0.0])
    assert _total_turn_deg(ep, 0, 4) == 180.0
    assert _total_turn_deg(ep, 1, 2) == 45.0
    assert _total_turn_deg(ep, 2, 2) == 0.0


def test_landmark_label_uses_the_query_pose():
    ep = _episode([0.0, 180.0])
    assert _landmark_ego_label(ep, 'Ahead|1|1', 0) == 'ahead of you'
    assert _landmark_ego_label(ep, 'Behind|1|2', 0) == 'behind you'
    assert _landmark_ego_label(ep, 'Behind|1|2', 1) == 'ahead of you'
    assert _landmark_ego_label(ep, 'Floor', 0) is None


def test_rotation_metrics_groups_by_total_turning():
    from cm_benchmark.evaluation.score import rotation_bin, rotation_metrics

    assert [rotation_bin(d) for d in (0, 179.9, 180, 360, 405)] == [
        '<180', '<180', '180-360', '180-360', '>360'
    ]
    items = [
        {'item_id': f'i{n}', 'construct': 'spatial_updating', 'answer': 'A',
         'options': {'A': 'x', 'B': 'y'}, 'context': {'rotation_since_last_seen': r}}
        for n, r in enumerate((90, 90, 405))
    ]
    out = rotation_metrics(items, {'i0': 'A', 'i1': 'B', 'i2': 'A'})
    assert list(out['spatial_updating']) == ['<180', '>360']
    assert out['spatial_updating']['<180']['n_correct'] == 1
    assert out['spatial_updating']['>360']['accuracy'] == 1.0
