"""Perspective-taking: no combined-asset parts, no resting pairs, stable label."""

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
