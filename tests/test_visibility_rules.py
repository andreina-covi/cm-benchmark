"""Rules from the round-7 review: in-frame share, hidden = no mask pixel,
confusables from every detection, ID destination landmark gap, PT spacing."""

from cm_benchmark.generation.constructs import in_frame_ok
from cm_benchmark.generation.disambiguation import objects_with_mask_pixels
from cm_benchmark.generation.planner import (
    _id_in_raw_fov,
    _id_landmark_far_from_object,
    _pt_pairs_well_spaced,
)
from cm_benchmark.generation.step_derived import present_pixels
from cm_benchmark.generator.visibility_filters import metrics_from_nav_row


def test_in_frame_share_keeps_the_45_to_50_band():
    assert in_frame_ok({'in_frame_fraction': 0.48})
    assert in_frame_ok({'in_frame_fraction': 0.45})
    assert not in_frame_ok({'in_frame_fraction': 0.40})
    # Older collections have no value: nothing is rejected.
    assert in_frame_ok({'visible_pixels': 900})


def test_stage1_reads_in_frame_fraction():
    row = {'cmin': 0, 'rmin': 0, 'cmax': 10, 'rmax': 10, 'visible-pixels': 50, 'in-frame-fraction': 0.6}
    assert metrics_from_nav_row(row, frame_w=396, frame_h=224)['in_frame_fraction'] == 0.6


def _ep(present):
    return {'steps': [{'step': 0, 'visible_objects': {}, 'present_objects': present}]}


def test_filtered_detection_still_counts_as_in_view():
    ep = _ep({'Chair|3': 1790})
    assert present_pixels(ep['steps'][0], 'Chair|3') == 1790
    assert present_pixels(ep['steps'][0], 'Chair|4') == 0
    # Not distinguishable (absent from visible_objects) but on screen: not hidden.
    assert _id_in_raw_fov(ep, 'Chair|3', 0)
    assert not _id_in_raw_fov(ep, 'Chair|4', 0)


def test_dim_same_name_object_joins_the_pixel_pool():
    ep = _ep({'Bread|2|6': 717})
    assert 'Bread|2|6' in objects_with_mask_pixels(ep, [0])


def _box(x, z, half=0.25):
    return {'aabb_min': [x - half, 0.0, z - half], 'aabb_max': [x + half, 1.0, z + half]}


def test_destination_landmark_must_be_away_from_the_old_box():
    ep = {'object_catalog': {
        'Bread|1': _box(0.0, 0.0, 0.1),
        'CounterTop|1': _box(0.0, 0.0, 1.0),  # the bread sits on it
        'Sofa|1': _box(3.0, 0.0, 0.5),
    }}
    assert not _id_landmark_far_from_object(ep, 'CounterTop|1', 'Bread|1')
    assert _id_landmark_far_from_object(ep, 'Sofa|1', 'Bread|1')


def test_perspective_objects_must_be_half_a_metre_to_five_metres_apart():
    def obj(oid, x):
        return {'obj_id': oid, 'position': [x, 0.5, 0.0], **_box(x, 0.0)}

    assert _pt_pairs_well_spaced({}, [obj('A|1', 0.0), obj('B|1', 1.5), obj('C|1', 3.0)])
    # 0.35 m apart (box gap) is too close
    assert not _pt_pairs_well_spaced({}, [obj('A|1', 0.0), obj('B|1', 0.85), obj('C|1', 3.0)])
    # more than 5 m apart is too far
    assert not _pt_pairs_well_spaced({}, [obj('A|1', 0.0), obj('B|1', 2.0), obj('C|1', 7.0)])


def test_route_reference_keeps_only_the_net_turn():
    from cm_benchmark.generation.route_replay import collapse_rotations

    spin = ['rotate_right'] * 8
    assert collapse_rotations(['move_ahead', *spin, 'move_ahead'], 45) == ['move_ahead', 'move_ahead']
    assert collapse_rotations(['rotate_right', 'rotate_right', 'rotate_left'], 45) == ['rotate_right']
    assert collapse_rotations(['rotate_right'] * 6, 45) == ['rotate_left'] * 2
    assert collapse_rotations(['rotate_left'] * 4, 45) == ['rotate_left'] * 4
    assert collapse_rotations(['rotate_right', 'move_ahead', 'rotate_left'], 45) == ['rotate_right', 'move_ahead', 'rotate_left']
