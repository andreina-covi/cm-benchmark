"""Ego labels with construct-specific ahead_half_width + distractor maps."""

from cm_benchmark.generation.constructs import (
    AHEAD_HALF_WIDTH_FOV,
    AHEAD_HALF_WIDTH_FULL,
    EGO_DIRECTION_OPTIONS,
    MIRRORED_LR,
    OPPOSITE,
    ORTHOGONAL,
    angle_to_ego_label,
    imagined_perspective_label,
    local_offset_to_ego_label,
    translated_egocentric_label,
)


def test_angle_to_ego_label_full_circle_45():
    assert angle_to_ego_label(0, ahead_half_width=45) == 'ahead of you'
    assert angle_to_ego_label(44.9, ahead_half_width=45) == 'ahead of you'
    assert angle_to_ego_label(45, ahead_half_width=45) == 'to your right'
    assert angle_to_ego_label(90, ahead_half_width=45) == 'to your right'
    assert angle_to_ego_label(135, ahead_half_width=45) == 'behind you'
    assert angle_to_ego_label(180, ahead_half_width=45) == 'behind you'
    assert angle_to_ego_label(225, ahead_half_width=45) == 'to your left'
    assert angle_to_ego_label(270, ahead_half_width=45) == 'to your left'
    assert angle_to_ego_label(315, ahead_half_width=45) == 'ahead of you'


def test_angle_to_ego_label_fov_20_balances_sides():
    # ±25° is ahead under half_width=45 but right/left under FOV=20
    assert angle_to_ego_label(25, ahead_half_width=AHEAD_HALF_WIDTH_FOV) == 'to your right'
    assert angle_to_ego_label(335, ahead_half_width=AHEAD_HALF_WIDTH_FOV) == 'to your left'
    assert angle_to_ego_label(15, ahead_half_width=AHEAD_HALF_WIDTH_FOV) == 'ahead of you'
    assert angle_to_ego_label(90, ahead_half_width=AHEAD_HALF_WIDTH_FOV) == 'to your right'
    # Behind sector shrinks but still exists for full-circle uses of FOV width
    assert angle_to_ego_label(180, ahead_half_width=AHEAD_HALF_WIDTH_FOV) == 'behind you'


def test_local_offset_respects_width():
    # atan2(1, 2) ≈ 26.5° → ahead at 45, right at 20
    assert (
        local_offset_to_ego_label((1.0, 0.0, 2.0), ahead_half_width=AHEAD_HALF_WIDTH_FULL)
        == 'ahead of you'
    )
    assert (
        local_offset_to_ego_label((1.0, 0.0, 2.0), ahead_half_width=AHEAD_HALF_WIDTH_FOV)
        == 'to your right'
    )


def test_distractor_maps_cover_all_ego_labels():
    for lab in EGO_DIRECTION_OPTIONS:
        assert OPPOSITE[lab] in EGO_DIRECTION_OPTIONS
        assert ORTHOGONAL[lab] in EGO_DIRECTION_OPTIONS
        assert MIRRORED_LR[lab] in EGO_DIRECTION_OPTIONS


def test_imagined_uses_full_width_disambiguator_uses_fov():
    assert (
        imagined_perspective_label(
            {'x': 0, 'z': 0},
            {'x': 0, 'z': 1},
            {'x': 1, 'z': 2},  # ~26.5° from +Z facing
            ahead_half_width=AHEAD_HALF_WIDTH_FULL,
        )
        == 'ahead of you'
    )
    assert (
        translated_egocentric_label(
            0.0,
            {'x': 0, 'z': 0},
            {'x': 1, 'z': 2},
            ahead_half_width=AHEAD_HALF_WIDTH_FOV,
        )
        == 'to the right of'
    )
