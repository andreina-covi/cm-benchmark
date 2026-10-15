"""Ego labels with construct-specific ahead_half_width + distractor maps."""

from cm_benchmark.generation.constructs import (
    AHEAD_HALF_WIDTH_FOV,
    AHEAD_HALF_WIDTH_FULL,
    EGO_DIRECTION_OPTIONS,
    MIRRORED_LR,
    OPPOSITE,
    ORTHOGONAL,
    angle_to_ego_label,
    angular_span_crosses_sector,
    imagined_perspective_label,
    local_offset_to_ego_label,
    translated_egocentric_label,
)
from cm_benchmark.generation.geometry import ego_label_from_world_pose


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


def test_yaw_only_ignores_camera_pitch():
    """28° look-down must not change a horizontal label (nav camera mount)."""
    pos = (0.0, 1.5, 0.0)
    obj = (2.0, 0.5, 1.0)  # atan2(2, 1) ≈ 63° → right under a 45° wedge
    flat = ego_label_from_world_pose(pos, (0.0, 0.0, 0.0), obj, ahead_half_width=45)
    tilted = ego_label_from_world_pose(pos, (28.0, 0.0, 0.0), obj, ahead_half_width=45)
    assert flat == tilted
    assert flat == 'to your right'


def test_angular_width_rejects_sector_boundary():
    # 40° center, 20° wide → spans 30–50°, which crosses the 45° edge.
    assert angular_span_crosses_sector(40.0, 20.0, ahead_half_width=45)
    assert not angular_span_crosses_sector(20.0, 10.0, ahead_half_width=45)
    # Missing width does not invent a ±10° margin.
    assert not angular_span_crosses_sector(44.0, None, ahead_half_width=45)
    assert (
        ego_label_from_world_pose(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),  # atan2(1, 1) = 45°
            ahead_half_width=45,
            ang_width_deg=16.0,
        )
        is None
    )


def test_imagined_uses_signed_bins_not_ahead_wedge():
    # ~26.5° to the right of +Z facing — old 45° wedge said "ahead"; PT bins → right
    assert (
        imagined_perspective_label(
            {'x': 0, 'z': 0},
            {'x': 0, 'z': 1},
            {'x': 1, 'z': 2},
        )
        == 'to your right'
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
