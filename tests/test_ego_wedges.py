"""Ego labels with construct-specific ahead_half_width + distractor maps."""

from cm_benchmark.generation.config import load_generation_config, use_generation_config
from cm_benchmark.generation.constructs import (
    AHEAD_HALF_WIDTH_FOV,
    AHEAD_HALF_WIDTH_FULL,
    EGO_DIRECTION_OPTIONS,
    MIRRORED_LR,
    OPPOSITE,
    ORTHOGONAL,
    angle_to_ego_label,
    angular_span_crosses_sector,
    bbox_edges_cross_sector,
    imagined_perspective_label,
    local_offset_to_ego_label,
    pixel_column_to_bearing_deg,
    translated_egocentric_label,
)
from cm_benchmark.generation.geometry import (
    aabb_corners_cross_sector,
    ego_label_from_world_pose,
)
from cm_benchmark.generation.planner import _ego_label_at


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


def test_bbox_edges_cross_sector_wide_doorway():
    """house_000436 door|2|3: bbox spans most of a 396-wide frame.

    Legacy center±ang_width (bearing≈294.6°, width≈90°) missed the 340°
    boundary by ~0.4° and kept 'to your left'. Edge angles must refuse.
    """
    cfg = load_generation_config()
    with use_generation_config(cfg):
        bbox = [169.0, 0.0, 395.0, 223.0]
        frame_w = 396.0
        # Match SPOC nav camera: 396×224, fov_v=59 → fov_h ≈ 90°
        from cm_benchmark.utils.spatial_transformer import get_focal_length

        _fx, _fy, fov_h = get_focal_length(396, 224, 59.0)
        assert bbox_edges_cross_sector(
            bbox, frame_w, float(fov_h), ahead_half_width=AHEAD_HALF_WIDTH_FOV
        )
        # Legacy near-miss: half-width 45.005 vs 45.4° to the 340° edge.
        assert not angular_span_crosses_sector(
            294.6, 90.01, ahead_half_width=AHEAD_HALF_WIDTH_FOV
        )


def test_doorway_visible_bbox_refuses_ego_label():
    """Visible doorway with wide 2D bbox must not get a direction answer."""
    cfg = load_generation_config()
    episode = {
        'camera': {'width': 396, 'height': 224, 'fov_vertical_deg': 59},
        'episode_meta': {
            'camera': {'width': 396, 'height': 224, 'fov_v': 59},
        },
        'object_catalog': {
            'door|2|3': {
                'aabb_min': [-0.5, 0.0, 1.5],
                'aabb_max': [0.5, 2.0, 1.7],
            }
        },
        'steps': [
            {
                'step': 0,
                'agent': {
                    'position': [0.0, 0.9, 0.0],
                    'rotation': [0.0, 0.0, 0.0],
                },
                'visible_objects': {
                    'door|2|3': {
                        # Bearing ~295° (left) under FOV wedge; wide box straddles.
                        'bbox_center': [-0.9, 1.0, 0.4],
                        'position': [0.0, 1.0, 1.6],
                        'bbox': [169.0, 0.0, 395.0, 223.0],
                        'ang_width_deg': 90.01,
                    }
                },
                'edges_egocentric': [
                    {'source': 'agent', 'target': 'door|2|3', 'angle_relation': 'left'}
                ],
            }
        ],
    }
    with use_generation_config(cfg):
        # Without bbox gate the center±ang_width path still labels left.
        assert not angular_span_crosses_sector(
            294.6, 90.01, ahead_half_width=AHEAD_HALF_WIDTH_FOV
        )
        assert (
            _ego_label_at(
                episode,
                0,
                'door|2|3',
                ahead_half_width=AHEAD_HALF_WIDTH_FOV,
            )
            is None
        )


def test_hidden_aabb_corners_refuse_straddle():
    """Hidden object whose AABB spans left and ahead must be refused."""
    cfg = load_generation_config()
    with use_generation_config(cfg):
        # Agent at origin facing +Z; box from left (−x) into ahead (+z near axis).
        assert aabb_corners_cross_sector(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (-2.0, 0.0, 0.5),
            (-0.1, 1.0, 3.0),
            ahead_half_width=AHEAD_HALF_WIDTH_FULL,
        )
        # Compact box entirely in the right sector → keep.
        assert not aabb_corners_cross_sector(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 1.0),
            (2.2, 1.0, 1.2),
            ahead_half_width=AHEAD_HALF_WIDTH_FULL,
        )


def test_pixel_column_center_is_ahead():
    bearing = pixel_column_to_bearing_deg(198.0, 396.0, 90.0)
    assert bearing is not None
    assert abs(bearing) < 1e-6 or abs(bearing - 360.0) < 1e-6


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
