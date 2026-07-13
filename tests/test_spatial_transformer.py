import numpy as np
import pytest

from cm_benchmark.utils.spatial_transformer import (
    world_to_local,
    transform_3d_to_2d_with_fov,
    projection_with_local_vector,
)

ORIGIN = [0, 0, 0]
NO_ROTATION = [0, 0, 0]
SQRT2_OVER_2 = np.sqrt(2) / 2

HYPERPARAMS = {
    'w': 396,
    'h': 224,
    'fov_v': 59,
    'ez': 0.15,
}


def assert_local(camera_pos, agent_rot_deg, object_pos, expected, atol=1e-9):
    result = world_to_local(camera_pos, agent_rot_deg, object_pos)
    np.testing.assert_allclose(result, expected, atol=atol)


class TestWorldToLocalIdentity:
    """Zero rotation: local axes align with world axes."""

    @pytest.mark.parametrize(
        "object_pos, expected",
        [
            ([0, 0, 1], [0, 0, 1]),       # ahead (front)
            ([0, 0, -1], [0, 0, -1]),     # behind
            ([1, 0, 0], [1, 0, 0]),       # right
            ([-1, 0, 0], [-1, 0, 0]),   # left
            ([0, 1, 0], [0, 1, 0]),       # above
            ([0, -1, 0], [0, -1, 0]),     # below
        ],
        ids=["ahead", "behind", "right", "left", "above", "below"],
    )
    def test_cardinal_directions(self, object_pos, expected):
        assert_local(ORIGIN, NO_ROTATION, object_pos, expected)

    def test_object_at_camera_origin(self):
        assert_local(ORIGIN, NO_ROTATION, ORIGIN, [0, 0, 0])

    def test_camera_offset_preserves_relative_vector(self):
        assert_local([1, 2, 3], NO_ROTATION, [1, 2, 4], [0, 0, 1])


class TestWorldToLocalDistance:
    """Distance along forward axis is preserved in local Z."""

    @pytest.mark.parametrize(
        "distance",
        [1e-10, 1e-6, 1.0, 100.0, 1e6],
        ids=["near_zero", "tiny", "unit", "far", "very_far"],
    )
    def test_forward_distance_scales(self, distance):
        assert_local(ORIGIN, NO_ROTATION, [0, 0, distance], [0, 0, distance])

    def test_distance_invariant_under_rotation(self):
        obj = [3, 4, 5]
        local = world_to_local(ORIGIN, [30, 45, 0], obj)
        np.testing.assert_allclose(np.linalg.norm(local), np.linalg.norm(obj), atol=1e-9)


class TestWorldToLocalYaw:
    """Yaw rotates the horizontal plane; positive yaw turns camera toward +X."""

    @pytest.mark.parametrize(
        "yaw, world_obj, expected",
        [
            (0, [0, 0, 1], [0, 0, 1]),
            (90, [0, 0, 1], [-1, 0, 0]),    # world +Z appears left when facing +X
            (90, [1, 0, 0], [0, 0, 1]),    # world +X is ahead at yaw 90
            (180, [0, 0, 1], [0, 0, -1]),
            (180, [0, 0, -1], [0, 0, 1]),   # world behind becomes local ahead
            (270, [0, 0, 1], [1, 0, 0]),
            (270, [1, 0, 0], [0, 0, -1]),
            (-90, [0, 0, 1], [1, 0, 0]),    # equivalent to yaw 270
            (-90, [1, 0, 0], [0, 0, -1]),
            (360, [0, 0, 1], [0, 0, 1]),    # full rotation wraps to identity
        ],
        ids=[
            "yaw0_ahead", "yaw90_ahead", "yaw90_right",
            "yaw180_ahead", "yaw180_behind_becomes_ahead",
            "yaw270_ahead", "yaw270_right",
            "yaw_neg90_ahead", "yaw_neg90_right", "yaw360_ahead",
        ],
    )
    def test_yaw_rotations(self, yaw, world_obj, expected):
        assert_local(ORIGIN, [0, yaw, 0], world_obj, expected)


class TestWorldToLocalPitch:
    """Pitch tilts the camera horizon around the X axis."""

    @pytest.mark.parametrize(
        "pitch, world_obj, expected",
        [
            (-90, [0, 0, 1], [0, -1, 0]),   # looking up: ahead maps to below
            (-90, [0, 1, 0], [0, 0, 1]),    # looking up: above maps to ahead
            (0, [0, 0, 1], [0, 0, 1]),
            (0, [0, 1, 0], [0, 1, 0]),
            (45, [0, 0, 1], [0, SQRT2_OVER_2, SQRT2_OVER_2]),
            (45, [0, 1, 0], [0, SQRT2_OVER_2, -SQRT2_OVER_2]),
            (-45, [0, 0, 1], [0, -SQRT2_OVER_2, SQRT2_OVER_2]),
            (-45, [0, 1, 0], [0, SQRT2_OVER_2, SQRT2_OVER_2]),
            (90, [0, 0, 1], [0, 1, 0]),     # looking straight down
            (90, [0, 1, 0], [0, 0, -1]),    # above maps to behind when pitched 90
        ],
        ids=[
            "pitch_neg90_ahead", "pitch_neg90_above",
            "pitch0_ahead", "pitch0_above",
            "pitch45_ahead", "pitch45_above",
            "pitch_neg45_ahead", "pitch_neg45_above",
            "pitch90_ahead", "pitch90_above",
        ],
    )
    def test_pitch_rotations(self, pitch, world_obj, expected):
        assert_local(ORIGIN, [pitch, 0, 0], world_obj, expected)


class TestWorldToLocalCombinedRotation:
    def test_pitch_and_yaw_combined(self):
        # yaw 90 faces +X; pitch 45 tilts horizon — world +Z stays in horizontal plane
        assert_local(ORIGIN, [45, 90, 0], [0, 0, 1], [-1, 0, 0])


class TestWorldToLocalRollIgnored:
    """Roll is accepted but not applied (Rz is commented out in implementation)."""

    def test_roll_does_not_affect_ahead(self):
        no_roll = world_to_local(ORIGIN, [0, 0, 0], [0, 0, 1])
        with_roll = world_to_local(ORIGIN, [0, 0, 90], [0, 0, 1])
        np.testing.assert_allclose(no_roll, with_roll)

    def test_roll_does_not_affect_right(self):
        no_roll = world_to_local(ORIGIN, [0, 0, 0], [1, 0, 0])
        with_roll = world_to_local(ORIGIN, [0, 0, 90], [1, 0, 0])
        np.testing.assert_allclose(no_roll, with_roll)


class TestWorldToLocalInputTypes:
    def test_accepts_numpy_arrays(self):
        assert_local(
            np.array(ORIGIN),
            np.array(NO_ROTATION),
            np.array([0, 0, 1]),
            [0, 0, 1],
        )

    def test_returns_numpy_flat_array(self):
        result = world_to_local(ORIGIN, NO_ROTATION, [0, 0, 1])
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)


class TestProjectionBorderline:
    """3D local → 2D pixel projection edge cases."""

    def test_object_on_optical_axis_projects_to_image_center(self):
        local = (0.0, 0.0, 1.0)
        c_point = (198, 112)
        foc = (200.0, 200.0)
        u, v = projection_with_local_vector(local, c_point, foc, HYPERPARAMS)
        assert u == pytest.approx(198)
        assert v == pytest.approx(112)

    def test_object_too_close_in_depth_raises(self):
        # z at or below ez → invalid projection
        local = (0.0, 0.0, HYPERPARAMS['ez'])
        with pytest.raises(ValueError):
            projection_with_local_vector(local, (198, 112), (200.0, 200.0), HYPERPARAMS)

    def test_object_just_beyond_depth_threshold_projects(self):
        local = (0.0, 0.0, HYPERPARAMS['ez'] + 1e-6)
        u, v = projection_with_local_vector(local, (198, 112), (200.0, 200.0), HYPERPARAMS)
        assert u == pytest.approx(198)
        assert v == pytest.approx(112)


class TestTransform3dTo2dWithFov:
    def test_object_ahead_has_near_zero_angles(self):
        local, pixel, alpha, betha = transform_3d_to_2d_with_fov(
            ORIGIN, NO_ROTATION, [0, 0, 2], HYPERPARAMS
        )
        np.testing.assert_allclose(local, [0, 0, 2], atol=1e-9)
        assert alpha == pytest.approx(0.0, abs=1e-6)
        assert betha == pytest.approx(0.0, abs=1e-6)
        assert pixel[0] is not None and pixel[1] is not None

    def test_object_behind_camera_has_no_pixel(self):
        # z_local < ez → projection fails → (None, None)
        _local, pixel, alpha, _betha = transform_3d_to_2d_with_fov(
            ORIGIN, NO_ROTATION, [0, 0, -1], HYPERPARAMS
        )
        assert pixel == (None, None)
        assert abs(alpha) == pytest.approx(180.0, abs=1e-6)

    def test_object_to_the_right_has_positive_alpha(self):
        _local, _pixel, alpha, _betha = transform_3d_to_2d_with_fov(
            ORIGIN, NO_ROTATION, [1, 0, 1], HYPERPARAMS
        )
        assert alpha > 0
