"""Simple tests for spatial relation helpers, including borderline thresholds."""

import pytest

from cm_benchmark.utils.spatial_relations import (
    get_x_direction,
    get_y_direction,
    get_z_direction,
    get_x_direction_angle,
    get_z_direction_angle,
    get_distance_text,
    get_direction_angle,
)

THR = 0.5
ANGLE_THR = 15
VERT_THR = 0.1


# --- axis directions (position thresholds) ---

def test_x_direction_clear_sides():
    assert get_x_direction(1.0, THR) == "right"
    assert get_x_direction(-1.0, THR) == "left"


def test_x_direction_borderline_at_threshold():
    # exactly at threshold → not yet "right"/"left" (strict >)
    assert get_x_direction(THR, THR) == ""
    assert get_x_direction(-THR, THR) == ""
    assert get_x_direction(THR + 0.01, THR) == "right"
    assert get_x_direction(-THR - 0.01, THR) == "left"


def test_y_direction_borderline_at_threshold():
    assert get_y_direction(VERT_THR, VERT_THR) == ""
    assert get_y_direction(-VERT_THR, VERT_THR) == ""
    assert get_y_direction(VERT_THR + 0.01, VERT_THR) == "above"
    assert get_y_direction(-VERT_THR - 0.01, VERT_THR) == "below"


def test_z_direction_borderline_at_threshold():
    assert get_z_direction(THR, THR) == ""
    assert get_z_direction(-THR, THR) == ""
    assert get_z_direction(THR + 0.01, THR) == "front"
    assert get_z_direction(-THR - 0.01, THR) == "behind"


# --- angle-based directions ---

def test_x_angle_straight_ahead_and_behind_are_ambiguous():
    assert get_x_direction_angle(0, ANGLE_THR) == ""
    assert get_x_direction_angle(180, ANGLE_THR) == ""
    assert get_x_direction_angle(-180, ANGLE_THR) == ""


def test_x_angle_borderline_near_threshold():
    # inside dead zone → no left/right
    assert get_x_direction_angle(ANGLE_THR - 0.1, ANGLE_THR) == ""
    # just outside dead zone → right
    assert get_x_direction_angle(ANGLE_THR + 0.1, ANGLE_THR) == "right"
    assert get_x_direction_angle(-(ANGLE_THR + 0.1), ANGLE_THR) == "left"


def test_z_angle_front_behind_and_side_band():
    assert get_z_direction_angle(0, ANGLE_THR) == "front"
    assert get_z_direction_angle(180, ANGLE_THR) == "behind"
    # near 90° ± threshold → ambiguous depth
    assert get_z_direction_angle(90, ANGLE_THR) == ""
    assert get_z_direction_angle(90 - ANGLE_THR - 0.1, ANGLE_THR) == "front"
    assert get_z_direction_angle(90 + ANGLE_THR + 0.1, ANGLE_THR) == "behind"


# --- distance labels ---

@pytest.mark.parametrize(
    "dist, expected",
    [
        (0.0, "within reach"),
        (0.5, "within reach"),       # inclusive upper bound of within_reach
        (0.5 + 1e-9, "nearby"),
        (1.0, "nearby"),
        (1.0 + 1e-9, "far"),
        (1.5, "far"),
        (1.5 + 1e-9, "beyond"),
        (10.0, "beyond"),
    ],
)
def test_distance_text_borders(dist, expected):
    assert get_distance_text(dist, 0.5, 1.0, 1.5) == expected


# --- combined direction from a 3D local offset ---

def test_direction_angle_object_ahead():
    # +Z is ahead in local frame
    assert get_direction_angle((0.0, 0.0, 1.0), ANGLE_THR, VERT_THR) == ("", "", "front")


def test_direction_angle_object_to_the_right():
    assert get_direction_angle((1.0, 0.0, 1.0), ANGLE_THR, VERT_THR) == ("right", "", "front")


def test_direction_angle_object_behind_left():
    assert get_direction_angle((-1.0, 0.0, -1.0), ANGLE_THR, VERT_THR) == ("left", "", "behind")


def test_direction_angle_object_slightly_above():
    x, y, z = get_direction_angle((0.0, VERT_THR + 0.05, 1.0), ANGLE_THR, VERT_THR)
    assert x == ""
    assert y == "above"
    assert z == "front"


def test_direction_angle_almost_on_axis_is_ambiguous_laterally():
    # small x relative to z → angle near 0 → no left/right
    x, y, z = get_direction_angle((0.01, 0.0, 1.0), ANGLE_THR, VERT_THR)
    assert x == ""
    assert z == "front"
