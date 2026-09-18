"""Perspective-taking signed-angle direction labels + ambiguity margins."""

from cm_benchmark.generation.constructs import (
    PT_DIRECTION_BOUNDARY_MARGIN_DEG,
    imagined_perspective_label,
    perspective_direction_label_from_signed,
    signed_rel_bearing_deg_xz,
)


def test_signed_rel_bearing_sign_convention_xz():
    """Facing +Z: +X is right (negative), −X is left (positive)."""
    right = signed_rel_bearing_deg_xz(0.0, 1.0, 1.0, 0.0)
    left = signed_rel_bearing_deg_xz(0.0, 1.0, -1.0, 0.0)
    ahead = signed_rel_bearing_deg_xz(0.0, 1.0, 0.0, 2.0)
    behind = signed_rel_bearing_deg_xz(0.0, 1.0, 0.0, -1.0)
    assert right is not None and abs(right - (-90.0)) < 1e-6
    assert left is not None and abs(left - 90.0) < 1e-6
    assert ahead is not None and abs(ahead) < 1e-6
    assert behind is not None and abs(abs(behind) - 180.0) < 1e-6


def test_perspective_bins_and_margin():
    assert perspective_direction_label_from_signed(-30.0) == 'to your right'
    assert perspective_direction_label_from_signed(30.0) == 'to your left'
    assert perspective_direction_label_from_signed(150.0) == 'behind you'
    assert perspective_direction_label_from_signed(-150.0) == 'behind you'
    # Near 0° / ±135° rejected
    assert perspective_direction_label_from_signed(5.0) is None
    assert perspective_direction_label_from_signed(-5.0) is None
    assert perspective_direction_label_from_signed(135.0) is None
    assert perspective_direction_label_from_signed(-135.0) is None
    assert perspective_direction_label_from_signed(130.0) is None  # within 10° of 135
    assert perspective_direction_label_from_signed(146.0) == 'behind you'
    assert perspective_direction_label_from_signed(
        5.0, margin_deg=0.0
    ) == 'to your left'


def test_countertop_fridge_window_regression_is_right():
    """House 007514 centroids: ~29° to the right of A→B → right, not ahead."""
    a = {'x': 0.49, 'y': 0.0, 'z': 1.64}
    b = {'x': 2.86, 'y': 0.01, 'z': 0.46}
    c = {'x': 1.51, 'y': 0.69, 'z': 0.17}
    signed = signed_rel_bearing_deg_xz(
        b['x'] - a['x'],
        b['z'] - a['z'],
        c['x'] - a['x'],
        c['z'] - a['z'],
    )
    assert signed is not None
    assert -135.0 + PT_DIRECTION_BOUNDARY_MARGIN_DEG < signed < -PT_DIRECTION_BOUNDARY_MARGIN_DEG
    assert imagined_perspective_label(a, b, c) == 'to your right'


def test_imagined_rejects_near_ahead():
    # Facing +Z, target almost ahead but slightly right → margin rejects
    assert (
        imagined_perspective_label(
            {'x': 0, 'z': 0},
            {'x': 0, 'z': 1},
            {'x': 0.05, 'z': 2},
        )
        is None
    )
    # Clear right
    assert (
        imagined_perspective_label(
            {'x': 0, 'z': 0},
            {'x': 0, 'z': 1},
            {'x': 1, 'z': 1},
        )
        == 'to your right'
    )
