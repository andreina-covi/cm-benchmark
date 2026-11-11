import pytest

from cm_benchmark.generation.config import get_generation_config, load_generation_config
from cm_benchmark.generation.constructs import bearings_cross_sector
from cm_benchmark.generation.planner import _ego_pool_with_diagnostics

pytestmark = pytest.mark.label_margin


def test_default_config_margin_and_visible_options():
    cfg = load_generation_config()
    assert cfg.directions.label_margin_deg == 10.0
    assert cfg.options.n_options_visible == 3


def test_bearing_near_edge_is_refused():
    m = get_generation_config().directions.label_margin_deg
    # ahead/right edge at 20° for objects in view
    assert bearings_cross_sector((20.0 + m - 1.0,), ahead_half_width=20.0)
    assert not bearings_cross_sector((20.0 + m + 1.0,), ahead_half_width=20.0)
    assert not bearings_cross_sector((0.0,), ahead_half_width=20.0)
    # full circle: 45° and 135° edges
    assert bearings_cross_sector((130.0,), ahead_half_width=45.0)
    assert not bearings_cross_sector((90.0,), ahead_half_width=45.0)


def test_extent_inside_sector_but_near_edge_is_refused():
    # both bbox edges are "to your right", but the left edge is 3° from the ahead edge
    assert bearings_cross_sector((23.0, 33.0), ahead_half_width=20.0)


def test_visible_pool_has_three_labels_and_no_behind():
    for correct in ('ahead of you', 'to your left', 'to your right'):
        pool, seeds = _ego_pool_with_diagnostics(correct, visible_only=True)
        assert len(pool) == 3
        assert 'behind you' not in pool
        assert correct in pool
    pool, _ = _ego_pool_with_diagnostics('ahead of you')
    assert len(pool) == 4


def test_visible_planners_skip_behind_label():
    import inspect

    from cm_benchmark.generation import planner

    for fn in (planner.plan_egocentric_encoding, planner.plan_spatial_working_memory):
        assert "if label == 'behind you':" in inspect.getsource(fn)
