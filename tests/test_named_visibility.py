import pytest

from cm_benchmark.generation.config import get_generation_config, load_generation_config
from cm_benchmark.generation.constructs import (
    fov_metrics_ok,
    frame_fraction_ok,
    is_asset_subpart,
)

pytestmark = pytest.mark.frame_fraction_floor


def test_default_config_floor_is_one_percent():
    assert load_generation_config().visibility.named_min_frame_fraction == 0.01


def test_asset_subpart_ids():
    assert is_asset_subpart('television|7|0|0___0')
    assert not is_asset_subpart('television|7|0|0')
    assert not is_asset_subpart(None)


def test_frame_fraction_floor_from_config():
    frac = get_generation_config().visibility.named_min_frame_fraction
    wh = (396, 224)
    floor = frac * wh[0] * wh[1]
    assert frame_fraction_ok({'visible_pixels': floor}, wh)
    assert not frame_fraction_ok({'visible_pixels': floor - 1}, wh)
    # legacy rows without pixel counts are not rejected
    assert frame_fraction_ok({'bbox_area': 10}, wh)


class _AlwaysYes:
    def passes_for_questions(self, metrics):
        return True


def test_floor_applies_with_decision_tree():
    odata = {'visible_pixels': 50, 'bbox_area': 5000, 'min_side': 60}
    assert not fov_metrics_ok(odata, image_wh=(396, 224), model=_AlwaysYes())
    odata['visible_pixels'] = 2000
    assert fov_metrics_ok(odata, image_wh=(396, 224), model=_AlwaysYes())
