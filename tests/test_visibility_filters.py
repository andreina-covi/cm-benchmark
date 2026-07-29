"""Tests for Q&A visibility filters and episode_meta → hyperparams."""

from pathlib import Path

from cm_benchmark.generator.ai2thor_nav_generator import Ai2ThorNavGenerator
from cm_benchmark.generator.visibility_filters import (
    metrics_from_nav_row,
    passes_question_visibility_filter,
    resolve_hyperparams_from_episode_meta,
)

FIXTURES = Path(__file__).parent / 'fixtures'
NAV_CSV = FIXTURES / 'navigation_tiny.csv'
OBJ_CSV = FIXTURES / 'objects_tiny.csv'
EPISODE_DIR = FIXTURES / 'episode_tiny'


def test_resolve_hyperparams_from_episode_meta_camera_agent():
    meta = {
        'camera': {'width': 640, 'height': 480, 'fov_vertical_deg': 70},
        'agent': {'movement_constant': 0.25},
    }
    hp = resolve_hyperparams_from_episode_meta(meta)
    assert hp['w'] == 640
    assert hp['h'] == 480
    assert hp['fov_v'] == 70
    assert hp['mov_constant'] == 0.25
    assert hp['question_visibility']['min_side'] is None
    assert hp['question_visibility']['min_occupancy_ratio'] is None


def test_passes_filter_off_by_default():
    m = metrics_from_nav_row(
        {'cmin': 0, 'rmin': 0, 'cmax': 2, 'rmax': 2},
        frame_w=396,
        frame_h=224,
    )
    assert passes_question_visibility_filter(m, None) is True


def test_metrics_prefer_exported_columns():
    m = metrics_from_nav_row(
        {
            'cmin': 0,
            'rmin': 0,
            'cmax': 10,
            'rmax': 20,
            'bbox-area': 200,
            'min-side': 10,
            'occupancy-ratio': 0.55,
            'visible-pixels': 110,
            'obj-distance': 1.25,
        },
        frame_w=396,
        frame_h=224,
    )
    assert m['bbox_area'] == 200
    assert m['min_side'] == 10
    assert m['occupancy_ratio'] == 0.55
    assert m['visible_pixels'] == 110
    assert m['obj_distance'] == 1.25


def test_passes_filter_min_side_and_occupancy():
    tiny = metrics_from_nav_row(
        {
            'cmin': 0,
            'rmin': 0,
            'cmax': 5,
            'rmax': 5,
            'bbox-area': 25,
            'min-side': 5,
            'occupancy-ratio': 0.2,
        }
    )
    assert passes_question_visibility_filter(
        tiny, {'min_side': 12, 'min_occupancy_ratio': 0.3}
    ) is False

    clear = metrics_from_nav_row(
        {
            'cmin': 0,
            'rmin': 0,
            'cmax': 40,
            'rmax': 40,
            'bbox-area': 1600,
            'min-side': 40,
            'occupancy-ratio': 0.8,
            'visible-pixels': 1280,
            'obj-distance': 1.0,
        }
    )
    assert passes_question_visibility_filter(
        clear, {'min_side': 12, 'min_occupancy_ratio': 0.3, 'min_bbox_area': 100}
    ) is True


def test_missing_occupancy_does_not_reject_when_threshold_set():
    m = metrics_from_nav_row({'cmin': 0, 'rmin': 0, 'cmax': 50, 'rmax': 50})
    assert m['occupancy_ratio'] is None
    assert passes_question_visibility_filter(m, {'min_occupancy_ratio': 0.3}) is True


def test_max_obj_distance():
    far = metrics_from_nav_row(
        {
            'cmin': 0,
            'rmin': 0,
            'cmax': 40,
            'rmax': 40,
            'bbox-area': 1600,
            'min-side': 40,
            'occupancy-ratio': 0.9,
            'obj-distance': 5.0,
        }
    )
    assert passes_question_visibility_filter(far, {'max_obj_distance': 2.0}) is False
    assert passes_question_visibility_filter(far, {'max_obj_distance': 6.0}) is True


def test_alias_min_bbox_side_maps_to_min_side():
    m = metrics_from_nav_row(
        {'cmin': 0, 'rmin': 0, 'cmax': 5, 'rmax': 5, 'min-side': 5, 'bbox-area': 25}
    )
    assert passes_question_visibility_filter(m, {'min_bbox_side': 12}) is False


def test_generator_reads_meta_and_filters_fov(tmp_path):
    gen = Ai2ThorNavGenerator(
        csv_path_folder=str(EPISODE_DIR),
        output_path=str(tmp_path),
        question_visibility={'min_side': 1000},
    )
    assert gen.hyperparams['w'] == 396
    assert gen.hyperparams['h'] == 224
    assert gen.hyperparams['fov_v'] == 59
    assert gen.hyperparams['mov_constant'] == 0.2
    total_fov = sum(len(v['objects']) for v in gen.dict_navigation.values())
    assert total_fov == 0
    assert len(gen._raw_fov_detections) > 0


def test_detection_visibility_table_and_sweep(tmp_path):
    gen = Ai2ThorNavGenerator(
        path_navigation=str(NAV_CSV),
        path_objects=str(OBJ_CSV),
        output_path=str(tmp_path),
    )
    table = gen.detection_visibility_table()
    assert not table.empty
    assert 'bbox_area' in table.columns
    assert 'min_side' in table.columns

    sweep = gen.visibility_threshold_sweep(
        side_values=[1, 1000],
        bbox_area_values=[1, 1_000_000],
    )
    side_rows = sweep[sweep['criterion'] == 'min_side']
    assert float(side_rows.loc[side_rows['threshold'] == 1, 'keep_rate'].iloc[0]) == 1.0
    assert float(side_rows.loc[side_rows['threshold'] == 1000, 'keep_rate'].iloc[0]) == 0.0


def test_set_question_visibility_rebuilds_nav(tmp_path):
    gen = Ai2ThorNavGenerator(
        path_navigation=str(NAV_CSV),
        path_objects=str(OBJ_CSV),
        output_path=str(tmp_path),
    )
    before = sum(len(v['objects']) for v in gen.dict_navigation.values())
    assert before >= 1
    gen.set_question_visibility({'min_side': 10_000})
    after = sum(len(v['objects']) for v in gen.dict_navigation.values())
    assert after == 0
