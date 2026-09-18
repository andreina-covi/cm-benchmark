"""Tests for Q&A visibility filters and episode_meta → hyperparams."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from cm_benchmark.generator.ai2thor_nav_generator import Ai2ThorNavGenerator
from cm_benchmark.generator.visibility_filters import (
    LABEL_AMBIGUOUS,
    LABEL_DISTINGUISHABLE,
    LABEL_NOT_DISTINGUISHABLE,
    classify_visibility_proba,
    load_visibility_filter_model,
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
    # Hard thresholds are on by default (no joblib required).
    assert hp['question_visibility']['min_side'] == 8.0
    assert hp['question_visibility']['min_bbox_area'] == 100.0
    assert hp['question_visibility']['min_visible_pixels'] == 40.0


def test_passes_filter_defaults_drop_tiny_blob():
    m = metrics_from_nav_row(
        {'cmin': 0, 'rmin': 0, 'cmax': 2, 'rmax': 2},
        frame_w=396,
        frame_h=224,
    )
    assert passes_question_visibility_filter(m, None) is False
    assert passes_question_visibility_filter(m, False) is True


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
            'visible-pixels': 20,
        }
    )
    # Explicit policy: only side + occupancy (other defaults cleared).
    only_side_occ = {
        'min_bbox_area': None,
        'min_side': 12,
        'min_occupancy_ratio': 0.3,
        'min_visible_pixels': None,
        'max_obj_distance': None,
    }
    assert passes_question_visibility_filter(tiny, only_side_occ) is False

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
    m = metrics_from_nav_row(
        {
            'cmin': 0,
            'rmin': 0,
            'cmax': 50,
            'rmax': 50,
            'bbox-area': 2500,
            'min-side': 50,
            'visible-pixels': 2000,
        }
    )
    assert m['occupancy_ratio'] is None
    only_occ = {
        'min_bbox_area': None,
        'min_side': None,
        'min_occupancy_ratio': 0.3,
        'min_visible_pixels': None,
        'max_obj_distance': None,
    }
    assert passes_question_visibility_filter(m, only_occ) is True


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
            'visible-pixels': 1400,
            'obj-distance': 5.0,
        }
    )
    only_dist = {
        'min_bbox_area': None,
        'min_side': None,
        'min_occupancy_ratio': None,
        'min_visible_pixels': None,
        'max_obj_distance': 2.0,
    }
    assert passes_question_visibility_filter(far, only_dist) is False
    only_dist_loose = dict(only_dist, max_obj_distance=6.0)
    assert passes_question_visibility_filter(far, only_dist_loose) is True


def test_alias_min_bbox_side_maps_to_min_side():
    m = metrics_from_nav_row(
        {
            'cmin': 0,
            'rmin': 0,
            'cmax': 5,
            'rmax': 5,
            'min-side': 5,
            'bbox-area': 25,
            'visible-pixels': 20,
        }
    )
    only_side = {
        'min_bbox_area': None,
        'min_side': None,  # will be overridden via alias after normalize... 
        'min_occupancy_ratio': None,
        'min_visible_pixels': None,
        'max_obj_distance': None,
        'min_bbox_side': 12,
    }
    assert passes_question_visibility_filter(m, only_side) is False


def test_legacy_all_null_thresholds_restore_defaults():
    from cm_benchmark.generator.visibility_filters import (
        normalize_question_visibility_thresholds,
    )

    thr = normalize_question_visibility_thresholds(
        {
            'min_bbox_area': None,
            'min_side': None,
            'min_occupancy_ratio': None,
            'min_visible_pixels': None,
            'max_obj_distance': None,
        }
    )
    assert thr['min_bbox_area'] == 100.0
    assert thr['min_side'] == 8.0


def test_apply_question_visibility_to_episode_drops_tiny():
    from cm_benchmark.generator.visibility_filters import (
        apply_question_visibility_to_episode,
    )

    ep = {
        'question_visibility': {
            'min_bbox_area': None,
            'min_side': None,
            'min_occupancy_ratio': None,
            'min_visible_pixels': None,
            'max_obj_distance': None,
        },
        'steps': [
            {
                'step': 0,
                'visible_objects': {
                    'Potato|1': {
                        'bbox_area': 9,
                        'min_side': 3,
                        'visible_pixels': 8,
                        'occupancy_ratio': 0.9,
                        'obj_distance': 2.0,
                    },
                    'Fridge|1': {
                        'bbox_area': 2000,
                        'min_side': 40,
                        'visible_pixels': 1800,
                        'occupancy_ratio': 0.9,
                        'obj_distance': 2.0,
                    },
                },
            }
        ],
    }
    out = apply_question_visibility_to_episode(ep, search_default_model=False)
    assert 'Potato|1' not in out['steps'][0]['visible_objects']
    assert 'Fridge|1' in out['steps'][0]['visible_objects']
    # Original episode untouched
    assert 'Potato|1' in ep['steps'][0]['visible_objects']


def test_apply_question_visibility_prefers_decision_tree(tmp_path):
    from cm_benchmark.generator.visibility_filters import (
        apply_question_visibility_to_episode,
        visibility_model_from_episode,
    )

    path = _toy_model_bundle(tmp_path)
    # Soft static floors would KEEP this (area 200 >= 100); tree rejects min_side < 15.
    ep = {
        'question_visibility': {
            'min_bbox_area': 100.0,
            'min_side': 8.0,
            'min_occupancy_ratio': None,
            'min_visible_pixels': 40.0,
            'max_obj_distance': None,
        },
        'steps': [
            {
                'step': 0,
                'visible_objects': {
                    'Mid|1': {
                        'bbox_area': 200.0,
                        'min_side': 10.0,
                        'visible_pixels': 180.0,
                        'occupancy_ratio': 0.5,
                        'obj_distance': 1.0,
                    },
                    'Big|1': {
                        'bbox_area': 1600.0,
                        'min_side': 40.0,
                        'visible_pixels': 1500.0,
                        'occupancy_ratio': 0.8,
                        'obj_distance': 1.5,
                    },
                },
            }
        ],
    }
    out = apply_question_visibility_to_episode(
        ep, model_path=path, search_default_model=False
    )
    assert visibility_model_from_episode(out) is not None
    assert 'Mid|1' not in out['steps'][0]['visible_objects']
    assert 'Big|1' in out['steps'][0]['visible_objects']
    assert out['visibility_filter_model']['path'] is not None


def test_fov_metrics_ok_uses_tree_not_static_floors(tmp_path):
    from cm_benchmark.generation.constructs import fov_metrics_ok

    model = load_visibility_filter_model(_toy_model_bundle(tmp_path))
    # Clears soft QUERY floors (1200/32/300) would fail; tree on min_side>=15 passes.
    mid = {
        'bbox_area': 400.0,
        'min_side': 20.0,
        'visible_pixels': 100.0,
        'occupancy_ratio': 0.4,
        'obj_distance': 1.0,
    }
    assert not fov_metrics_ok(mid)  # static QUERY fallback rejects
    assert fov_metrics_ok(mid, model=model)  # tree accepts


def test_plan_episode_attaches_model_for_distinguishability(tmp_path):
    from cm_benchmark.generation.planner import (
        _distinguishable_encoding_sighting,
        plan_episode,
    )
    from cm_benchmark.generator.visibility_filters import (
        EPISODE_VISIBILITY_MODEL_KEY,
        apply_question_visibility_to_episode,
    )

    path = _toy_model_bundle(tmp_path)
    # Object passes tree (side 40) but fails QUERY floors — with model attached,
    # distinguishability must follow the tree.
    episode = {
        'episode_meta': {'camera': {'width': 396, 'height': 224}},
        'steps': [
            {
                'step': 0,
                'image_path': '/img_0.png',
                'agent': {'position': [0, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {
                    'Chair|1': {
                        'category': 'Chair',
                        'position': [0.0, 0.9, 1.0],
                        'bbox': [50, 50, 90, 90],
                        'bbox_area': 400.0,
                        'min_side': 40.0,
                        'visible_pixels': 350.0,
                        'occupancy_ratio': 0.7,
                        'obj_distance': 1.2,
                    }
                },
                'edges_egocentric': [
                    {
                        'source': 'agent',
                        'target': 'Chair|1',
                        'angle_relation': ['ahead', '', ''],
                    }
                ],
            }
        ],
    }
    filtered = apply_question_visibility_to_episode(
        episode, model_path=str(path), search_default_model=False
    )
    assert EPISODE_VISIBILITY_MODEL_KEY in filtered
    assert _distinguishable_encoding_sighting(filtered, 0, 'Chair|1')
    facts = plan_episode(
        episode,
        constructs=['egocentric_encoding'],
        max_per_construct=2,
        visibility_model_path=str(path),
    )
    assert any(
        f.status == 'ok' and f.queried_object_id == 'Chair|1' for f in facts
    )


def test_classify_visibility_proba_bands():
    assert classify_visibility_proba(0.9, low=0.3, high=0.7) == LABEL_DISTINGUISHABLE
    assert classify_visibility_proba(0.1, low=0.3, high=0.7) == LABEL_NOT_DISTINGUISHABLE
    assert classify_visibility_proba(0.5, low=0.3, high=0.7) == LABEL_AMBIGUOUS


def _toy_model_bundle(tmp_path: Path) -> Path:
    """Tree on bbox-area + min-side (derivable from fixtures that only have bboxes)."""
    rng = np.random.default_rng(0)
    n = 40
    min_side = np.concatenate([rng.uniform(1, 8, n // 2), rng.uniform(20, 40, n // 2)])
    y = (min_side >= 15).astype(int)
    X = np.column_stack([min_side * min_side, min_side])
    features = ['bbox-area', 'min-side']
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)
    path = tmp_path / 'visibility_filter.joblib'
    joblib.dump(
        {
            'model': clf,
            'features': features,
            'low': 0.3,
            'high': 0.7,
            'ambiguous_proba_stats': {'n': 0, 'min': 0.3, 'max': 0.7, 'mean': 0.5},
        },
        path,
    )
    return path


def test_load_and_classify_with_model(tmp_path):
    path = _toy_model_bundle(tmp_path)
    model = load_visibility_filter_model(path)
    clear = metrics_from_nav_row(
        {
            'cmin': 0,
            'rmin': 0,
            'cmax': 40,
            'rmax': 40,
            'bbox-area': 1600,
            'min-side': 40,
        }
    )
    tiny = metrics_from_nav_row(
        {
            'cmin': 0,
            'rmin': 0,
            'cmax': 4,
            'rmax': 4,
            'bbox-area': 16,
            'min-side': 4,
        }
    )
    assert model.passes_for_questions(clear) is True
    assert model.passes_for_questions(tiny) is False
    assert model.classify_metrics(clear) == LABEL_DISTINGUISHABLE


def test_generator_uses_visibility_model(tmp_path):
    path = _toy_model_bundle(tmp_path)
    gen = Ai2ThorNavGenerator(
        path_navigation=str(NAV_CSV),
        path_objects=str(OBJ_CSV),
        output_path=str(tmp_path / 'out'),
        visibility_model_path=str(path),
    )
    assert gen.visibility_model is not None
    raw = gen._raw_fov_detections
    assert raw
    assert any(r.get('visibility_proba') is not None for r in raw)
    kept = sum(len(v['objects']) for v in gen.dict_navigation.values())
    assert kept <= len(raw)


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
