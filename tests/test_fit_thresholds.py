"""Tests for multi-scene load and DT robustness helpers in fit_thresholds."""

from pathlib import Path

import pandas as pd
import pytest

from cm_benchmark.utils.fit_thresholds import (
    discover_scene_pairs,
    load_data,
    load_data_from_folder,
    robustness_score,
)


def _write_scene(folder: Path, scene_id: str, n: int = 4):
    rows = []
    labels = {}
    for i in range(n):
        item_id = str(i)
        rows.append(
            {
                'item_id': item_id,
                'obj_id': f'Obj|{i}',
                'scene_id': scene_id,
                'obj-distance': 0.5 + 0.1 * i,
                'bbox-area': 100 + 10 * i,
                'min-side': 5 + i,
                'occupancy-ratio': 0.2 + 0.1 * i,
                'visible-pixels': 50 + 10 * i,
            }
        )
        labels[item_id] = {'obj_id': f'Obj|{i}', 'label': i % 2}
    man = folder / f'{scene_id}_calibration_manifest.csv'
    lab = folder / f'{scene_id}_labels.json'
    pd.DataFrame(rows).to_csv(man, index=False)
    lab.write_text(__import__('json').dumps(labels))
    return man, lab


def test_discover_and_load_folder(tmp_path):
    _write_scene(tmp_path, 'house_a', n=4)
    _write_scene(tmp_path, 'house_b', n=3)
    pairs = discover_scene_pairs(tmp_path)
    assert len(pairs) == 2

    df = load_data_from_folder(tmp_path)
    assert len(df) == 7
    assert set(df['scene_id']) == {'house_a', 'house_b'}
    assert 'item_id_global' in df.columns
    assert df['item_id_global'].is_unique


def test_load_data_single_pair(tmp_path):
    man, lab = _write_scene(tmp_path, 'house_x', n=2)
    df = load_data(man, lab)
    assert len(df) == 2
    assert (df['scene_id'] == 'house_x').all()


def test_robustness_score_penalizes_std():
    assert robustness_score(0.8, 0.0) > robustness_score(0.8, 0.2)
    assert robustness_score(0.9, 0.3) < robustness_score(0.85, 0.05)


def test_tune_tree_hyperparams_small_grid(tmp_path):
    pytest.importorskip('sklearn')
    from cm_benchmark.utils.fit_thresholds import tune_tree_hyperparams

    _write_scene(tmp_path, 'house_a', n=8)
    _write_scene(tmp_path, 'house_b', n=8)
    _write_scene(tmp_path, 'house_c', n=8)
    df = load_data_from_folder(tmp_path)
    df = df[df['y'] != 2].copy()

    grid = {
        'max_depth': [2, 3],
        'min_samples_leaf': [2, 3],
        'min_samples_split': [2],
        'criterion': ['gini'],
        'class_weight': ['balanced'],
    }
    results, best = tune_tree_hyperparams(
        df, param_grid=grid, primary_metric='f1', verbose=False
    )
    assert len(results) == 4
    assert 'robustness' in results.columns
    assert 'max_depth' in best
    assert results.iloc[0]['robustness'] >= results.iloc[-1]['robustness']


def test_plot_tune_results(tmp_path):
    pytest.importorskip('sklearn')
    pytest.importorskip('matplotlib')
    from cm_benchmark.utils.fit_thresholds import plot_tune_results, tune_tree_hyperparams

    _write_scene(tmp_path, 'house_a', n=8)
    _write_scene(tmp_path, 'house_b', n=8)
    _write_scene(tmp_path, 'house_c', n=8)
    df = load_data_from_folder(tmp_path)
    grid = {
        'max_depth': [2, 3],
        'min_samples_leaf': [2],
        'min_samples_split': [2],
        'criterion': ['gini'],
        'class_weight': ['balanced'],
    }
    results, best = tune_tree_hyperparams(
        df, param_grid=grid, primary_metric='f1', verbose=False
    )
    plots = tmp_path / 'plots'
    paths = plot_tune_results(results, plots, df=df, best_params=best)
    assert (plots / 'top_k_robustness.png').is_file()
    assert (plots / 'mean_vs_std_f1.png').is_file()
    assert (plots / 'best_loso_per_scene.png').is_file()
    assert len(paths) >= 3
