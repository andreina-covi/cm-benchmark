"""Tests for visibility threshold analysis helpers."""

from pathlib import Path

from cm_benchmark.analysis.visibility_threshold_analysis import (
    load_detection_features,
    run_analysis,
    summarize_features,
)

FIXTURES = Path(__file__).parent / 'fixtures'
NAV_CSV = FIXTURES / 'navigation_tiny.csv'
OBJ_CSV = FIXTURES / 'objects_tiny.csv'
EPISODE_DIR = FIXTURES / 'episode_tiny'


def test_load_detection_features_from_tiny_nav():
    df = load_detection_features(NAV_CSV, objects_csv=OBJ_CSV)
    assert len(df) >= 1
    assert 'bbox_area' in df.columns
    assert 'min_side' in df.columns
    assert 'obj_id' in df.columns


def test_run_analysis_writes_per_timestep_dirs(tmp_path):
    out = tmp_path / 'vis_out'
    report = run_analysis(
        NAV_CSV,
        out,
        objects_csv=OBJ_CSV,
        n_clusters=2,
    )
    assert report['n_detections'] >= 1
    assert report['n_timesteps_analyzed'] >= 1
    assert (out / 'features.csv').is_file()
    assert (out / 'summary.json').is_file()
    assert (out / 'detections_per_timestep.png').is_file()
    # Plots are under timestep_* folders, not pooled at root
    assert not (out / 'corr_pearson.png').exists()
    t0 = out / 'timestep_0000'
    assert t0.is_dir()
    assert (t0 / 'features.csv').is_file()
    assert (t0 / 'bars_min_side_by_object.png').is_file() or (t0 / 'hist_min_side.png').is_file()
    assert (t0 / 'threshold_suggestions.json').is_file()
    summary = summarize_features(load_detection_features(NAV_CSV))
    assert summary['n_detections'] == report['n_detections']


def test_run_analysis_with_episode_folder_meta(tmp_path):
    nav = EPISODE_DIR / 'navigation-house_tiny.csv'
    meta = EPISODE_DIR / 'episode_meta-house_tiny.json'
    objs = EPISODE_DIR / 'objects-house_tiny.csv'
    if not nav.is_file():
        return
    out = tmp_path / 'ep'
    report = run_analysis(nav, out, objects_csv=objs, episode_meta=meta, n_clusters=2)
    assert report['n_detections'] >= 1
    assert (out / 'features.csv').is_file()
    assert any(p.is_dir() and p.name.startswith('timestep_') for p in out.iterdir())


def test_timesteps_filter(tmp_path):
    nav = EPISODE_DIR / 'navigation-house_tiny.csv'
    if not nav.is_file():
        return
    out = tmp_path / 'filt'
    feats = load_detection_features(nav)
    ts = sorted(int(t) for t in feats['timestep'].unique())
    if len(ts) < 1:
        return
    report = run_analysis(nav, out, timesteps=[ts[0]], n_clusters=2)
    assert report['n_timesteps_analyzed'] == 1
    assert (out / f'timestep_{ts[0]:04d}').is_dir()
