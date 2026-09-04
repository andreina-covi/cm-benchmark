"""Question-eligible FOV detection filters (post-process; not collection).

SPOC navigation rows export visibility metrics. Q&A FOV filtering can use:

1. Hard thresholds (``question_visibility``) — **on by default** so tiny /
   barely-visible blobs are dropped even when no joblib model is configured.
2. A trained DecisionTree bundle (``visibility_filter.joblib``) via
   ``predict_proba`` + probability bands (preferred when a model exists).

Pass ``question_visibility=False`` to disable hard thresholds entirely.
The model path is configuration, not code: replace the ``.joblib`` file to
update filtering without changing this module.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Threshold keys ↔ CSV / derived metrics. None = do not enforce that criterion.
QUESTION_VISIBILITY_KEYS = (
    'min_bbox_area',
    'min_side',
    'min_occupancy_ratio',
    'min_visible_pixels',
    'max_obj_distance',
)

# CSV column names used for Q&A FOV filtering and threshold calibration.
VISIBILITY_METRIC_COLUMNS = (
    'obj-distance',
    'bbox-area',
    'min-side',
    'occupancy-ratio',
    'visible-pixels',
)

# metrics_from_nav_row keys ↔ hyphenated CSV / training feature names
FEATURE_TO_METRIC_KEY = {
    'obj-distance': 'obj_distance',
    'bbox-area': 'bbox_area',
    'min-side': 'min_side',
    'occupancy-ratio': 'occupancy_ratio',
    'visible-pixels': 'visible_pixels',
    'ang-width-deg': 'ang_width_deg',
    'ang-height-deg': 'ang_height_deg',
    'expected-bbox-area': 'expected_bbox_area',
}

LABEL_DISTINGUISHABLE = 'distinguible'
LABEL_NOT_DISTINGUISHABLE = 'no_distinguible'
LABEL_AMBIGUOUS = 'ambiguo'

DEFAULT_PROBA_LOW = 0.3
DEFAULT_PROBA_HIGH = 0.7

# Built-in keep rules when no visibility_filter.joblib is provided.
# Tuned to drop speck detections (Potato/Wrench ~10px) while keeping furniture.
DEFAULT_QUESTION_VISIBILITY_THRESHOLDS: dict[str, Optional[float]] = {
    'min_bbox_area': 100.0,
    'min_side': 8.0,
    'min_occupancy_ratio': None,
    'min_visible_pixels': 40.0,
    'max_obj_distance': None,
}


def _all_none_visibility(thresholds: Mapping[str, Any]) -> bool:
    """True if mapping only disables criteria (legacy export / explicit nulls)."""
    vals = [thresholds.get(k) for k in QUESTION_VISIBILITY_KEYS]
    if not any(k in thresholds for k in QUESTION_VISIBILITY_KEYS):
        return False
    return all(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals)


def _finite(val) -> bool:
    if val is None:
        return False
    try:
        f = float(val)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(f) or math.isinf(f))


def _row_get(row: Mapping[str, Any], key: str, default=None):
    if hasattr(row, 'index') and key in row.index:
        return row.get(key, default)
    if isinstance(row, Mapping):
        return row.get(key, default)
    return default


def bbox_sides(cmin, rmin, cmax, rmax) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (width, height, area) or (None, None, None) if bbox invalid."""
    if not all(_finite(v) for v in (cmin, rmin, cmax, rmax)):
        return None, None, None
    w = float(cmax) - float(cmin)
    h = float(rmax) - float(rmin)
    if w <= 0 or h <= 0:
        return None, None, None
    return w, h, w * h


def metrics_from_nav_row(row: Mapping[str, Any], *, frame_w: int = 396, frame_h: int = 224) -> dict[str, Any]:
    """
    Read visibility metrics from a navigation CSV row.

    Prefers exported columns ``bbox-area``, ``min-side``, ``occupancy-ratio``,
    ``visible-pixels``, ``obj-distance``. Falls back to deriving area/side from
    ``cmin..rmax`` when those columns are absent (legacy fixtures).
    """
    cmin = _row_get(row, 'cmin')
    rmin = _row_get(row, 'rmin')
    cmax = _row_get(row, 'cmax')
    rmax = _row_get(row, 'rmax')
    bw, bh, derived_area = bbox_sides(cmin, rmin, cmax, rmax)

    bbox_area = float(_row_get(row, 'bbox-area')) if _finite(_row_get(row, 'bbox-area')) else derived_area
    min_side = float(_row_get(row, 'min-side')) if _finite(_row_get(row, 'min-side')) else (
        min(bw, bh) if bw is not None and bh is not None else None
    )
    occupancy = (
        float(_row_get(row, 'occupancy-ratio'))
        if _finite(_row_get(row, 'occupancy-ratio'))
        else None
    )
    visible_pixels = (
        float(_row_get(row, 'visible-pixels'))
        if _finite(_row_get(row, 'visible-pixels'))
        else None
    )
    obj_distance = (
        float(_row_get(row, 'obj-distance'))
        if _finite(_row_get(row, 'obj-distance'))
        else None
    )

    frame_area = max(int(frame_w) * int(frame_h), 1)
    out = {
        'bbox_w': bw,
        'bbox_h': bh,
        'bbox_area': bbox_area,
        'min_side': min_side,
        'occupancy_ratio': occupancy,
        'visible_pixels': visible_pixels,
        'obj_distance': obj_distance,
        'frame_w': int(frame_w),
        'frame_h': int(frame_h),
        'frame_area': frame_area,
    }
    # Optional angular / expected columns when present in CSV (model features).
    for csv_key, metric_key in (
        ('ang-width-deg', 'ang_width_deg'),
        ('ang-height-deg', 'ang_height_deg'),
        ('expected-bbox-area', 'expected_bbox_area'),
    ):
        val = _row_get(row, csv_key)
        out[metric_key] = float(val) if _finite(val) else None
    return out


def normalize_question_visibility_thresholds(
    thresholds: Optional[Mapping[str, Any] | bool] = None,
) -> dict[str, Optional[float]]:
    """Resolve Q&A FOV thresholds.

    - ``None`` / omitted → built-in defaults (filter **on**)
    - ``False`` → all criteria off (legacy unfiltered behaviour)
    - ``True`` → built-in defaults
    - mapping → merge onto defaults; explicit ``null`` disables that criterion
    - mapping of only nulls (legacy episode export) → restore defaults so
      drafts still filter without a joblib model
    """
    if thresholds is False:
        return {k: None for k in QUESTION_VISIBILITY_KEYS}
    if thresholds is None or thresholds is True:
        return dict(DEFAULT_QUESTION_VISIBILITY_THRESHOLDS)
    if not isinstance(thresholds, Mapping):
        return dict(DEFAULT_QUESTION_VISIBILITY_THRESHOLDS)
    if _all_none_visibility(thresholds):
        return dict(DEFAULT_QUESTION_VISIBILITY_THRESHOLDS)

    out = dict(DEFAULT_QUESTION_VISIBILITY_THRESHOLDS)
    aliases = {
        'min_bbox_side': 'min_side',
        'min_unoccluded_ratio': 'min_occupancy_ratio',
        'min_visible_area_px': 'min_visible_pixels',
        'min_visible_frac': None,
    }
    remapped = {}
    for k, v in thresholds.items():
        key = aliases.get(k, k)
        if key is None or key not in QUESTION_VISIBILITY_KEYS:
            continue
        remapped[key] = v
    for key, val in remapped.items():
        if val is None or (isinstance(val, float) and math.isnan(val)):
            out[key] = None
        else:
            out[key] = float(val)
    return out


def question_visibility_active(thresholds: Mapping[str, Optional[float]]) -> bool:
    return any(thresholds.get(k) is not None for k in QUESTION_VISIBILITY_KEYS)


def passes_question_visibility_filter(
    metrics: Mapping[str, Any],
    thresholds: Optional[Mapping[str, Any]] = None,
) -> bool:
    """
    True if the detection is eligible for Q&A FOV / edges.

    Only criteria with a non-None threshold are enforced. If a metric is missing
    and its threshold is set, that check is skipped (legacy rows without the
    new columns are not mass-dropped).
    """
    thr = normalize_question_visibility_thresholds(thresholds)
    if not question_visibility_active(thr):
        return True

    if thr['min_side'] is not None:
        side = metrics.get('min_side')
        if side is not None and side < thr['min_side']:
            return False

    if thr['min_bbox_area'] is not None:
        area = metrics.get('bbox_area')
        if area is not None and area < thr['min_bbox_area']:
            return False

    if thr['min_occupancy_ratio'] is not None:
        occ = metrics.get('occupancy_ratio')
        if occ is not None and occ < thr['min_occupancy_ratio']:
            return False

    if thr['min_visible_pixels'] is not None:
        pix = metrics.get('visible_pixels')
        if pix is not None and pix < thr['min_visible_pixels']:
            return False

    if thr['max_obj_distance'] is not None:
        dist = metrics.get('obj_distance')
        if dist is not None and dist > thr['max_obj_distance']:
            return False

    return True


def filter_visible_objects_map(
    visible_objects: Optional[Mapping[str, Any]],
    thresholds: Optional[Mapping[str, Any] | bool] = None,
) -> dict[str, Any]:
    """Drop detections that fail ``question_visibility`` (metrics already on values)."""
    thr = normalize_question_visibility_thresholds(thresholds)
    if not question_visibility_active(thr):
        return dict(visible_objects or {})
    kept: dict[str, Any] = {}
    for oid, odata in (visible_objects or {}).items():
        if not isinstance(odata, Mapping):
            kept[oid] = odata
            continue
        metrics = {
            'bbox_area': odata.get('bbox_area'),
            'min_side': odata.get('min_side'),
            'occupancy_ratio': odata.get('occupancy_ratio'),
            'visible_pixels': odata.get('visible_pixels'),
            'obj_distance': odata.get('obj_distance'),
        }
        if passes_question_visibility_filter(metrics, thr):
            kept[oid] = odata
    return kept


def apply_question_visibility_to_episode(
    episode: dict,
    *,
    thresholds: Optional[Mapping[str, Any] | bool] = None,
    inplace: bool = False,
) -> dict:
    """Filter each step's ``visible_objects`` for Q&A drafting.

    Uses ``thresholds`` if given, else episode ``question_visibility``, else
    built-in defaults. Ensures drafts drop tiny blobs even when the episode
    was exported without a visibility joblib / with all-null thresholds.
    """
    ep = episode if inplace else copy.deepcopy(episode)
    if thresholds is None:
        thr = normalize_question_visibility_thresholds(ep.get('question_visibility'))
    else:
        thr = normalize_question_visibility_thresholds(thresholds)
    ep['question_visibility'] = dict(thr)
    for step in ep.get('steps') or []:
        if not isinstance(step, dict):
            continue
        step['visible_objects'] = filter_visible_objects_map(
            step.get('visible_objects'), thr
        )
    return ep


def classify_visibility_proba(
    proba: float,
    low: float = DEFAULT_PROBA_LOW,
    high: float = DEFAULT_PROBA_HIGH,
) -> str:
    """
    Map P(distinguishable) to a ternary label.

    - ``proba >= high`` → distinguible
    - ``proba <= low`` → no_distinguible
    - otherwise → ambiguo
    """
    p = float(proba)
    if p >= high:
        return LABEL_DISTINGUISHABLE
    if p <= low:
        return LABEL_NOT_DISTINGUISHABLE
    return LABEL_AMBIGUOUS


def feature_vector_from_metrics(
    metrics: Mapping[str, Any],
    features: Sequence[str],
) -> Optional[np.ndarray]:
    """Build a 1×F float vector in training column order. None if any feature missing."""
    values = []
    for feat in features:
        metric_key = FEATURE_TO_METRIC_KEY.get(feat, feat.replace('-', '_'))
        val = metrics.get(metric_key)
        if val is None and feat in metrics:
            val = metrics.get(feat)
        if not _finite(val):
            return None
        values.append(float(val))
    return np.asarray(values, dtype=float).reshape(1, -1)


def positive_class_proba(clf: Any, X: np.ndarray) -> np.ndarray:
    """P(class==1) from ``predict_proba`` (never uses ``predict``)."""
    proba = clf.predict_proba(X)
    classes = list(getattr(clf, 'classes_', range(proba.shape[1])))
    if 1 in classes:
        return proba[:, classes.index(1)]
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, -1]


@dataclass
class VisibilityFilterModel:
    """Trained visibility classifier + probability bands for ternary labels."""

    model: Any
    features: list[str]
    low: float = DEFAULT_PROBA_LOW
    high: float = DEFAULT_PROBA_HIGH
    ambiguous_proba_stats: Optional[dict[str, float]] = None
    source_path: Optional[str] = None

    def predict_proba_distinguishable(self, metrics: Mapping[str, Any]) -> Optional[float]:
        X = feature_vector_from_metrics(metrics, self.features)
        if X is None:
            return None
        return float(positive_class_proba(self.model, X)[0])

    def classify_metrics(self, metrics: Mapping[str, Any]) -> str:
        proba = self.predict_proba_distinguishable(metrics)
        if proba is None:
            return LABEL_AMBIGUOUS
        return classify_visibility_proba(proba, low=self.low, high=self.high)

    def passes_for_questions(self, metrics: Mapping[str, Any]) -> bool:
        """Keep only clearly distinguishable detections for Q&A FOV."""
        return self.classify_metrics(metrics) == LABEL_DISTINGUISHABLE


def load_visibility_filter_model(
    path: Union[str, Path],
    *,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> VisibilityFilterModel:
    """
    Load a filter bundle written by ``fit_thresholds.export_visibility_filter_bundle``.

    Accepts either:
    - a dict ``{model, features, low, high, ...}`` (preferred), or
    - a bare sklearn estimator (features default to ``VISIBILITY_METRIC_COLUMNS``).
    """
    import joblib

    path = Path(path)
    obj = joblib.load(path)

    if isinstance(obj, dict) and 'model' in obj:
        features = list(obj.get('features') or VISIBILITY_METRIC_COLUMNS)
        band_low = float(obj['low']) if obj.get('low') is not None else DEFAULT_PROBA_LOW
        band_high = float(obj['high']) if obj.get('high') is not None else DEFAULT_PROBA_HIGH
        stats = obj.get('ambiguous_proba_stats')
        model = obj['model']
    else:
        features = list(VISIBILITY_METRIC_COLUMNS)
        band_low, band_high = DEFAULT_PROBA_LOW, DEFAULT_PROBA_HIGH
        stats = None
        model = obj

    if low is not None:
        band_low = float(low)
    if high is not None:
        band_high = float(high)
    if band_low > band_high:
        raise ValueError(f'low ({band_low}) must be <= high ({band_high})')

    return VisibilityFilterModel(
        model=model,
        features=features,
        low=band_low,
        high=band_high,
        ambiguous_proba_stats=stats,
        source_path=str(path.resolve()),
    )


def calibrate_proba_bands_from_ambiguous(
    clf: Any,
    df_ambiguous: pd.DataFrame,
    features: Sequence[str],
    *,
    default_low: float = DEFAULT_PROBA_LOW,
    default_high: float = DEFAULT_PROBA_HIGH,
) -> tuple[float, float, dict[str, float]]:
    """
    Set (low, high) from P(class=1) on human-labeled ambiguous rows (y=2).

    Uses min/max of those probabilities as the ambiguous band so new samples
    inside that range stay ``ambiguo``. Falls back to defaults if empty.
    """
    if df_ambiguous is None or df_ambiguous.empty:
        stats = {'n': 0.0, 'min': float('nan'), 'max': float('nan'), 'mean': float('nan')}
        return default_low, default_high, stats

    missing = [c for c in features if c not in df_ambiguous.columns]
    if missing:
        raise KeyError(f'Ambiguous frame missing feature columns: {missing}')

    X = df_ambiguous[list(features)].to_numpy(dtype=float)
    proba = positive_class_proba(clf, X)
    stats = {
        'n': float(len(proba)),
        'min': float(np.min(proba)),
        'max': float(np.max(proba)),
        'mean': float(np.mean(proba)),
    }
    low = stats['min']
    high = stats['max']
    if low > high:
        low, high = default_low, default_high
    # Degenerate band (all amb at one proba) → keep a small margin around it
    if abs(high - low) < 1e-9:
        mid = low
        low = max(0.0, mid - 0.05)
        high = min(1.0, mid + 0.05)
    return float(low), float(high), stats


def threshold_sweep_keep_rates(
    df: pd.DataFrame,
    *,
    bbox_area_values: Optional[list[float]] = None,
    side_values: Optional[list[float]] = None,
    occupancy_values: Optional[list[float]] = None,
    visible_pixels_values: Optional[list[float]] = None,
    distance_values: Optional[list[float]] = None,
) -> pd.DataFrame:
    """1D keep-rate sweeps for calibrating question_visibility thresholds."""
    cols = ['criterion', 'threshold', 'n_keep', 'n_total', 'keep_rate']
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    n = len(df)
    rows = []

    def _ge(criterion, values, series):
        if not values or series is None:
            return
        for t in values:
            keep = int(((series.isna()) | (series >= t)).sum())
            rows.append({
                'criterion': criterion,
                'threshold': t,
                'n_keep': keep,
                'n_total': n,
                'keep_rate': keep / n if n else 0.0,
            })

    def _le(criterion, values, series):
        if not values or series is None:
            return
        for t in values:
            keep = int(((series.isna()) | (series <= t)).sum())
            rows.append({
                'criterion': criterion,
                'threshold': t,
                'n_keep': keep,
                'n_total': n,
                'keep_rate': keep / n if n else 0.0,
            })

    _ge('min_bbox_area', bbox_area_values, df['bbox_area'] if 'bbox_area' in df else None)
    _ge('min_side', side_values, df['min_side'] if 'min_side' in df else None)
    _ge(
        'min_occupancy_ratio',
        occupancy_values,
        df['occupancy_ratio'] if 'occupancy_ratio' in df else None,
    )
    _ge(
        'min_visible_pixels',
        visible_pixels_values,
        df['visible_pixels'] if 'visible_pixels' in df else None,
    )
    _le(
        'max_obj_distance',
        distance_values,
        df['obj_distance'] if 'obj_distance' in df else None,
    )
    return pd.DataFrame(rows)


def resolve_hyperparams_from_episode_meta(
    episode_meta: Optional[Mapping[str, Any]] = None,
    hyperparams: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build generator hyperparams; prefer episode_meta.camera / .agent over defaults."""
    base = {
        'w': 396,
        'h': 224,
        'fov_v': 59,
        'k_neighbors': 3,
        'radius': 1.5,
        'ex': 0.1,
        'ey': 0.1,
        'ez': 0.15,
        'angle_threshold_xz': 15,
        'min_distance': 0.5,
        'med_distance': 1.0,
        'max_distance': 1.5,
        'mov_constant': 0.2,
        'question_visibility': dict(DEFAULT_QUESTION_VISIBILITY_THRESHOLDS),
        'visibility_model_path': None,
    }

    meta = episode_meta or {}
    camera = meta.get('camera') or {}
    agent = meta.get('agent') or {}

    if _finite(camera.get('width')):
        base['w'] = int(camera['width'])
    if _finite(camera.get('height')):
        base['h'] = int(camera['height'])
    if _finite(camera.get('fov_vertical_deg')):
        base['fov_v'] = float(camera['fov_vertical_deg'])
    if _finite(agent.get('movement_constant')):
        base['mov_constant'] = float(agent['movement_constant'])

    if hyperparams:
        qv = hyperparams.get('question_visibility')
        for k, v in hyperparams.items():
            if k == 'question_visibility':
                continue
            base[k] = v
        if qv is not None:
            base['question_visibility'] = normalize_question_visibility_thresholds(qv)

    base['question_visibility'] = normalize_question_visibility_thresholds(
        base.get('question_visibility')
    )
    return base
