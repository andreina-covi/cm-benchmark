"""Question-eligible FOV detection filters (post-process; not collection).

SPOC navigation rows now export simple visibility fields:

- ``obj-distance`` — agent→object distance (m)
- ``bbox-area`` — detection bbox area (px²)
- ``min-side`` — shorter bbox side (px)
- ``occupancy-ratio`` — mask pixels / bbox area (how full the box is)
- ``visible-pixels`` — instance-mask pixel count (optional criterion)

Tiny / barely filled detections may still be exported; Q&A FOV filtering uses
tunable thresholds (default: all off) so you can calibrate from stats/plots.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

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

DEFAULT_QUESTION_VISIBILITY_THRESHOLDS: dict[str, Optional[float]] = {
    'min_bbox_area': None,
    'min_side': None,
    'min_occupancy_ratio': None,
    'min_visible_pixels': None,
    'max_obj_distance': None,
}


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
    return {
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


def normalize_question_visibility_thresholds(
    thresholds: Optional[Mapping[str, Any]] = None,
) -> dict[str, Optional[float]]:
    out = dict(DEFAULT_QUESTION_VISIBILITY_THRESHOLDS)
    if not thresholds:
        return out
    # Accept a few aliases from earlier drafts
    aliases = {
        'min_bbox_side': 'min_side',
        'min_unoccluded_ratio': 'min_occupancy_ratio',
        'min_visible_area_px': 'min_visible_pixels',
        'min_visible_frac': None,  # dropped; ignore if passed
    }
    remapped = {}
    for k, v in thresholds.items():
        key = aliases.get(k, k)
        if key is None:
            continue
        remapped[key] = v
    for key in QUESTION_VISIBILITY_KEYS:
        if key not in remapped:
            continue
        val = remapped[key]
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
            # Missing metric → keep (same as filter policy)
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
