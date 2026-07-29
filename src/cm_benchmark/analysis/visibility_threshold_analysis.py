"""Analyze navigation FOV detections **per timestep** to choose Q&A thresholds.

Each timestep is analyzed separately (objects visible in that frame only).
Outputs live under ``output_dir/timestep_XXXX/``.

Reads SPOC ``navigation-*.csv`` visibility fields (``obj-distance``, ``bbox-area``,
``min-side``, ``occupancy-ratio``, ``visible-pixels``).

Example::

    python -m cm_benchmark.analysis.visibility_threshold_analysis \\
      --navigation_csv /path/to/annotations/navigation-house_XXXXXX.csv \\
      --objects_csv    /path/to/annotations/objects-house_XXXXXX.csv \\
      --episode_meta   /path/to/annotations/episode_meta-house_XXXXXX.json \\
      --output_dir     analysis/visibility_house_XXXXXX
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cm_benchmark.generator.episode_paths import is_structural_object
from cm_benchmark.generator.visibility_filters import (
    metrics_from_nav_row,
    threshold_sweep_keep_rates,
)


def _finite(val) -> bool:
    if val is None:
        return False
    try:
        f = float(val)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(f) or math.isinf(f))

FEATURE_COLS = (
    'bbox_area',
    'min_side',
    'occupancy_ratio',
    'visible_pixels',
    'obj_distance',
)

# Defaults for keep-rate sweeps (calibration candidates, not locked thresholds)
DEFAULT_SIDE_SWEEP = [4, 8, 12, 16, 24, 32]
DEFAULT_AREA_SWEEP = [50, 100, 200, 400, 800, 1600]
DEFAULT_OCC_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
DEFAULT_PIXEL_SWEEP = [50, 100, 200, 400, 800]
DEFAULT_DIST_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def _load_frame_size(
    episode_meta_path: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> tuple[int, int]:
    w, h = 396, 224
    if episode_meta_path is not None and episode_meta_path.is_file():
        meta = json.loads(episode_meta_path.read_text())
        cam = meta.get('camera') or {}
        if _finite(cam.get('width')):
            w = int(cam['width'])
        if _finite(cam.get('height')):
            h = int(cam['height'])
    if width is not None:
        w = int(width)
    if height is not None:
        h = int(height)
    return w, h


def load_detection_features(
    navigation_csv: str | Path,
    *,
    objects_csv: Optional[str | Path] = None,
    episode_meta: Optional[str | Path] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build one row per non-structural FOV detection with visibility features.

    Columns: timestep, obj_id, obj_type (if catalog joined), bbox, FEATURE_COLS, …
    """
    nav_path = Path(navigation_csv)
    if not nav_path.is_file():
        raise FileNotFoundError(nav_path)

    frame_w, frame_h = _load_frame_size(
        Path(episode_meta) if episode_meta else None,
        width=frame_width,
        height=frame_height,
    )

    type_by_id: dict[str, str] = {}
    if objects_csv is not None:
        obj_path = Path(objects_csv)
        if obj_path.is_file():
            odf = pd.read_csv(obj_path)
            for _, row in odf.iterrows():
                oid = row.get('obj-id')
                if oid is None or (isinstance(oid, float) and math.isnan(oid)):
                    continue
                type_by_id[str(oid)] = row.get('obj-type')

    df = pd.read_csv(nav_path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        oid = row.get('obj-id')
        if oid is None or (isinstance(oid, float) and math.isnan(oid)):
            continue
        oid_s = str(oid)
        obj_type = type_by_id.get(oid_s)
        if is_structural_object(obj_type=obj_type, object_id=oid_s):
            continue

        metrics = metrics_from_nav_row(row, frame_w=frame_w, frame_h=frame_h)
        entry = {
            'timestep': int(row['timestep']) if pd.notna(row.get('timestep')) else None,
            'obj_id': oid_s,
            'obj_type': obj_type,
            'cmin': row.get('cmin'),
            'rmin': row.get('rmin'),
            'cmax': row.get('cmax'),
            'rmax': row.get('rmax'),
            **metrics,
        }
        rows.append(entry)

    out = pd.DataFrame(rows)
    out.attrs['frame_w'] = frame_w
    out.attrs['frame_h'] = frame_h
    return out


def summarize_features(df: pd.DataFrame) -> dict[str, Any]:
    """Quantiles / missingness for visibility features."""
    summary: dict[str, Any] = {
        'n_detections': int(len(df)),
        'n_unique_objects': int(df['obj_id'].nunique()) if len(df) else 0,
        'frame_w': df.attrs.get('frame_w'),
        'frame_h': df.attrs.get('frame_h'),
        'features': {},
    }
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce')
        summary['features'][col] = {
            'n_non_null': int(s.notna().sum()),
            'missing_frac': float(s.isna().mean()) if len(s) else 1.0,
            'min': float(s.min()) if s.notna().any() else None,
            'p25': float(s.quantile(0.25)) if s.notna().any() else None,
            'median': float(s.median()) if s.notna().any() else None,
            'p75': float(s.quantile(0.75)) if s.notna().any() else None,
            'p90': float(s.quantile(0.90)) if s.notna().any() else None,
            'p95': float(s.quantile(0.95)) if s.notna().any() else None,
            'max': float(s.max()) if s.notna().any() else None,
            'mean': float(s.mean()) if s.notna().any() else None,
        }
    return summary


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close()


def plot_histograms(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    title_prefix: str = '',
) -> list[Path]:
    paths = []
    for col in FEATURE_COLS:
        if col not in df.columns or df[col].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col].dropna(), bins=min(40, max(5, len(df))), kde=len(df) >= 8, ax=ax, color='#009384')
        ax.set_title(f'{title_prefix}{col}')
        ax.set_xlabel(col)
        ax.set_ylabel('count')
        p = output_dir / f'hist_{col}.png'
        _save_fig(p)
        paths.append(p)
    return paths


def plot_correlation_matrix(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    title_prefix: str = '',
) -> Optional[Path]:
    cols = [c for c in FEATURE_COLS if c in df.columns and df[c].notna().sum() >= 2]
    if len(cols) < 2:
        return None
    sub = df[cols].apply(pd.to_numeric, errors='coerce')
    corr = sub.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='vlag',
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title(f'{title_prefix}Pearson correlation')
    p = output_dir / 'corr_pearson.png'
    _save_fig(p)

    corr.to_csv(output_dir / 'corr_pearson.csv')
    spear = sub.corr(method='spearman')
    spear.to_csv(output_dir / 'corr_spearman.csv')
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        spear,
        annot=True,
        fmt='.2f',
        cmap='vlag',
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title(f'{title_prefix}Spearman correlation')
    p2 = output_dir / 'corr_spearman.png'
    _save_fig(p2)
    return p


def plot_scatter_pairs(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    title_prefix: str = '',
) -> list[Path]:
    pairs = [
        ('min_side', 'occupancy_ratio'),
        ('bbox_area', 'occupancy_ratio'),
        ('obj_distance', 'min_side'),
        ('obj_distance', 'bbox_area'),
        ('visible_pixels', 'bbox_area'),
        ('visible_pixels', 'occupancy_ratio'),
    ]
    paths = []
    for x, y in pairs:
        if x not in df.columns or y not in df.columns:
            continue
        cols = [x, y]
        if 'obj_id' in df.columns:
            cols.append('obj_id')
        sub = df[cols].dropna(subset=[x, y])
        if len(sub) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.scatter(sub[x], sub[y], s=28, alpha=0.7, c='#002F4A')
        # Label points with short object id when few detections in this frame
        if 'obj_id' in sub.columns and len(sub) <= 40:
            for _, r in sub.iterrows():
                oid = str(r['obj_id'])
                short = oid if len(oid) <= 18 else oid.split('|')[0][:14]
                ax.annotate(short, (r[x], r[y]), fontsize=6, alpha=0.8)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f'{title_prefix}{y} vs {x}')
        p = output_dir / f'scatter_{x}_vs_{y}.png'
        _save_fig(p)
        paths.append(p)
    return paths


def plot_detections_bar_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Episode overview only: how many FOV detections at each timestep."""
    if 'timestep' not in df.columns or df['timestep'].dropna().empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df.groupby('timestep').size()
    ax.bar(counts.index.astype(int), counts.values, color='#009384', width=0.8)
    ax.set_xlabel('timestep')
    ax.set_ylabel('n detections')
    ax.set_title('FOV detections per timestep (episode overview)')
    p = output_dir / 'detections_per_timestep.png'
    _save_fig(p)
    return p


def plot_objects_bar_at_timestep(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    title_prefix: str = '',
) -> list[Path]:
    """Bar charts of each object's metric values at this single timestep."""
    if df.empty or 'obj_id' not in df.columns:
        return []
    paths = []
    work = df.sort_values('obj_id')
    for col in FEATURE_COLS:
        if col not in work.columns or work[col].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(work)), 4.5))
        labels = [
            (oid if len(str(oid)) <= 22 else str(oid)[:19] + '…')
            for oid in work['obj_id']
        ]
        ax.bar(range(len(work)), work[col].to_numpy(), color='#009384')
        ax.set_xticks(range(len(work)))
        ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=7)
        ax.set_ylabel(col)
        ax.set_title(f'{title_prefix}{col} by object')
        ax.grid(True, axis='y', alpha=0.25)
        p = output_dir / f'bars_{col}_by_object.png'
        _save_fig(p)
        paths.append(p)
    return paths


def plot_keep_rate_sweeps(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    title_prefix: str = '',
) -> tuple[pd.DataFrame, list[Path]]:
    sweep = threshold_sweep_keep_rates(
        df,
        bbox_area_values=DEFAULT_AREA_SWEEP,
        side_values=DEFAULT_SIDE_SWEEP,
        occupancy_values=DEFAULT_OCC_SWEEP,
        visible_pixels_values=DEFAULT_PIXEL_SWEEP,
        distance_values=DEFAULT_DIST_SWEEP,
    )
    sweep_path = output_dir / 'keep_rate_sweep.csv'
    sweep.to_csv(sweep_path, index=False)
    paths = [sweep_path]

    if sweep.empty:
        return sweep, paths

    for criterion, group in sweep.groupby('criterion'):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(group['threshold'], group['keep_rate'], marker='o', color='#B85741')
        ax.set_xlabel(criterion)
        ax.set_ylabel('keep rate')
        ax.set_ylim(0, 1.05)
        ax.set_title(f'{title_prefix}Keep rate vs {criterion}')
        ax.grid(True, alpha=0.3)
        p = output_dir / f'sweep_{criterion}.png'
        _save_fig(p)
        paths.append(p)
    return sweep, paths


def plot_joint_keep_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    side_values: Sequence[float] = DEFAULT_SIDE_SWEEP,
    occ_values: Sequence[float] = DEFAULT_OCC_SWEEP,
    title_prefix: str = '',
) -> Optional[Path]:
    """2D keep-rate: min_side × min_occupancy_ratio."""
    if df.empty or 'min_side' not in df.columns or 'occupancy_ratio' not in df.columns:
        return None
    side = pd.to_numeric(df['min_side'], errors='coerce')
    occ = pd.to_numeric(df['occupancy_ratio'], errors='coerce')
    n = len(df)
    if n == 0:
        return None

    mat = np.zeros((len(occ_values), len(side_values)))
    for i, o_thr in enumerate(occ_values):
        for j, s_thr in enumerate(side_values):
            keep = (
                ((side.isna()) | (side >= s_thr))
                & ((occ.isna()) | (occ >= o_thr))
            ).sum()
            mat[i, j] = keep / n

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        mat,
        annot=True,
        fmt='.2f',
        xticklabels=[str(v) for v in side_values],
        yticklabels=[str(v) for v in occ_values],
        cmap='YlGnBu',
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel('min_side')
    ax.set_ylabel('min_occupancy_ratio')
    ax.set_title(f'{title_prefix}Joint keep rate (missing metric = keep)')
    p = output_dir / 'keep_heatmap_side_x_occupancy.png'
    _save_fig(p)
    return p


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    # Columns that are all-NaN → mean 0, std 1
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(np.isnan(std) | (std < 1e-8), 1.0, std)
    X = X.copy()
    inds = np.where(np.isnan(X))
    X[inds] = np.take(mean, inds[1])
    return (X - mean) / std, mean, std


def kmeans_numpy(
    X: np.ndarray,
    n_clusters: int = 3,
    *,
    max_iter: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """Simple k-means (Lloyd). Returns cluster labels."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    k = min(n_clusters, n)
    centers = X[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = X[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    return labels


def cluster_detections(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    n_clusters: int = 3,
    seed: int = 0,
    title_prefix: str = '',
) -> tuple[pd.DataFrame, Optional[Path]]:
    """
    Cluster detections on log1p(bbox_area), min_side, occupancy_ratio, obj_distance.

    Helps see natural groups (tiny/sliver vs clear nearby vs distant).
    """
    use_cols = ['bbox_area', 'min_side', 'occupancy_ratio', 'obj_distance']
    present = [c for c in use_cols if c in df.columns]
    if len(present) < 2 or df.empty:
        return df, None

    work = df.copy()
    X_list = []
    names = []
    for c in present:
        s = pd.to_numeric(work[c], errors='coerce')
        if s.notna().sum() == 0:
            continue
        if c == 'bbox_area':
            s = np.log1p(s)
            names.append('log1p_bbox_area')
        else:
            names.append(c)
        X_list.append(s.to_numpy(dtype=float))
    if len(X_list) < 2:
        return df, None
    X = np.column_stack(X_list)
    Xs, _, _ = _standardize(X)
    labels = kmeans_numpy(Xs, n_clusters=n_clusters, seed=seed)
    work['cluster'] = labels

    profile_rows = []
    for cid, group in work.groupby('cluster'):
        row = {'cluster': int(cid), 'n': int(len(group))}
        for c in FEATURE_COLS:
            if c in group.columns:
                row[f'{c}_median'] = float(pd.to_numeric(group[c], errors='coerce').median())
        profile_rows.append(row)
    profile = pd.DataFrame(profile_rows).sort_values('cluster')
    profile.to_csv(output_dir / 'cluster_profiles.csv', index=False)
    work.to_csv(output_dir / 'features_with_clusters.csv', index=False)

    path = None
    if 'min_side' in work.columns and 'occupancy_ratio' in work.columns:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for cid, group in work.groupby('cluster'):
            ax.scatter(
                group['min_side'],
                group['occupancy_ratio'],
                s=28,
                alpha=0.7,
                label=f'cluster {cid} (n={len(group)})',
            )
        ax.set_xlabel('min_side')
        ax.set_ylabel('occupancy_ratio')
        ax.set_title(f'{title_prefix}K-means clusters (k={n_clusters})')
        ax.legend(fontsize=8)
        path = output_dir / 'clusters_min_side_vs_occupancy.png'
        _save_fig(path)

    if 'obj_type' in work.columns and work['obj_type'].notna().any():
        counts = (
            work.dropna(subset=['obj_type'])
            .groupby(['cluster', 'obj_type'])
            .size()
            .reset_index(name='n')
        )
        counts.to_csv(output_dir / 'cluster_obj_type_counts.csv', index=False)

    return work, path


def plot_by_obj_type(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 12,
    *,
    title_prefix: str = '',
) -> list[Path]:
    if 'obj_type' not in df.columns or df['obj_type'].dropna().empty:
        return []
    paths = []
    top = df['obj_type'].value_counts().head(top_n).index.tolist()
    sub = df[df['obj_type'].isin(top)]
    for col in ('min_side', 'occupancy_ratio', 'bbox_area', 'obj_distance'):
        if col not in sub.columns or sub[col].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        order = (
            sub.groupby('obj_type')[col]
            .median()
            .sort_values()
            .index.tolist()
        )
        sns.boxplot(data=sub, x='obj_type', y=col, order=order, ax=ax, color='#D9C4B1')
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha('right')
        ax.set_title(f'{title_prefix}{col} by obj_type')
        p = output_dir / f'boxplot_{col}_by_type.png'
        _save_fig(p)
        paths.append(p)
    return paths


def _threshold_suggestions(df: pd.DataFrame) -> dict[str, Any]:
    suggestions: dict[str, Any] = {}
    for col, key in (
        ('min_side', 'min_side'),
        ('bbox_area', 'min_bbox_area'),
        ('occupancy_ratio', 'min_occupancy_ratio'),
        ('visible_pixels', 'min_visible_pixels'),
    ):
        if col in df.columns and df[col].notna().any():
            suggestions[key] = {
                'p25': float(df[col].quantile(0.25)),
                'median': float(df[col].median()),
                'note': 'Frame-local quantiles; compare across timesteps before locking.',
            }
    if 'obj_distance' in df.columns and df['obj_distance'].notna().any():
        suggestions['max_obj_distance'] = {
            'p75': float(df['obj_distance'].quantile(0.75)),
            'p90': float(df['obj_distance'].quantile(0.90)),
            'note': 'Optional upper bound for this timestep.',
        }
    return suggestions


def analyze_single_timestep(
    df_t: pd.DataFrame,
    output_dir: Path,
    *,
    timestep: int,
    n_clusters: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the full plot suite on detections from one timestep only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'timestep={timestep} | '
    df_t.to_csv(output_dir / 'features.csv', index=False)

    summary = summarize_features(df_t)
    summary['timestep'] = int(timestep)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

    plot_objects_bar_at_timestep(df_t, output_dir, title_prefix=prefix)
    plot_histograms(df_t, output_dir, title_prefix=prefix)
    plot_correlation_matrix(df_t, output_dir, title_prefix=prefix)
    plot_scatter_pairs(df_t, output_dir, title_prefix=prefix)
    sweep, _ = plot_keep_rate_sweeps(df_t, output_dir, title_prefix=prefix)
    plot_joint_keep_heatmap(df_t, output_dir, title_prefix=prefix)
    plot_by_obj_type(df_t, output_dir, title_prefix=prefix)
    k = min(n_clusters, max(1, len(df_t)))
    clustered, _ = cluster_detections(
        df_t, output_dir, n_clusters=k, seed=seed, title_prefix=prefix
    )

    suggestions = _threshold_suggestions(df_t)
    (output_dir / 'threshold_suggestions.json').write_text(json.dumps(suggestions, indent=2))

    return {
        'timestep': int(timestep),
        'n_detections': len(df_t),
        'n_clusters': int(clustered['cluster'].nunique()) if 'cluster' in clustered.columns else 0,
        'sweep_rows': int(len(sweep)),
        'threshold_suggestions': suggestions,
    }


def run_analysis(
    navigation_csv: str | Path,
    output_dir: str | Path,
    *,
    objects_csv: Optional[str | Path] = None,
    episode_meta: Optional[str | Path] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    n_clusters: int = 3,
    seed: int = 0,
    timesteps: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """
    Analyze FOV detections **separately for each timestep**.

    Layout::

        output_dir/
          features.csv                 # all detections (timestep column)
          detections_per_timestep.png  # episode overview count only
          summary.json
          report.json
          timestep_0000/               # objects at t=0 only
            features.csv, hist_*.png, scatter_*.png, ...
          timestep_0001/
            ...
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_detection_features(
        navigation_csv,
        objects_csv=objects_csv,
        episode_meta=episode_meta,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    df.to_csv(out / 'features.csv', index=False)

    summary = summarize_features(df)
    summary['n_timesteps'] = int(df['timestep'].nunique()) if 'timestep' in df.columns else 0
    if 'timestep' in df.columns:
        summary['detections_per_timestep'] = {
            str(int(t)): int(n) for t, n in df.groupby('timestep').size().items()
        }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))

    # Episode-level overview only (not pooled metric plots)
    plot_detections_bar_chart(df, out)

    if 'timestep' not in df.columns or df['timestep'].dropna().empty:
        # Fallback: treat whole table as one frame
        per_t = [analyze_single_timestep(df, out / 'timestep_0000', timestep=0, n_clusters=n_clusters, seed=seed)]
    else:
        ts = sorted(int(t) for t in df['timestep'].dropna().unique())
        if timesteps is not None:
            wanted = {int(t) for t in timesteps}
            ts = [t for t in ts if t in wanted]
        per_t = []
        for t in ts:
            df_t = df[df['timestep'] == t].copy()
            t_dir = out / f'timestep_{t:04d}'
            per_t.append(
                analyze_single_timestep(
                    df_t, t_dir, timestep=t, n_clusters=n_clusters, seed=seed
                )
            )

    report = {
        'n_detections': len(df),
        'n_timesteps_analyzed': len(per_t),
        'output_dir': str(out.resolve()),
        'summary': summary,
        'per_timestep': per_t,
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Per-timestep plots of nav FOV visibility metrics for Q&A thresholds.'
    )
    p.add_argument('--navigation_csv', required=True, help='Path to navigation-*.csv')
    p.add_argument('--objects_csv', default=None, help='Optional objects-*.csv for obj-type')
    p.add_argument('--episode_meta', default=None, help='Optional episode_meta-*.json (W/H)')
    p.add_argument('--frame_width', type=int, default=None)
    p.add_argument('--frame_height', type=int, default=None)
    p.add_argument('--output_dir', required=True, help='Directory for CSV/PNG/JSON outputs')
    p.add_argument('--n_clusters', type=int, default=3, help='K-means clusters per timestep')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument(
        '--timesteps',
        default=None,
        help='Optional comma-separated timesteps to analyze (default: all)',
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    ts = None
    if args.timesteps:
        ts = [int(x.strip()) for x in args.timesteps.split(',') if x.strip() != '']
    report = run_analysis(
        args.navigation_csv,
        args.output_dir,
        objects_csv=args.objects_csv,
        episode_meta=args.episode_meta,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        n_clusters=args.n_clusters,
        seed=args.seed,
        timesteps=ts,
    )
    print(
        json.dumps(
            {
                'n_detections': report['n_detections'],
                'n_timesteps_analyzed': report['n_timesteps_analyzed'],
                'output_dir': report['output_dir'],
            },
            indent=2,
        )
    )
    print(f"Wrote per-timestep analysis under {report['output_dir']}")


if __name__ == '__main__':
    main()
