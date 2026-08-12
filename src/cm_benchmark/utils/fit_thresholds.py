"""
Calibrate visibility thresholds from labeled scenes.

With labels JSON (from the HTML labeling tool) + calibration_manifest CSV
(from build_labeling_set.py), this:

1. Fits a shallow decision tree over features → readable threshold rules.
2. Validates with leave-one-scene-out (LOSO).
3. Optionally **tunes** DecisionTree hyperparameters by grid search scored on
   LOSO mean ± std (robustness = high mean, low cross-scene variance).
4. Writes tune plots when ``--tune`` is used (unless ``--no_plots``).

Multi-scene usage
-----------------
Put pairs in one folder::

    house_007514_calibration_manifest.csv
    house_007514_labels.json
    ...

Examples::

    python -m cm_benchmark.utils.fit_thresholds --folder path/to/scenes
    python -m cm_benchmark.utils.fit_thresholds --folder path/to/scenes --tune
    python -m cm_benchmark.utils.fit_thresholds --folder path/to/scenes --tune \\
        --tune_out analysis/dt_tune
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier, export_text

from cm_benchmark.generator.visibility_filters import VISIBILITY_METRIC_COLUMNS

# Align with question_visibility / visibility_filters (not angular proxies).
FEATURES = list(VISIBILITY_METRIC_COLUMNS)

# Default grid for robustness search (keep trees shallow → readable thresholds)
DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf': [3, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None],
}


def load_data(manifest_csv: str | Path, labels_json: str | Path) -> pd.DataFrame:
    """
    Merge one scene's manifest CSV with its labels JSON.

    manifest_csv : calibration_manifest.csv from build_labeling_set.py
                   (must include item_id; preferably scene_id)
    labels_json  : {item_id: {label: 0|1|2, ...}, ...} from the HTML tool
    """
    manifest_csv = Path(manifest_csv)
    labels_json = Path(labels_json)
    df = pd.read_csv(manifest_csv)
    with open(labels_json) as f:
        labels = json.load(f)

    df['item_id'] = df['item_id'].astype(str)
    lab_df = pd.DataFrame(labels).T.rename(columns={'label': 'y'})
    lab_df.index.name = 'item_id'
    lab_df = lab_df.reset_index()[['item_id', 'y']]
    lab_df['item_id'] = lab_df['item_id'].astype(str)
    df = df.merge(lab_df, on='item_id', how='inner')

    if 'scene_id' not in df.columns or df['scene_id'].isna().all():
        stem = _scene_stem_from_manifest_name(manifest_csv.name)
        if stem:
            df['scene_id'] = stem
        else:
            df['scene_id'] = manifest_csv.stem

    df['scene_id'] = df['scene_id'].astype(str)
    df['item_id_global'] = df['scene_id'] + '::' + df['item_id'].astype(str)
    df['source_manifest'] = str(manifest_csv.resolve())
    df['source_labels'] = str(labels_json.resolve())
    return df


def _scene_stem_from_manifest_name(name: str) -> Optional[str]:
    for suf in ('_calibration_manifest.csv', '_calibration_sample.csv'):
        if name.endswith(suf):
            return name[: -len(suf)] or None
    if name in ('calibration_manifest.csv', 'calibration_sample.csv'):
        return None
    return None


def discover_scene_pairs(folder: str | Path) -> list[tuple[Path, Path, str]]:
    """
    Find (manifest_csv, labels_json, scene_id) pairs in a folder.

    Preferred:: ``{scene}_calibration_manifest.csv`` + ``{scene}_labels.json``
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    manifests = sorted(folder.glob('*calibration_manifest.csv')) + sorted(
        folder.glob('*calibration_sample.csv')
    )
    seen: set[Path] = set()
    manifests_u: list[Path] = []
    for p in manifests:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        manifests_u.append(p)

    labels_files = {
        p.name: p
        for p in list(folder.glob('*_labels.json'))
        + list(folder.glob('*_label.json'))
        + list(folder.glob('labels.json'))
        + list(folder.glob('label.json'))
    }

    pairs: list[tuple[Path, Path, str]] = []
    used_labels: set[Path] = set()

    for man in manifests_u:
        stem = _scene_stem_from_manifest_name(man.name)
        label_path = None
        scene_id = stem

        if stem:
            for cand in (f'{stem}_labels.json', f'{stem}_label.json'):
                if cand in labels_files:
                    label_path = labels_files[cand]
                    break
        if label_path is None and man.name in (
            'calibration_manifest.csv',
            'calibration_sample.csv',
        ):
            for cand in ('labels.json', 'label.json'):
                if cand in labels_files:
                    label_path = labels_files[cand]
                    scene_id = scene_id or folder.name or 'scene'
                    break

        if label_path is None:
            print(f'WARN: no labels JSON paired with {man.name}; skipping')
            continue

        used_labels.add(label_path.resolve())
        pairs.append((man, label_path, scene_id or man.stem))

    for name, p in labels_files.items():
        if p.resolve() not in used_labels:
            print(f'WARN: labels file not paired with a manifest: {name}')

    if not pairs:
        raise FileNotFoundError(
            f'No scene pairs found in {folder}. Expected files like '
            f'house_XXXXXX_calibration_manifest.csv + house_XXXXXX_labels.json'
        )
    return pairs


def load_data_from_folder(folder: str | Path) -> pd.DataFrame:
    """Load every scene pair in ``folder`` and concatenate (keeps ``scene_id``)."""
    frames = []
    for man, lab, scene_id in discover_scene_pairs(folder):
        df = load_data(man, lab)
        if df['scene_id'].nunique() == 1 and df['scene_id'].iloc[0] in (
            man.stem,
            Path(man).stem,
        ):
            df['scene_id'] = scene_id
        df['scene_id'] = df['scene_id'].fillna(scene_id).astype(str)
        df['item_id_global'] = df['scene_id'] + '::' + df['item_id'].astype(str)
        print(
            f'  scene={df["scene_id"].iloc[0]}: {len(df)} labeled items '
            f'({man.name} + {lab.name})'
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    n_scenes = combined['scene_id'].nunique()
    print(
        f'Combined: {len(combined)} items across {n_scenes} scene(s): '
        f'{sorted(combined["scene_id"].unique())}'
    )
    return combined


def _prepare_xy(
    df: pd.DataFrame,
    features: list[str],
    scene_col: str = 'scene_id',
):
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f'Missing feature columns: {missing}')
    if scene_col not in df.columns:
        raise KeyError(f'Missing scene column: {scene_col}')
    X = df[features].to_numpy()
    y = df['y'].astype(int).to_numpy()
    groups = df[scene_col].to_numpy()
    return X, y, groups


def fit_tree(
    df,
    features=FEATURES,
    *,
    max_depth=3,
    min_samples_leaf=10,
    min_samples_split=2,
    criterion='gini',
    class_weight='balanced',
    random_state=42,
):
    X = df[list(features)].to_numpy()
    y = df['y'].astype(int).to_numpy()
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        criterion=criterion,
        class_weight=class_weight,
        random_state=random_state,
    )
    clf.fit(X, y)
    print(export_text(clf, feature_names=list(features)))
    return clf


def loso_fold_metrics(
    df: pd.DataFrame,
    tree_params: dict[str, Any],
    *,
    features: list[str] = FEATURES,
    scene_col: str = 'scene_id',
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run leave-one-scene-out once for a fixed DecisionTree hyperparameter set.

    Returns one row per held-out scene with AUC / F1 / n_test.
    """
    X, y, groups = _prepare_xy(df, list(features), scene_col=scene_col)
    n_scenes = pd.Series(groups).nunique()
    if n_scenes < 2:
        raise ValueError(
            f'LOSO needs ≥2 scenes; found {n_scenes}. '
            'Put multiple CSV+JSON pairs in --folder.'
        )

    logo = LeaveOneGroupOut()
    rows = []
    for train_idx, test_idx in logo.split(X, y, groups):
        params = {**tree_params, 'random_state': random_state}
        clf = DecisionTreeClassifier(**params)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])
        # Class 1 column if present
        if proba.shape[1] == 1:
            p1 = proba[:, 0]
        else:
            classes = list(clf.classes_)
            p1 = proba[:, classes.index(1)] if 1 in classes else proba[:, -1]
        pred = clf.predict(X[test_idx])
        held_out = groups[test_idx][0]
        y_te = y[test_idx]
        auc = roc_auc_score(y_te, p1) if len(set(y_te)) > 1 else float('nan')
        f1 = f1_score(y_te, pred, zero_division=0)
        rows.append(
            {
                'held_out_scene': held_out,
                'auc': auc,
                'f1': f1,
                'n_test': int(len(test_idx)),
                **{f'param_{k}': v for k, v in tree_params.items()},
            }
        )
    return pd.DataFrame(rows)


def validate_leave_one_scene_out(
    df,
    features=FEATURES,
    scene_col='scene_id',
    max_depth=3,
    min_samples_leaf=5,
    min_samples_split=2,
    criterion='gini',
    class_weight='balanced',
):
    """Print LOSO folds for one hyperparameter setting."""
    if scene_col not in df.columns:
        print(
            f"Aviso: no encuentro columna '{scene_col}'; "
            'usa load_data_from_folder() o agrega scene_id al manifest.'
        )
        return None

    tree_params = {
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'criterion': criterion,
        'class_weight': class_weight,
    }
    try:
        folds = loso_fold_metrics(df, tree_params, features=list(features), scene_col=scene_col)
    except ValueError as e:
        print(str(e))
        return None

    for _, r in folds.iterrows():
        print(
            f"[fold sin escena={r['held_out_scene']}] "
            f"AUC={r['auc']:.3f} F1={r['f1']:.3f} n_test={int(r['n_test'])}"
        )
    print(
        f"Mean AUC={folds['auc'].mean():.3f}±{folds['auc'].std(ddof=0):.3f}  "
        f"Mean F1={folds['f1'].mean():.3f}±{folds['f1'].std(ddof=0):.3f}  "
        f"(n_folds={len(folds)})"
    )
    return folds


def _expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def robustness_score(mean_metric: float, std_metric: float, *, lam: float = 1.0) -> float:
    """
    Higher is better: reward mean performance, penalize fold variance.

    robustness ≈ mean - λ * std   (NaN mean → -inf)
    """
    if pd.isna(mean_metric):
        return float('-inf')
    s = 0.0 if pd.isna(std_metric) else float(std_metric)
    return float(mean_metric) - lam * s


def tune_tree_hyperparams(
    df: pd.DataFrame,
    *,
    features: list[str] = FEATURES,
    scene_col: str = 'scene_id',
    param_grid: Optional[dict[str, list[Any]]] = None,
    primary_metric: str = 'f1',
    lambda_std: float = 1.0,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Grid-search DecisionTree hyperparameters; score each config with LOSO.

    Ranking uses ``robustness = mean(metric) - lambda_std * std(metric)`` so
    stable cross-scene performance beats a high but brittle mean.

    Returns
    -------
    results : DataFrame
        One row per hyperparameter combination (sorted best-first).
    best_params : dict
        Tree kwargs for the top row (without ranking columns).
    """
    if primary_metric not in ('f1', 'auc'):
        raise ValueError("primary_metric must be 'f1' or 'auc'")

    grid = param_grid or DEFAULT_PARAM_GRID
    combos = _expand_grid(grid)
    if verbose:
        print(
            f'Tuning {len(combos)} configs by LOSO '
            f'(primary={primary_metric}, lambda_std={lambda_std})…'
        )

    rows = []
    for i, params in enumerate(combos, start=1):
        folds = loso_fold_metrics(
            df, params, features=list(features), scene_col=scene_col, random_state=random_state
        )
        mean_auc = float(folds['auc'].mean())
        std_auc = float(folds['auc'].std(ddof=0))
        mean_f1 = float(folds['f1'].mean())
        std_f1 = float(folds['f1'].std(ddof=0))
        mean_m = mean_f1 if primary_metric == 'f1' else mean_auc
        std_m = std_f1 if primary_metric == 'f1' else std_auc
        rob = robustness_score(mean_m, std_m, lam=lambda_std)
        row = {
            **params,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'robustness': rob,
            'n_folds': len(folds),
            'min_auc': float(folds['auc'].min()),
            'min_f1': float(folds['f1'].min()),
        }
        rows.append(row)
        if verbose and (i <= 3 or i == len(combos) or i % 20 == 0):
            print(
                f'  [{i}/{len(combos)}] {params} → '
                f'mean_{primary_metric}={mean_m:.3f}±{std_m:.3f}  rob={rob:.3f}'
            )

    results = pd.DataFrame(rows).sort_values(
        ['robustness', f'mean_{primary_metric}', f'min_{primary_metric}'],
        ascending=False,
    ).reset_index(drop=True)

    best = results.iloc[0]
    best_params = {k: best[k] for k in grid.keys()}
    # JSON-friendly None
    if best_params.get('class_weight') is None or (
        isinstance(best_params.get('class_weight'), float)
        and pd.isna(best_params['class_weight'])
    ):
        best_params['class_weight'] = None

    if verbose:
        print('\n=== Best hyperparameters (LOSO robustness) ===')
        print(json.dumps(best_params, indent=2, default=str))
        print(
            f"mean_auc={best['mean_auc']:.3f}±{best['std_auc']:.3f}  "
            f"mean_f1={best['mean_f1']:.3f}±{best['std_f1']:.3f}  "
            f"robustness={best['robustness']:.3f}"
        )
    return results, best_params


def _save_fig(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close()


def _config_label(row: pd.Series) -> str:
    cw = row.get('class_weight', 'None')
    if cw is None or (isinstance(cw, float) and pd.isna(cw)):
        cw = 'None'
    return (
        f"d={int(row['max_depth'])} "
        f"leaf={int(row['min_samples_leaf'])} "
        f"split={int(row['min_samples_split'])} "
        f"{row['criterion']}/{cw}"
    )


def plot_tune_results(
    results: pd.DataFrame,
    output_dir: str | Path,
    *,
    df: Optional[pd.DataFrame] = None,
    best_params: Optional[dict[str, Any]] = None,
    features: list[str] = FEATURES,
    primary_metric: str = 'f1',
    top_k: int = 15,
) -> list[Path]:
    """
    Graphics for DecisionTree hyperparameter / LOSO robustness analysis.

    Writes PNGs under ``output_dir``:
      - top_k_robustness.png
      - mean_vs_std_f1.png / mean_vs_std_auc.png
      - heatmap_depth_x_leaf_*.png
      - best_loso_per_scene.png (if df + best_params given)
      - metric_by_max_depth.png
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if results is None or results.empty:
        return paths

    res = results.copy()
    # Normalize class_weight for plotting
    if 'class_weight' in res.columns:
        res['class_weight'] = res['class_weight'].apply(
            lambda x: 'None' if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
        )

    # --- 1) Top-K robustness bars ---
    top = res.head(min(top_k, len(res))).iloc[::-1]  # best at top after barh
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.35 * len(top))))
    labels = [_config_label(r) for _, r in top.iterrows()]
    ax.barh(range(len(top)), top['robustness'], color='#009384')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f'robustness = mean({primary_metric}) − λ·std')
    ax.set_title(f'Top {len(top)} DecisionTree configs (LOSO robustness)')
    ax.grid(True, axis='x', alpha=0.3)
    p = out / 'top_k_robustness.png'
    _save_fig(p)
    paths.append(p)

    # --- 2) Mean vs std scatter (stability frontier) ---
    for metric in ('f1', 'auc'):
        mean_c, std_c = f'mean_{metric}', f'std_{metric}'
        if mean_c not in res.columns or std_c not in res.columns:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sc = ax.scatter(
            res[std_c],
            res[mean_c],
            c=res['robustness'],
            cmap='viridis',
            s=36,
            alpha=0.85,
        )
        # Mark best
        best_row = res.iloc[0]
        ax.scatter(
            [best_row[std_c]],
            [best_row[mean_c]],
            s=120,
            facecolors='none',
            edgecolors='#B85741',
            linewidths=2,
            label='best robustness',
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('robustness')
        ax.set_xlabel(f'std {metric} (across held-out scenes)')
        ax.set_ylabel(f'mean {metric} (LOSO)')
        ax.set_title(f'Stability frontier: mean vs std ({metric})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        p = out / f'mean_vs_std_{metric}.png'
        _save_fig(p)
        paths.append(p)

    # --- 3) Heatmaps: max_depth × min_samples_leaf (best robustness in cell) ---
    if {'max_depth', 'min_samples_leaf'}.issubset(res.columns):
        for value_col, title in (
            ('robustness', 'Best robustness'),
            ('mean_f1', 'Best mean F1'),
            ('mean_auc', 'Best mean AUC'),
        ):
            if value_col not in res.columns:
                continue
            pivot = (
                res.groupby(['max_depth', 'min_samples_leaf'])[value_col]
                .max()
                .unstack('min_samples_leaf')
                .sort_index()
            )
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)
            ax.set_title(f'{title} over max_depth × min_samples_leaf')
            ax.set_xlabel('min_samples_leaf')
            ax.set_ylabel('max_depth')
            p = out / f'heatmap_depth_x_leaf_{value_col}.png'
            _save_fig(p)
            paths.append(p)

    # --- 4) Metric by max_depth (distribution across other hyperparameters) ---
    if 'max_depth' in res.columns and 'mean_f1' in res.columns:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, col, title in (
            (axes[0], 'mean_f1', 'mean F1 by max_depth'),
            (axes[1], 'robustness', 'robustness by max_depth'),
        ):
            sns.boxplot(data=res, x='max_depth', y=col, ax=ax, color='#D9C4B1')
            ax.set_title(title)
            ax.grid(True, axis='y', alpha=0.3)
        p = out / 'metric_by_max_depth.png'
        _save_fig(p)
        paths.append(p)

    # --- 5) Best config: per-scene LOSO bars ---
    if df is not None and best_params is not None and len(df) > 0:
        try:
            folds = loso_fold_metrics(df, best_params, features=list(features))
        except ValueError:
            folds = None
        if folds is not None and not folds.empty:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            scenes = folds['held_out_scene'].astype(str)
            axes[0].bar(scenes, folds['f1'], color='#009384')
            axes[0].axhline(folds['f1'].mean(), color='#B85741', ls='--', label='mean')
            axes[0].set_title('Best config — F1 per held-out scene')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].set_ylim(0, 1.05)
            axes[0].legend(fontsize=8)
            axes[0].grid(True, axis='y', alpha=0.3)

            axes[1].bar(scenes, folds['auc'], color='#002F4A')
            axes[1].axhline(folds['auc'].mean(), color='#B85741', ls='--', label='mean')
            axes[1].set_title('Best config — AUC per held-out scene')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].set_ylim(0, 1.05)
            axes[1].legend(fontsize=8)
            axes[1].grid(True, axis='y', alpha=0.3)
            p = out / 'best_loso_per_scene.png'
            _save_fig(p)
            paths.append(p)
            folds.to_csv(out / 'best_loso_folds.csv', index=False)

    return paths


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Fit / tune DecisionTree thresholds with multi-scene LOSO robustness.'
    )
    p.add_argument(
        '--folder',
        type=str,
        default=None,
        help='Folder with per-scene *_calibration_manifest.csv + *_labels.json',
    )
    p.add_argument(
        'manifest_csv',
        nargs='?',
        default=None,
        help='Legacy: single-scene calibration_manifest.csv',
    )
    p.add_argument(
        'labels_json',
        nargs='?',
        default=None,
        help='Legacy: single-scene labels.json',
    )
    p.add_argument('--max_depth', type=int, default=3, help='Used when not --tune')
    p.add_argument('--min_samples_leaf', type=int, default=10, help='Used when not --tune')
    p.add_argument(
        '--tune',
        action='store_true',
        help='Grid-search tree hyperparameters scored by LOSO robustness',
    )
    p.add_argument(
        '--tune_out',
        type=str,
        default=None,
        help='Directory for tune_results.csv / best_params.json (default: cwd)',
    )
    p.add_argument(
        '--primary_metric',
        choices=('f1', 'auc'),
        default='f1',
        help='Metric whose mean−λ·std defines robustness ranking',
    )
    p.add_argument(
        '--lambda_std',
        type=float,
        default=1.0,
        help='Penalty on cross-scene std in robustness score (default 1.0)',
    )
    p.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip writing tune graphics under --tune_out',
    )
    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    if args.folder:
        print(f'Loading scenes from folder: {args.folder}')
        df = load_data_from_folder(args.folder)
    elif args.manifest_csv and args.labels_json:
        df = load_data(args.manifest_csv, args.labels_json)
    else:
        raise SystemExit(
            'Provide --folder DIR  or  MANIFEST.csv LABELS.json\n'
            'Example:\n'
            '  python -m cm_benchmark.utils.fit_thresholds '
            '--folder path/to/calibration_scenes --tune'
        )

    if 'y' in df.columns and (df['y'] == 2).any():
        n_amb = int((df['y'] == 2).sum())
        print(f'Excluding {n_amb} ambiguous (y=2) items from tree fit / LOSO')
        df = df[df['y'] != 2].copy()

    print(
        f'Labeled sample: {len(df)} items, '
        f"{df['y'].mean():.1%} distinguishable, "
        f"{df['scene_id'].nunique()} scene(s)\n"
    )

    tree_kwargs: dict[str, Any] = {
        'max_depth': args.max_depth,
        'min_samples_leaf': args.min_samples_leaf,
        'min_samples_split': 2,
        'criterion': 'gini',
        'class_weight': 'balanced',
    }

    if args.tune:
        print('=== Hyperparameter search (LOSO robustness) ===')
        results, best = tune_tree_hyperparams(
            df,
            primary_metric=args.primary_metric,
            lambda_std=args.lambda_std,
        )
        out_dir = Path(args.tune_out) if args.tune_out else Path('.')
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / 'dt_tune_results.csv'
        best_path = out_dir / 'dt_best_params.json'
        results.to_csv(results_path, index=False)
        best_path.write_text(json.dumps(best, indent=2, default=str))
        print(f'Wrote {results_path}')
        print(f'Wrote {best_path}')
        tree_kwargs.update(best)

        print('\n=== Top 10 configs by robustness ===')
        cols = [
            'max_depth',
            'min_samples_leaf',
            'min_samples_split',
            'criterion',
            'class_weight',
            'mean_f1',
            'std_f1',
            'mean_auc',
            'std_auc',
            'robustness',
            'min_f1',
        ]
        print(results[cols].head(10).to_string(index=False))

        if not args.no_plots:
            plots_dir = out_dir / 'plots'
            print(f'\n=== Writing tune graphics → {plots_dir} ===')
            paths = plot_tune_results(
                results,
                plots_dir,
                df=df,
                best_params=best,
                primary_metric=args.primary_metric,
            )
            for p in paths:
                print(f'  {p.name}')

    print('\n=== Decision tree (rules / thresholds) ===')
    fit_tree(df, **tree_kwargs)

    print('\n=== Leave-one-scene-out validation ===')
    validate_leave_one_scene_out(df, **tree_kwargs)


if __name__ == '__main__':
    main()
