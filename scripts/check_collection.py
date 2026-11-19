"""Check a SPOC collection before the full export. Exit code 1 if any check fails.

Run it on 2-3 houses collected with the current collector:

    PYTHONPATH=src python scripts/check_collection.py \
        --collection /path/to/RoundN/nav \
        --reference-items /path/to/Round7/items \
        --visibility_model_path analysis/dt_tune/visibility_filter.joblib

Hard checks (fail):
  1. every navigation row has a pose, an action and step-source;
  2. in-frame-fraction is set on every row with mask pixels > 0;
  3. glance rows match episode_meta.glances (rows = turns out + turns back) and
     their turns cancel out;
  4. timeouts are at most --max-timeout-share of the episodes;
  5. cm-benchmark generates items: no construct gives 0 items in a scene where
     the reference items had some.

Reported only (not failures, they are expected for unseen objects):
  - placement anchors found in world_layout or objects CSV;
  - placement objects ever seen (have an object_state track).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

TURN = {'rotate_right': 1, 'rotate_left': -1}
POSE_COLS = ('ag-pos-x', 'ag-pos-y', 'ag-pos-z', 'ag-rot-y')


def _one(pattern: str):
    hits = glob.glob(pattern)
    return hits[0] if hits else None


def check_episode(ep_dir: Path) -> tuple[str, list[str], dict]:
    """``(scene, failures, stats)`` for one collected episode folder."""
    ann = ep_dir / 'annotations'
    meta = json.load(open(_one(str(ann / 'episode_meta*.json'))))
    scene = meta.get('scene_id') or ep_dir.name
    nav = pd.read_csv(_one(str(ann / 'navigation-*.csv')))
    fails: list[str] = []

    # 1. pose, action, step-source on every row
    for col in (*POSE_COLS, 'ag-action', 'step-source'):
        if col not in nav.columns:
            fails.append(f'navigation has no {col} column')
        elif nav[col].isna().any():
            fails.append(f'{int(nav[col].isna().sum())} rows without {col}')

    # 2. in-frame-fraction where the object has mask pixels
    if 'in-frame-fraction' not in nav.columns:
        fails.append('navigation has no in-frame-fraction column')
    else:
        seen = nav['visible-pixels'].fillna(0) > 0
        missing = int(nav.loc[seen, 'in-frame-fraction'].isna().sum())
        if missing:
            fails.append(f'{missing} rows with mask pixels but no in-frame-fraction')

    # 3. glances: rows = |turns| out + back, and they cancel out
    steps = nav.drop_duplicates('timestep')
    if 'step-source' in steps.columns:
        glance = steps[steps['step-source'] == 'glance']
    else:
        glance = steps[steps['scripted-glance'] == True]  # noqa: E712
    gcfg = (meta.get('collection_config') or {}).get('glance') or {}
    per_glance = 2 if gcfg.get('rotate_back', True) else 1
    expected = per_glance * sum(abs(int(g.get('turns') or 0)) for g in meta.get('glances') or [])
    if not meta.get('truncated'):
        if len(glance) != expected:
            fails.append(f'{len(glance)} glance rows, episode_meta.glances expects {expected}')
        net = sum(TURN.get(a, 0) for a in glance['ag-action'])
        if per_glance == 2 and net != 0:
            fails.append(f'glance turns do not cancel out (net {net} turns)')

    # Reported: placement coverage
    stats: dict = {'save_reason': meta.get('save_reason'), 'done_overrides': meta.get('done_overrides')}
    if 'step-source' in steps.columns:
        stats['done_override_rows'] = int((steps['step-source'] == 'done_override').sum())
    pc = _one(str(ann / 'placement_candidates-*.csv'))
    if pc:
        p = pd.read_csv(pc)
        layout = json.load(open(_one(str(ann / 'world_layout*.json'))))
        landmarks = {e.get('landmark_id') for e in layout.get('landmarks') or []}
        objs = set(pd.read_csv(_one(str(ann / 'objects-*.csv')))['obj-id'])
        state = pd.read_csv(_one(str(ann / 'object_state-*.csv')))
        state_col = 'obj_id' if 'obj_id' in state.columns else 'obj-id'
        tracked = set(state[state_col])
        anchors = p['anchor-id'].dropna()
        movers = p['obj-id'].dropna().unique()
        stats['anchors_known'] = f'{int(anchors.isin(landmarks | objs).sum())}/{len(anchors)}'
        stats['objects_seen'] = f'{sum(o in tracked for o in movers)}/{len(movers)}'
    return scene, fails, stats


def _counts(items_root: Path, scenes: set[str]) -> dict[str, Counter]:
    """Concise ok items per scene and construct."""
    out: dict[str, Counter] = {}
    for f in glob.glob(str(items_root / '*' / 'items_*.json')):
        scene = Path(f).parent.name
        if scene not in scenes:
            continue
        data = json.load(open(f))
        rows = data['items'] if isinstance(data, dict) else data
        out[scene] = Counter(
            r['construct']
            for r in rows
            if r.get('status') == 'ok' and r.get('question_style') == 'concise'
        )
    return out


def run_generator(collection: Path, out: Path, model: str | None, config: str | None) -> None:
    """Stage 1 + stage 2 into ``out`` (episode GT under nav_data, items under items)."""
    env = dict(os.environ)
    env['PYTHONPATH'] = 'src' + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    model_args = ['--visibility_model_path', model] if model else []
    subprocess.run(
        [sys.executable, '-m', 'cm_benchmark.generator.ai2thor_nav_generator',
         '--csv_path_folder', str(collection), '--export_json',
         '--output_path', str(out / 'nav_data'), *model_args],
        check=True, env=env, stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [sys.executable, '-m', 'cm_benchmark.generation.generate_items',
         '--episode_json', str(out / 'nav_data'), '--output_path', str(out / 'items'),
         *(['--config', config] if config else []), *model_args],
        check=True, env=env, stdout=subprocess.DEVNULL,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--collection', required=True, help='folder of collected episode folders')
    ap.add_argument('--reference-items', help='items folder to compare against (e.g. Round 7)')
    ap.add_argument('--visibility_model_path')
    ap.add_argument('--config', default='configs/generation_max.yaml')
    ap.add_argument('--max-timeout-share', type=float, default=0.2)
    ap.add_argument('--skip-generation', action='store_true')
    args = ap.parse_args()

    collection = Path(args.collection)
    episodes = sorted(p for p in collection.iterdir() if (p / 'annotations').is_dir())
    if not episodes:
        print(f'no episode folders under {collection}')
        return 1

    failed = False
    reasons = Counter()
    scenes: set[str] = set()
    for ep in episodes:
        scene, fails, stats = check_episode(ep)
        scenes.add(scene)
        reasons[stats.get('save_reason')] += 1
        status = 'FAIL' if fails else 'ok'
        print(f'{scene:16s} {status:4s} ' + ' '.join(f'{k}={v}' for k, v in stats.items()))
        for f in fails:
            print(f'    - {f}')
        failed |= bool(fails)

    # 4. timeouts are rare
    share = reasons.get('timeout', 0) / len(episodes)
    print(f'save_reason: {dict(reasons)}')
    if share > args.max_timeout_share:
        print(f'FAIL timeouts are {share:.0%} of episodes (max {args.max_timeout_share:.0%})')
        failed = True

    # 5. cm-benchmark generates items where the reference had some
    if not args.skip_generation:
        with tempfile.TemporaryDirectory() as tmp:
            run_generator(collection, Path(tmp), args.visibility_model_path, args.config)
            new = _counts(Path(tmp) / 'items', scenes)
        ref = _counts(Path(args.reference_items), scenes) if args.reference_items else {}
        constructs = sorted({c for d in (*new.values(), *ref.values()) for c in d})
        print(f"{'scene':16s} " + ' '.join(c[:12].rjust(12) for c in constructs))
        for scene in sorted(scenes):
            n, r = new.get(scene, Counter()), ref.get(scene, Counter())
            cells = []
            for c in constructs:
                cell = f'{n[c]}' + (f'/{r[c]}' if scene in ref else '')
                if scene in ref and r[c] > 0 and n[c] == 0:
                    cell += '!'
                    failed = True
                cells.append(cell.rjust(12))
            print(f'{scene:16s} ' + ' '.join(cells))
        if ref:
            print('cells are new/reference; ! = 0 items where the reference had some')

    print('FAILED' if failed else 'PASSED')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
