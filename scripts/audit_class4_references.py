#!/usr/bin/env python3
"""Score every class-4 reference answer; report success, nDTW, and frame stats.

Usage:
  python scripts/audit_class4_references.py --items-root <dir>
  python scripts/audit_class4_references.py --items-root <dir> --json-out report.json

Expects frozen items (FREEZE). Does not regenerate. Success rate must be 100%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cm_benchmark.evaluation.score import (
    score_route_knowledge,
    score_survey_based_route_planning,
)
from cm_benchmark.generation.nav_graph import parse_nav_actions

NAV = frozenset({'route_knowledge', 'survey_based_route_planning'})


def load_items(root: Path) -> list[dict]:
    """Every item dict under ``root`` (a folder or one JSON file)."""
    paths = sorted(root.rglob('items_*.json')) if root.is_dir() else [root]
    items: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        blob = json.loads(path.read_text())
        rows = blob if isinstance(blob, list) else blob.get('items') or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item_id = str(row.get('item_id') or '')
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            items.append(row)
    return items


def _ctx(item: dict) -> dict:
    return item.get('context') or item.get('extra') or {}


def _actions(item: dict) -> list[str] | None:
    ctx = _ctx(item)
    seq = ctx.get('action_sequence')
    if seq:
        return [str(a) for a in seq]
    answer = item.get('answer')
    if answer is None:
        return None
    return parse_nav_actions(str(answer))


def audit_item(item: dict) -> dict:
    construct = item.get('construct') or ''
    actions = _actions(item)
    ctx = _ctx(item)
    shown = list(ctx.get('shown_frame_steps') or [])
    row = {
        'item_id': item.get('item_id'),
        'construct': construct,
        'shown_frame_steps': shown,
        'max_shown_gap_steps': ctx.get('max_shown_gap_steps'),
        'turns_covered': ctx.get('turns_covered'),
        'success': False,
        'ndtw': None,
        'sdtw': None,
        'efficiency': None,
        'validity': None,
        'error': None,
    }
    if actions is None:
        row['error'] = 'no_actions'
        return row
    if construct == 'route_knowledge':
        scored = score_route_knowledge(item, None, actions)
    elif construct == 'survey_based_route_planning':
        scored = score_survey_based_route_planning(item, None, actions)
    else:
        row['error'] = 'not_class4'
        return row
    row['success'] = bool(scored.get('success'))
    row['ndtw'] = float(scored.get('ndtw') or 0.0)
    row['sdtw'] = float(scored.get('sdtw') or 0.0)
    row['efficiency'] = float(scored.get('efficiency') or 0.0)
    row['validity'] = bool(scored.get('validity'))
    return row


def summarise(rows: list[dict]) -> dict:
    class4 = [r for r in rows if r.get('construct') in NAV and r.get('error') is None]
    route = [r for r in class4 if r['construct'] == 'route_knowledge']
    survey = [r for r in class4 if r['construct'] == 'survey_based_route_planning']
    n = len(class4)
    n_ok = sum(1 for r in class4 if r['success'])
    mean_ndtw = (
        sum(float(r['ndtw'] or 0.0) for r in route) / len(route) if route else None
    )
    return {
        'n_items': n,
        'n_route': len(route),
        'n_survey': len(survey),
        'success_rate': (n_ok / n) if n else None,
        'mean_ndtw_route': mean_ndtw,
        'rows': class4,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--items-root', type=Path, required=True)
    parser.add_argument('--json-out', type=Path, default=None)
    args = parser.parse_args(argv)
    items = [
        item
        for item in load_items_from_root(args.items_root)
        if item.get('construct') in NAV
    ]
    rows = [audit_item(item) for item in items]
    report = summarise(rows)
    rate = report['success_rate']
    print(
        f"class-4 references: {report['n_items']} "
        f"(route={report['n_route']}, survey={report['n_survey']})"
    )
    print(f"success rate: {rate if rate is not None else 'n/a'}")
    if report['mean_ndtw_route'] is not None:
        print(f"mean nDTW (route): {report['mean_ndtw_route']:.4f}")
    for row in report['rows']:
        print(
            f"  {row['item_id']}: success={row['success']} "
            f"ndtw={row['ndtw']:.4f} gap={row['max_shown_gap_steps']} "
            f"turns_covered={row['turns_covered']} "
            f"frames={row['shown_frame_steps']}"
        )
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if report['n_items'] == 0:
        return 0
    if rate is None or rate < 1.0 - 1e-12:
        print('FAIL: reference success rate must be 100%', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
