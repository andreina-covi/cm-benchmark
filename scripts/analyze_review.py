#!/usr/bin/env python3
"""Merge reviewer CSVs and report agreement against the package manifest.

Per construct (study items only): human accuracy, nominal Krippendorff's
alpha, and the share of reviews with any problem flag. Check-item accuracy
is reported per reviewer. Items that both reviewers flagged, or that both
answered wrongly, are listed.

Practice rows are ignored. Gold answers come from ``manifest.json`` written
by ``build_review_package.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

CONSTRUCTS = (
    'egocentric_encoding',
    'allocentric_encoding',
    'spatial_working_memory',
    'invisible_displacement',
    'spatial_updating',
    'perspective_taking',
    'route_knowledge',
    'survey_based_route_planning',
)


def krippendorff_alpha(units: list[list[str]]) -> float | None:
    """Nominal Krippendorff's alpha. Each unit is one item's rater labels."""
    values = [[v for v in unit if v is not None and v != ''] for unit in units]
    values = [unit for unit in values if len(unit) >= 2]
    if len(values) < 2:
        return None
    levels = sorted({v for unit in values for v in unit})
    if len(levels) == 1:
        return 1.0
    index = {level: i for i, level in enumerate(levels)}
    q = len(levels)
    coinc = [[0.0] * q for _ in range(q)]
    n = 0
    for unit in values:
        m = len(unit)
        n += m
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                coinc[index[unit[i]]][index[unit[j]]] += 1.0 / (m - 1)
    if n < 2:
        return None
    diag = sum(coinc[c][c] for c in range(q))
    do = 1.0 - diag / n
    nc = [sum(coinc[c]) for c in range(q)]
    expected_agreement = sum(nc[c] * (nc[c] - 1) for c in range(q)) / (n * (n - 1))
    de = 1.0 - expected_agreement
    if de == 0:
        return 1.0 if do == 0 else None
    return 1.0 - do / de


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    if 'items' not in data:
        raise ValueError(f'{path} has no items table')
    return data


def _parse_flags(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in str(text).replace(',', '|').split('|') if part.strip()]


def load_rows(paths: list[Path]) -> list[dict]:
    """CSV rows. A later file overrides the same reviewer and item."""
    merged: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for path in paths:
        with path.open(newline='') as handle:
            for raw in csv.DictReader(handle):
                reviewer = (raw.get('reviewer_id') or '').strip()
                item_id = (raw.get('item_id') or '').strip()
                if not reviewer or not item_id:
                    continue
                key = (reviewer, item_id)
                if key not in merged:
                    order.append(key)
                merged[key] = {
                    'reviewer_id': reviewer,
                    'item_id': item_id,
                    'construct': (raw.get('construct') or '').strip(),
                    'answer': (raw.get('answer') or '').strip(),
                    'flags': _parse_flags(raw.get('flags') or ''),
                    'comment': raw.get('comment') or '',
                    'seconds': (raw.get('seconds') or '').strip(),
                    'role': (raw.get('role') or '').strip(),
                }
    return [merged[key] for key in order]


def _role(row: dict, manifest: dict) -> str:
    meta = (manifest.get('items') or {}).get(row['item_id']) or {}
    return meta.get('role') or row.get('role') or 'review'


def _correct(meta: dict, answer: str) -> bool:
    gold = (meta.get('gold') or '').strip()
    given = (answer or '').strip()
    if not gold or not given:
        return False
    return given.casefold() == gold.casefold()


def analyze(manifest: dict, rows: list[dict]) -> str:
    """Text report for one merged set of reviews."""
    items = manifest.get('items') or {}
    checks = list(manifest.get('check_item_ids') or [])
    by_key = {(row['reviewer_id'], row['item_id']): row for row in rows}

    study_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if _role(row, manifest) != 'review':
            continue
        meta = items.get(row['item_id'])
        if not meta:
            continue
        study_rows[meta.get('construct') or row['construct']].append(row)

    lines = [
        f'{"construct":<32} {"reviews":>8} {"accuracy":>16} {"alpha":>8} {"flag_rate":>10}',
    ]
    for construct in CONSTRUCTS:
        group = study_rows.get(construct) or []
        if not group and not any(
            meta.get('role') == 'review' and meta.get('construct') == construct
            for meta in items.values()
        ):
            continue
        answered = [row for row in group if row['answer']]
        n_correct = sum(1 for row in answered if _correct(items[row['item_id']], row['answer']))
        if answered:
            accuracy = f'{n_correct / len(answered):.3f} ({n_correct}/{len(answered)})'
        else:
            accuracy = 'n/a'
        by_item: dict[str, list[str]] = defaultdict(list)
        for row in answered:
            by_item[row['item_id']].append(row['answer'])
        alpha = krippendorff_alpha(list(by_item.values()))
        alpha_s = 'n/a' if alpha is None else f'{alpha:.3f}'
        flagged = sum(1 for row in group if row['flags'])
        flag_rate = f'{flagged / len(group):.3f}' if group else 'n/a'
        lines.append(
            f'{construct:<32} {len(group):>8} {accuracy:>16} {alpha_s:>8} {flag_rate:>10}'
        )

    lines.append('')
    lines.append('check-item accuracy')
    reviewers = sorted({row['reviewer_id'] for row in rows})
    if not checks:
        lines.append('  no check items')
    elif not reviewers:
        lines.append('  no reviews')
    else:
        width = max(len(name) for name in reviewers)
        for reviewer in reviewers:
            hits = 0
            for item_id in checks:
                row = by_key.get((reviewer, item_id))
                meta = items.get(item_id) or {}
                if row and _correct(meta, row['answer']):
                    hits += 1
            lines.append(f'  {reviewer:<{width}}  {hits}/{len(checks)}')

    lines.append('')
    lines.append('flagged or wrong by both reviewers')
    problems: list[tuple[str, str, bool, bool]] = []
    by_item_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if _role(row, manifest) != 'review':
            continue
        if row['item_id'] in items:
            by_item_rows[row['item_id']].append(row)
    for item_id in sorted(by_item_rows):
        group = by_item_rows[item_id]
        if len(group) < 2:
            continue
        meta = items[item_id]
        both_flagged = all(row['flags'] for row in group)
        answered = [row for row in group if row['answer']]
        both_wrong = len(answered) >= 2 and all(
            not _correct(meta, row['answer']) for row in answered
        ) and len(answered) == len(group)
        if both_flagged or both_wrong:
            problems.append((item_id, meta.get('construct') or '', both_flagged, both_wrong))
    if not problems:
        lines.append('  none')
    else:
        lines.append(f'  {"item_id":<44} {"construct":<32} {"both_flagged":>13} {"both_wrong":>11}')
        for item_id, construct, flagged, wrong in problems:
            lines.append(
                f'  {item_id:<44} {construct:<32} {str(flagged):>13} {str(wrong):>11}'
            )
    return '\n'.join(lines)


def _csv_paths(package: Path, extra: list[Path]) -> list[Path]:
    paths = list(extra)
    if not paths:
        paths = sorted(package.rglob('*.csv'))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True, help='Folder that contains manifest.json')
    parser.add_argument('csvs', nargs='*', type=Path, help='Reviewer CSVs. Default: every CSV under --package')
    args = parser.parse_args(argv)
    manifest_path = args.package / 'manifest.json' if args.package.is_dir() else args.package
    try:
        manifest = load_manifest(manifest_path)
        paths = _csv_paths(manifest_path.parent, list(args.csvs))
        if not paths:
            print('no CSV files', file=sys.stderr)
            return 1
        text = analyze(manifest, load_rows(paths))
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
