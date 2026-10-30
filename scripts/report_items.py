#!/usr/bin/env python3
"""Summarize how many questions ``generate_items`` produced.

Reads ``items_*.json`` under a generation ``--output_path`` (or one file).
Counts concise facts only (a verbose partner is not counted again).

Prints:
  - grand totals (ok / thin / unsupported)
  - per-scene counts
  - per-construct counts
  - answer-label distributions for concise ok items

``--json-out`` writes structured data for plots/tables.
``-o`` / ``--txt-out`` saves the printed text.
``--tables-dir`` writes CSV tables (totals, by scene, by construct, labels).
Plot with ``scripts/plot_review_report.py <json> -o <plots_dir>``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load(root: Path) -> list[dict]:
    """Every item dict under ``root``."""
    paths = sorted(root.rglob('items_*.json')) if root.is_dir() else [root]
    items: list[dict] = []
    for path in paths:
        blob = json.loads(path.read_text())
        rows = blob if isinstance(blob, list) else blob.get('items') or []
        scene_from_file = (
            path.stem.removeprefix('items_') if path.stem.startswith('items_') else ''
        )
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not row.get('scene_id') and scene_from_file:
                row = {**row, 'scene_id': scene_from_file}
            items.append(row)
    return items


def _concise(items: list[dict]) -> list[dict]:
    return [item for item in items if item.get('question_style') != 'verbose']


def _label_and_letter(item: dict) -> tuple[str, str | None]:
    options = item.get('options') or {}
    answer = item.get('answer')
    if isinstance(options, dict) and answer in options:
        return str(options[answer]), str(answer)
    if answer is None:
        return '', None
    return str(answer), None


def _status_counts(rows: list[dict]) -> Counter:
    c: Counter = Counter()
    for item in rows:
        status = item.get('status') or 'unknown'
        c[status] += 1
    return c


def collect_items_summary(
    items: list[dict],
    *,
    items_dir: Path | None = None,
) -> dict:
    """Structured generation summary (JSON-serializable)."""
    rows = _concise(items)
    totals = _status_counts(rows)

    by_scene: dict[str, Counter] = defaultdict(Counter)
    by_construct: dict[str, Counter] = defaultdict(Counter)
    scenes_ok: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, Counter] = defaultdict(Counter)
    letters: dict[str, Counter] = defaultdict(Counter)
    freeform: dict[str, Counter] = defaultdict(Counter)
    constructs: list[str] = []

    for item in rows:
        construct = str(item.get('construct') or '?')
        scene = str(item.get('scene_id') or '?')
        status = str(item.get('status') or 'unknown')
        by_scene[scene][status] += 1
        by_construct[construct][status] += 1
        if construct not in constructs:
            constructs.append(construct)
        if status == 'ok':
            scenes_ok[construct].add(scene)
            label, letter = _label_and_letter(item)
            if letter:
                labels[construct][label] += 1
                letters[construct][letter] += 1
            else:
                freeform[construct][label] += 1

    for construct in sorted(by_construct):
        if construct not in constructs:
            constructs.append(construct)

    other = sum(v for k, v in totals.items() if k not in {'ok', 'thin', 'unsupported'})
    scene_rows = []
    for scene in sorted(by_scene):
        c = by_scene[scene]
        scene_rows.append(
            {
                'scene': scene,
                'ok': c.get('ok', 0),
                'thin': c.get('thin', 0),
                'unsupported': c.get('unsupported', 0),
                'all': sum(c.values()),
            }
        )
    construct_rows = []
    for construct in constructs:
        c = by_construct[construct]
        construct_rows.append(
            {
                'construct': construct,
                'ok': c.get('ok', 0),
                'thin': c.get('thin', 0),
                'unsupported': c.get('unsupported', 0),
                'n_scenes_ok': len(scenes_ok[construct]),
            }
        )
    label_rows = []
    for construct in constructs:
        total = sum(labels[construct].values())
        for label, count in labels[construct].most_common():
            label_rows.append(
                {
                    'construct': construct,
                    'kind': 'mc_label',
                    'label': label,
                    'count': count,
                    'share': (count / total) if total else None,
                }
            )
        letter_total = sum(letters[construct].values())
        for letter, count in sorted(letters[construct].items()):
            label_rows.append(
                {
                    'construct': construct,
                    'kind': 'mc_letter',
                    'label': letter,
                    'count': count,
                    'share': (count / letter_total) if letter_total else None,
                }
            )
        n_actions = sum(freeform[construct].values())
        if n_actions:
            label_rows.append(
                {
                    'construct': construct,
                    'kind': 'action_sequences',
                    'label': 'action_sequence',
                    'count': n_actions,
                    'share': 1.0,
                    'n_distinct': len(freeform[construct]),
                }
            )

    return {
        'kind': 'items',
        'items_dir': str(items_dir) if items_dir is not None else None,
        'n_scenes': len(by_scene),
        'totals': {
            'ok': totals.get('ok', 0),
            'thin': totals.get('thin', 0),
            'unsupported': totals.get('unsupported', 0),
            'other': other,
            'all': len(rows),
        },
        'by_scene': scene_rows,
        'by_construct': construct_rows,
        'labels': label_rows,
    }


def format_items_summary(data: dict) -> str:
    """Text tables for a ``collect_items_summary`` payload."""
    lines: list[str] = []
    items_dir = data.get('items_dir')
    if items_dir:
        lines.append(f'Items root: {items_dir}')
    lines.append(f'Scenes with items: {int(data.get("n_scenes") or 0)}')
    lines.append('')
    totals = data.get('totals') or {}
    lines.append('TOTAL (concise)')
    for key in ('ok', 'thin', 'unsupported'):
        lines.append(f'  {key:<12} {int(totals.get(key, 0)):>6}')
    other = int(totals.get('other') or 0)
    if other:
        lines.append(f'  {"other":<12} {other:>6}')
    lines.append(f'  {"all":<12} {int(totals.get("all", 0)):>6}')
    lines.append('')

    lines.append('By scene (concise)')
    lines.append(f'{"scene":<28} {"ok":>5} {"thin":>5} {"unsupported":>12} {"all":>5}')
    scene_ok = scene_thin = scene_un = scene_all = 0
    for row in data.get('by_scene') or []:
        ok = int(row['ok'])
        thin = int(row['thin'])
        un = int(row['unsupported'])
        n = int(row['all'])
        scene_ok += ok
        scene_thin += thin
        scene_un += un
        scene_all += n
        lines.append(f'{row["scene"]:<28} {ok:>5} {thin:>5} {un:>12} {n:>5}')
    lines.append(
        f'{"TOTAL":<28} {scene_ok:>5} {scene_thin:>5} {scene_un:>12} {scene_all:>5}'
    )
    lines.append('')

    lines.append('By construct (concise)')
    lines.append(
        f'{"construct":<32} {"ok":>5} {"thin":>5} {"unsupported":>12} {"scenes":>7}'
    )
    for row in data.get('by_construct') or []:
        lines.append(
            f'{row["construct"]:<32} {int(row["ok"]):>5} {int(row["thin"]):>5} '
            f'{int(row["unsupported"]):>12} {int(row["n_scenes_ok"]):>7}'
        )
    lines.append('')

    lines.append('Labels and letters (concise ok)')
    by_construct: dict[str, list[dict]] = defaultdict(list)
    for row in data.get('labels') or []:
        by_construct[row['construct']].append(row)
    if not by_construct:
        lines.append('  (none)')
    else:
        for construct, rows in by_construct.items():
            lines.append(construct)
            mc_labels = [r for r in rows if r.get('kind') == 'mc_label']
            letters = [r for r in rows if r.get('kind') == 'mc_letter']
            actions = [r for r in rows if r.get('kind') == 'action_sequences']
            for row in mc_labels:
                share = row.get('share')
                share_s = f'{float(share):>5.0%}' if share is not None else '  n/a'
                lines.append(f'  {row["label"]:<20} {int(row["count"]):>4}  {share_s}')
            if letters:
                letter_bits = '  '.join(
                    f'{row["label"]} {int(row["count"])}'
                    for row in letters
                    if row['label'] in {'A', 'B', 'C', 'D'}
                )
                extra = [
                    f'{row["label"]} {int(row["count"])}'
                    for row in letters
                    if row['label'] not in {'A', 'B', 'C', 'D'}
                ]
                if extra:
                    letter_bits = (letter_bits + '  ' + '  '.join(extra)).strip()
                if letter_bits:
                    lines.append(f'  letters  {letter_bits}')
            for row in actions:
                distinct = row.get('n_distinct')
                extra = f' ({distinct} distinct)' if distinct is not None else ''
                lines.append(f'  action sequences: {int(row["count"])}{extra}')
    return '\n'.join(lines) + '\n'


def report(items: list[dict], *, items_dir: Path | None = None) -> str:
    """Text summary of concise facts: totals, by scene, by construct, labels."""
    return format_items_summary(collect_items_summary(items, items_dir=items_dir))


def write_tables(data: dict, tables_dir: Path) -> list[Path]:
    """Write CSV tables used in a generation report."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    totals_path = tables_dir / 'totals.csv'
    with totals_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['ok', 'thin', 'unsupported', 'other', 'all']
        )
        writer.writeheader()
        writer.writerow(data.get('totals') or {})
    written.append(totals_path)

    scenes_path = tables_dir / 'by_scene.csv'
    with scenes_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['scene', 'ok', 'thin', 'unsupported', 'all']
        )
        writer.writeheader()
        for row in data.get('by_scene') or []:
            writer.writerow(row)
    written.append(scenes_path)

    constructs_path = tables_dir / 'by_construct.csv'
    with constructs_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['construct', 'ok', 'thin', 'unsupported', 'n_scenes_ok'],
        )
        writer.writeheader()
        for row in data.get('by_construct') or []:
            writer.writerow(row)
    written.append(constructs_path)

    labels_path = tables_dir / 'labels.csv'
    with labels_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['construct', 'kind', 'label', 'count', 'share', 'n_distinct'],
        )
        writer.writeheader()
        for row in data.get('labels') or []:
            writer.writerow(
                {
                    'construct': row.get('construct'),
                    'kind': row.get('kind'),
                    'label': row.get('label'),
                    'count': row.get('count'),
                    'share': row.get('share'),
                    'n_distinct': row.get('n_distinct', ''),
                }
            )
    written.append(labels_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'items_dir',
        type=Path,
        help='Folder of items_<scene>.json files, or one such file',
    )
    parser.add_argument(
        '-o',
        '--output',
        '--txt-out',
        dest='txt_out',
        type=Path,
        default=None,
        help='Write the text report to this file (also printed)',
    )
    parser.add_argument(
        '--json-out',
        type=Path,
        default=None,
        help='Write structured JSON for scripts/plot_review_report.py',
    )
    parser.add_argument(
        '--tables-dir',
        type=Path,
        default=None,
        help='Write CSV tables (totals, by_scene, by_construct, labels)',
    )
    args = parser.parse_args(argv)
    if not args.items_dir.exists():
        parser.error(f'not found: {args.items_dir}')
    data = collect_items_summary(_load(args.items_dir), items_dir=args.items_dir)
    text = format_items_summary(data)
    print(text, end='')
    if args.txt_out is not None:
        args.txt_out.parent.mkdir(parents=True, exist_ok=True)
        args.txt_out.write_text(text, encoding='utf-8')
        print(f'Wrote {args.txt_out}', file=sys.stderr)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {'sections': [data]}
        args.json_out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        print(f'Wrote {args.json_out}', file=sys.stderr)
    if args.tables_dir is not None:
        for path in write_tables(data, args.tables_dir):
            print(f'Wrote {path}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
