#!/usr/bin/env python3
"""Summarize how many questions ``generate_items`` produced.

Reads ``items_*.json`` under a generation ``--output_path`` (or one file).
Counts concise facts only (a verbose partner is not counted again).

Prints:
  - grand totals (ok / thin / unsupported)
  - per-scene counts
  - per-construct counts
  - answer-label distributions for concise ok items

Write the same text with ``-o path/to/report.txt``.
"""

from __future__ import annotations

import argparse
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
        scene_from_file = path.stem.removeprefix('items_') if path.stem.startswith('items_') else ''
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


def report(items: list[dict], *, items_dir: Path | None = None) -> str:
    """Text summary of concise facts: totals, by scene, by construct, labels."""
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

    lines: list[str] = []
    if items_dir is not None:
        lines.append(f'Items root: {items_dir}')
    n_scenes = len(by_scene)
    lines.append(f'Scenes with items: {n_scenes}')
    lines.append('')
    lines.append('TOTAL (concise)')
    for key in ('ok', 'thin', 'unsupported'):
        lines.append(f'  {key:<12} {totals.get(key, 0):>6}')
    other = sum(v for k, v in totals.items() if k not in {'ok', 'thin', 'unsupported'})
    if other:
        lines.append(f'  {"other":<12} {other:>6}')
    lines.append(f'  {"all":<12} {len(rows):>6}')
    lines.append('')

    lines.append('By scene (concise)')
    header = f'{"scene":<28} {"ok":>5} {"thin":>5} {"unsupported":>12} {"all":>5}'
    lines.append(header)
    scene_ok = scene_thin = scene_un = scene_all = 0
    for scene in sorted(by_scene):
        c = by_scene[scene]
        ok = c.get('ok', 0)
        thin = c.get('thin', 0)
        un = c.get('unsupported', 0)
        n = sum(c.values())
        scene_ok += ok
        scene_thin += thin
        scene_un += un
        scene_all += n
        lines.append(f'{scene:<28} {ok:>5} {thin:>5} {un:>12} {n:>5}')
    lines.append(
        f'{"TOTAL":<28} {scene_ok:>5} {scene_thin:>5} {scene_un:>12} {scene_all:>5}'
    )
    lines.append('')

    lines.append('By construct (concise)')
    lines.append(f'{"construct":<32} {"ok":>5} {"thin":>5} {"unsupported":>12} {"scenes":>7}')
    for construct in constructs:
        c = by_construct[construct]
        lines.append(
            f'{construct:<32} {c.get("ok", 0):>5} {c.get("thin", 0):>5} '
            f'{c.get("unsupported", 0):>12} {len(scenes_ok[construct]):>7}'
        )
    lines.append('')

    lines.append('Labels and letters (concise ok)')
    any_ok = False
    for construct in constructs:
        if by_construct[construct].get('ok', 0) == 0:
            continue
        any_ok = True
        lines.append(construct)
        if labels[construct]:
            total = sum(labels[construct].values())
            for label, count in labels[construct].most_common():
                lines.append(f'  {label:<20} {count:>4}  {count / total:>5.0%}')
            letter_bits = '  '.join(
                f'{letter} {letters[construct][letter]}'
                for letter in ('A', 'B', 'C', 'D')
                if letters[construct][letter]
            )
            extra = [
                f'{letter} {count}'
                for letter, count in sorted(letters[construct].items())
                if letter not in {'A', 'B', 'C', 'D'}
            ]
            if extra:
                letter_bits = (letter_bits + '  ' + '  '.join(extra)).strip()
            if letter_bits:
                lines.append(f'  letters  {letter_bits}')
        if freeform[construct]:
            n_actions = sum(freeform[construct].values())
            lines.append(
                f'  action sequences: {n_actions} ({len(freeform[construct])} distinct)'
            )
    if not any_ok:
        lines.append('  (none)')
    return '\n'.join(lines) + '\n'


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
        type=Path,
        default=None,
        help='Write the report to this file (also printed)',
    )
    args = parser.parse_args(argv)
    if not args.items_dir.exists():
        parser.error(f'not found: {args.items_dir}')
    text = report(_load(args.items_dir), items_dir=args.items_dir)
    print(text, end='')
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f'Wrote {args.output}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
