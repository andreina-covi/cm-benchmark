#!/usr/bin/env python3
"""Print ok and unsupported counts, scenes, and answer distributions.

Reads ``items_*.json`` under a generation output folder. Concise items are
the facts; a verbose partner is not counted again.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load(root: Path) -> list[dict]:
    """Every item dict under ``root``."""
    paths = sorted(root.rglob('items_*.json')) if root.is_dir() else [root]
    items: list[dict] = []
    for path in paths:
        blob = json.loads(path.read_text())
        rows = blob if isinstance(blob, list) else blob.get('items') or []
        for row in rows:
            if isinstance(row, dict):
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


def report(items: list[dict]) -> str:
    """Text table of concise facts."""
    rows = _concise(items)
    constructs: list[str] = []
    ok_n: Counter = Counter()
    unsup_n: Counter = Counter()
    scenes: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, Counter] = defaultdict(Counter)
    letters: dict[str, Counter] = defaultdict(Counter)
    freeform: dict[str, Counter] = defaultdict(Counter)
    for item in rows:
        construct = item.get('construct') or ''
        if construct not in ok_n and construct not in unsup_n:
            constructs.append(construct)
        status = item.get('status')
        if status == 'ok':
            ok_n[construct] += 1
            scene = item.get('scene_id') or ''
            if scene:
                scenes[construct].add(str(scene))
            label, letter = _label_and_letter(item)
            if letter:
                labels[construct][label] += 1
                letters[construct][letter] += 1
            else:
                freeform[construct][label] += 1
        elif status == 'unsupported':
            unsup_n[construct] += 1
    for construct in ok_n:
        if construct not in constructs:
            constructs.append(construct)
    lines = [
        f'{"construct":<32} {"ok":>5} {"unsupported":>12} {"scenes":>7}',
    ]
    for construct in constructs:
        lines.append(
            f'{construct:<32} {ok_n[construct]:>5} {unsup_n[construct]:>12} '
            f'{len(scenes[construct]):>7}'
        )
    lines.append('')
    lines.append('labels and letters (concise ok)')
    for construct in constructs:
        if ok_n[construct] == 0:
            continue
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
            lines.append(f'  letters  {letter_bits}')
        if freeform[construct]:
            n_actions = sum(freeform[construct].values())
            lines.append(
                f'  action sequences: {n_actions} ({len(freeform[construct])} distinct)'
            )
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'items_dir',
        type=Path,
        help='Folder of items_<scene>.json files, or one such file',
    )
    args = parser.parse_args(argv)
    if not args.items_dir.exists():
        parser.error(f'not found: {args.items_dir}')
    print(report(_load(args.items_dir)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
