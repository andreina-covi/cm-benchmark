#!/usr/bin/env python3
"""Compare two run_baseline.sh output folders item-by-item and nav-by-nav.

Exits 0 when every items_*.json and nav_*.json matches after normalisation:
run-folder prefixes in paths are removed, and keys in ``--ignore`` (config
provenance, marked-frame paths, …) are dropped before comparing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_IGNORE = ('generation_config', 'generation_config_hash')


def _strip(value, roots: tuple[str, ...], ignore: frozenset[str]):
    if isinstance(value, dict):
        return {k: _strip(v, roots, ignore) for k, v in value.items() if k not in ignore}
    if isinstance(value, list):
        return [_strip(v, roots, ignore) for v in value]
    if isinstance(value, str):
        for root in roots:
            value = value.replace(root, '<RUN>')
    return value


def _load(path: Path, root: Path, ignore: frozenset[str]):
    roots = (str(root.resolve()), str(root))
    return _strip(json.loads(path.read_text()), roots, ignore)


def _first_diff(a, b, where='$'):
    if type(a) is not type(b):
        return f'{where}: type {type(a).__name__} != {type(b).__name__}'
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f'{where}.{k}: missing on one side'
            d = _first_diff(a[k], b[k], f'{where}.{k}')
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f'{where}: len {len(a)} != {len(b)}'
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f'{where}[{i}]')
            if d:
                return d
        return None
    return None if a == b else f'{where}: {a!r} != {b!r}'


def compare(base: Path, other: Path, ignore: frozenset[str], kinds=('items', 'nav_data')) -> list[str]:
    problems = []
    for kind in kinds:
        files = sorted((base / kind).rglob('*.json'))
        if not files:
            problems.append(f'no {kind} JSON under {base}')
        for f in files:
            rel = f.relative_to(base)
            g = other / rel
            if not g.is_file():
                problems.append(f'missing {rel}')
                continue
            d = _first_diff(_load(f, base, ignore), _load(g, other, ignore))
            if d:
                problems.append(f'{rel}: {d}')
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('base', type=Path)
    p.add_argument('other', type=Path)
    p.add_argument('--ignore', nargs='*', default=list(DEFAULT_IGNORE))
    p.add_argument('--items-only', action='store_true')
    args = p.parse_args(argv)
    kinds = ('items',) if args.items_only else ('items', 'nav_data')
    problems = compare(args.base, args.other, frozenset(args.ignore), kinds)
    for line in problems:
        print(line)
    print('IDENTICAL' if not problems else f'{len(problems)} difference(s)')
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
