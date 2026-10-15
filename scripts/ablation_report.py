#!/usr/bin/env python3
"""Compare disambiguation ablations: counts, McNemar, Kendall's tau.

Reads ``<root>/<setting>/items/**/items_*.json`` (the layout of
``scripts/run_ablation.sh``). Items pair by ``item_id``, which is stable
across proximity settings.

Without ``--eval``, McNemar uses item *presence* (emitted vs dropped) and
Kendall's tau ranks constructs by ok-item count. Pass ``--eval`` JSON
``{item_id: true/false}`` (or ``{setting: {item_id: bool}}``) to score
correctness instead.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


def _load_items(root: Path) -> dict[str, list[dict]]:
    """``setting → ok items`` from ``root/<setting>/items/**/*.json``."""
    out: dict[str, list[dict]] = {}
    for setting_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        items: list[dict] = []
        for path in setting_dir.rglob('items_*.json'):
            blob = json.loads(path.read_text())
            rows = blob if isinstance(blob, list) else blob.get('items') or []
            items.extend(r for r in rows if isinstance(r, dict) and r.get('status') == 'ok')
        if items:
            out[setting_dir.name] = items
    return out


def _mode_counts(items: list[dict]) -> dict[str, Counter]:
    """construct → Counter(disambiguation mode)."""
    by: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        mode = item.get('disambiguation') or 'none'
        by[item.get('construct') or '?'][mode] += 1
        by['_all'][mode] += 1
    return by


def _mcnemar(n01: int, n10: int) -> tuple[float, float]:
    """Continuity-corrected McNemar χ² and two-sided p (χ² df=1)."""
    if n01 + n10 == 0:
        return 0.0, 1.0
    chi2 = (n01 - n10) ** 2 / (n01 + n10)
    # Regularized gamma Q(s,x) = 1 - P(s,x); df=1 → s=0.5.
    p = _chi2_sf(chi2, 1)
    return chi2, p


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of χ²(df)."""
    if x <= 0:
        return 1.0
    # erfc for df=1: P(χ² > x) = erfc(sqrt(x/2))
    if df == 1:
        return math.erfc(math.sqrt(x / 2.0))
    return 1.0


def _kendall_tau(xs: list[float], ys: list[float]) -> tuple[float, int, int]:
    """Kendall's tau-b on paired ranks. Returns (tau, concordant, discordant)."""
    n = len(xs)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                conc += 1
            else:
                disc += 1
    denom = conc + disc
    tau = 0.0 if denom == 0 else (conc - disc) / denom
    return tau, conc, disc


def _correctness(
    items: list[dict], eval_map: Optional[dict[str, bool]]
) -> dict[str, bool]:
    out = {}
    for item in items:
        iid = item.get('item_id')
        if not iid:
            continue
        if eval_map is not None and iid in eval_map:
            out[iid] = bool(eval_map[iid])
        elif 'model_correct' in item:
            out[iid] = bool(item['model_correct'])
    return out


def report(
    by_setting: dict[str, list[dict]],
    eval_blob: Optional[dict[str, Any]] = None,
) -> str:
    lines: list[str] = []
    settings = list(by_setting)
    lines.append('# Disambiguation ablation')
    lines.append('')
    lines.append('## Counts (ok items)')
    lines.append('')
    lines.append('| setting | construct | none | proximity | mark | total |')
    lines.append('| --- | --- | ---: | ---: | ---: | ---: |')
    for setting, items in by_setting.items():
        counts = _mode_counts(items)
        constructs = sorted(c for c in counts if c != '_all')
        for c in constructs:
            ctr = counts[c]
            total = sum(ctr.values())
            lines.append(
                f'| {setting} | {c} | {ctr["none"]} | {ctr["proximity"]} | '
                f'{ctr["mark"]} | {total} |'
            )
        ctr = counts['_all']
        lines.append(
            f'| {setting} | **all** | {ctr["none"]} | {ctr["proximity"]} | '
            f'{ctr["mark"]} | {sum(ctr.values())} |'
        )
    lines.append('')

    if len(settings) < 2:
        lines.append('Need at least two settings for pairwise tests.')
        return '\n'.join(lines)

    lines.append('## Pairwise McNemar and Kendall tau')
    lines.append('')
    ref = settings[0]
    for other in settings[1:]:
        a = {i['item_id']: i for i in by_setting[ref] if i.get('item_id')}
        b = {i['item_id']: i for i in by_setting[other] if i.get('item_id')}
        ids = sorted(set(a) | set(b))
        # Presence: emitted in A vs B
        n01 = n10 = 0
        for iid in ids:
            in_a, in_b = iid in a, iid in b
            if in_a and not in_b:
                n10 += 1
            elif in_b and not in_a:
                n01 += 1
        eval_a = eval_blob.get(ref) if isinstance(eval_blob, dict) and ref in eval_blob else eval_blob
        eval_b = eval_blob.get(other) if isinstance(eval_blob, dict) and other in eval_blob else eval_blob
        if isinstance(eval_a, dict) and isinstance(eval_b, dict):
            ca = _correctness(by_setting[ref], eval_a)
            cb = _correctness(by_setting[other], eval_b)
            shared = set(ca) & set(cb)
            n01 = n10 = 0
            for iid in shared:
                if (not ca[iid]) and cb[iid]:
                    n01 += 1
                elif ca[iid] and (not cb[iid]):
                    n10 += 1
            label = 'correctness'
        else:
            label = 'item presence'
        chi2, p = _mcnemar(n01, n10)
        constructs = sorted(
            {
                i.get('construct')
                for items in (by_setting[ref], by_setting[other])
                for i in items
            }
        )
        xs = [
            sum(1 for i in by_setting[ref] if i.get('construct') == c)
            for c in constructs
        ]
        ys = [
            sum(1 for i in by_setting[other] if i.get('construct') == c)
            for c in constructs
        ]
        tau, conc, disc = _kendall_tau([float(v) for v in xs], [float(v) for v in ys])
        lines.append(
            f'- **{ref} vs {other}** ({label}): McNemar n10={n10} n01={n01} '
            f'χ²={chi2:.3f} p={p:.4f}; Kendall τ={tau:.3f} '
            f'(concordant={conc}, discordant={disc}) over construct counts'
        )
    lines.append('')
    return '\n'.join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path, help='analysis/ablations (run_ablation.sh output)')
    p.add_argument('--eval', type=Path, default=None, help='optional correctness JSON')
    p.add_argument('-o', '--output', type=Path, default=None)
    args = p.parse_args(argv)
    by_setting = _load_items(args.root)
    if not by_setting:
        print(f'no items under {args.root}', file=__import__('sys').stderr)
        return 1
    eval_blob = json.loads(args.eval.read_text()) if args.eval else None
    text = report(by_setting, eval_blob)
    print(text)
    if args.output:
        args.output.write_text(text + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
