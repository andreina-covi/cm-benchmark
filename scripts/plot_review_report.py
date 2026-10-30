#!/usr/bin/env python3
"""Plot report JSON from ``analyze_review.py`` or ``report_items.py`` ``--json-out``.

Writes PNG figures into ``--output`` (default: ``<json_stem>_plots/``):

Human / VLM (``analyze_review.py``):
  human_accuracy.png, human_agreement.png, human_nav_metrics.png,
  human_checks.png, human_problems.png, vlm_accuracy.png

Generation items (``report_items.py``):
  items_totals.png, items_by_construct.png, items_by_scene_top.png,
  items_ok_share.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'expected object in {path}')
    return data


def _sections(data: dict) -> list[dict]:
    sections = data.get('sections')
    if isinstance(sections, list):
        return [s for s in sections if isinstance(s, dict)]
    # Allow a bare section payload.
    if data.get('kind') in {'human', 'vlm', 'items'}:
        return [data]
    return []


def _short(name: str) -> str:
    return {
        'egocentric_encoding': 'ego',
        'allocentric_encoding': 'allo',
        'spatial_working_memory': 'SWM',
        'invisible_displacement': 'ID',
        'spatial_updating': 'SU',
        'perspective_taking': 'PT',
        'route_knowledge': 'route',
        'survey_based_route_planning': 'survey',
    }.get(name, name)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


def _bar(
    plt,
    *,
    labels: list[str],
    series: list[tuple[str, list[float | None]]],
    ylabel: str,
    title: str,
    path: Path,
    ylim: tuple[float, float] | None = (0.0, 1.05),
) -> None:
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(labels) + 2), 4.2))
    x = list(range(len(labels)))
    width = 0.8 / max(1, len(series))
    for i, (name, values) in enumerate(series):
        ys = [float(v) if v is not None else float('nan') for v in values]
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], ys, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if len(series) > 1:
        ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_human(section: dict, out: Path, plt) -> list[Path]:
    written: list[Path] = []
    constructs = section.get('constructs') or []
    if constructs:
        labels = [_short(c['construct']) for c in constructs]
        path = out / 'human_accuracy.png'
        _bar(
            plt,
            labels=labels,
            series=[
                ('accuracy', [c.get('accuracy') for c in constructs]),
                ('flag rate', [c.get('flag_rate') for c in constructs]),
            ],
            ylabel='rate',
            title='Human review: accuracy and flag rate',
            path=path,
        )
        written.append(path)

        path = out / 'human_agreement.png'
        _bar(
            plt,
            labels=labels,
            series=[('alpha', [c.get('alpha') for c in constructs])],
            ylabel="Krippendorff's α",
            title='Human review: inter-annotator agreement',
            path=path,
            ylim=(-0.05, 1.05),
        )
        written.append(path)

        nav = [
            c
            for c in constructs
            if c.get('kind') == 'nav'
            and (
                c.get('mean_spl') is not None
                or c.get('mean_ndtw') is not None
                or c.get('mean_sdtw') is not None
            )
        ]
        if nav:
            path = out / 'human_nav_metrics.png'
            _bar(
                plt,
                labels=[_short(c['construct']) for c in nav],
                series=[
                    ('SPL', [c.get('mean_spl') for c in nav]),
                    ('nDTW', [c.get('mean_ndtw') for c in nav]),
                    ('SDTW', [c.get('mean_sdtw') for c in nav]),
                ],
                ylabel='score',
                title='Human review: class-4 navigation metrics',
                path=path,
            )
            written.append(path)

    checks = section.get('check_accuracy') or []
    if checks and int(section.get('n_check_items') or 0) > 0:
        path = out / 'human_checks.png'
        _bar(
            plt,
            labels=[c['reviewer_id'] for c in checks],
            series=[('check accuracy', [c.get('accuracy') for c in checks])],
            ylabel='accuracy',
            title='Human review: check-item accuracy',
            path=path,
        )
        written.append(path)

    problems = section.get('problems') or []
    path = out / 'human_problems.png'
    n_flag = sum(1 for p in problems if p.get('both_flagged'))
    n_wrong = sum(1 for p in problems if p.get('both_wrong'))
    _bar(
        plt,
        labels=['both_flagged', 'both_wrong'],
        series=[('items', [float(n_flag), float(n_wrong)])],
        ylabel='count',
        title='Human review: items both reviewers flagged or missed',
        path=path,
        ylim=None,
    )
    written.append(path)
    return written


def plot_vlm(section: dict, out: Path, plt) -> list[Path]:
    rows = section.get('results') or []
    if not rows:
        return []
    constructs: list[str] = []
    for row in rows:
        name = row['construct']
        if name not in constructs:
            constructs.append(name)
    vision = []
    blind = []
    chance = []
    human = []
    for name in constructs:
        v = b = c = h = None
        for row in rows:
            if row['construct'] != name:
                continue
            if row['condition'] == 'vision':
                v = row.get('accuracy')
            elif row['condition'] == 'blind':
                b = row.get('accuracy')
            if row.get('chance') is not None:
                c = row.get('chance')
            if row.get('human_accuracy') is not None:
                h = row.get('human_accuracy')
        vision.append(v)
        blind.append(b)
        chance.append(c)
        human.append(h)
    series = [
        ('vision', vision),
        ('blind', blind),
    ]
    if any(v is not None for v in chance):
        series.append(('chance', chance))
    if any(v is not None for v in human):
        series.append(('human', human))
    path = out / 'vlm_accuracy.png'
    _bar(
        plt,
        labels=[_short(c) for c in constructs],
        series=series,
        ylabel='accuracy',
        title='VLM chat kit: vision vs blind',
        path=path,
    )
    return [path]


def plot_items(section: dict, out: Path, plt) -> list[Path]:
    written: list[Path] = []
    totals = section.get('totals') or {}
    path = out / 'items_totals.png'
    _bar(
        plt,
        labels=['ok', 'thin', 'unsupported', 'all'],
        series=[
            (
                'count',
                [
                    float(totals.get('ok') or 0),
                    float(totals.get('thin') or 0),
                    float(totals.get('unsupported') or 0),
                    float(totals.get('all') or 0),
                ],
            )
        ],
        ylabel='items',
        title='Generated items: status totals (concise)',
        path=path,
        ylim=None,
    )
    written.append(path)

    constructs = section.get('by_construct') or []
    if constructs:
        labels = [_short(c['construct']) for c in constructs]
        path = out / 'items_by_construct.png'
        _bar(
            plt,
            labels=labels,
            series=[
                ('ok', [float(c.get('ok') or 0) for c in constructs]),
                ('thin', [float(c.get('thin') or 0) for c in constructs]),
                ('unsupported', [float(c.get('unsupported') or 0) for c in constructs]),
            ],
            ylabel='items',
            title='Generated items: by construct',
            path=path,
            ylim=None,
        )
        written.append(path)

        ok_vals = [float(c.get('ok') or 0) for c in constructs]
        un_vals = [float(c.get('unsupported') or 0) for c in constructs]
        share = []
        for ok, un in zip(ok_vals, un_vals):
            denom = ok + un
            share.append((ok / denom) if denom else None)
        path = out / 'items_ok_share.png'
        _bar(
            plt,
            labels=labels,
            series=[('ok / (ok+unsupported)', share)],
            ylabel='share',
            title='Generated items: ok share by construct',
            path=path,
        )
        written.append(path)

    scenes = list(section.get('by_scene') or [])
    if scenes:
        # Top scenes by ok count (cap so the figure stays readable).
        ranked = sorted(scenes, key=lambda r: (-int(r.get('ok') or 0), r.get('scene') or ''))
        top = ranked[:min(25, len(ranked))]
        path = out / 'items_by_scene_top.png'
        _bar(
            plt,
            labels=[str(r.get('scene') or '?') for r in top],
            series=[('ok', [float(r.get('ok') or 0) for r in top])],
            ylabel='ok items',
            title=f'Generated items: top {len(top)} scenes by ok count',
            path=path,
            ylim=None,
        )
        written.append(path)
    return written


def plot_report(data: dict, output: Path) -> list[Path]:
    """Render every recognized section; return written PNG paths."""
    output.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()
    written: list[Path] = []
    for section in _sections(data):
        kind = section.get('kind')
        if kind == 'human':
            written.extend(plot_human(section, output, plt))
        elif kind == 'vlm':
            written.extend(plot_vlm(section, output, plt))
        elif kind == 'items':
            written.extend(plot_items(section, output, plt))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'json_path',
        type=Path,
        help='JSON from analyze_review.py or report_items.py --json-out',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=None,
        help='Directory for PNG plots (default: <json_stem>_plots)',
    )
    args = parser.parse_args(argv)
    if not args.json_path.is_file():
        parser.error(f'not found: {args.json_path}')
    out = args.output
    if out is None:
        out = args.json_path.with_name(args.json_path.stem + '_plots')
    try:
        written = plot_report(_load(args.json_path), out)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not written:
        print('no plottable sections in JSON', file=sys.stderr)
        return 1
    print(f'Wrote {len(written)} plot(s) → {out}')
    for path in written:
        print(f'  {path.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
