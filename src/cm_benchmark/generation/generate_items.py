"""CLI: generate taxonomy-aligned Q&A items from episode GT."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from cm_benchmark.generation.episode_io import (
    items_output_file,
    list_nav_episode_jsons,
    load_episode,
    scene_id_from_nav_json,
    scene_id_of,
    write_items,
)
from cm_benchmark.generation.pipeline import ALL_CONSTRUCTS, generate_items_for_episode


def _generate_one(args, episode, constructs, styles) -> list[dict]:
    return generate_items_for_episode(
        episode,
        constructs=constructs,
        max_per_construct=args.max_per_construct,
        swm_min_delay=args.swm_min_delay,
        swm_max_delay=args.swm_max_delay,
        su_min_delay=args.su_min_delay,
        su_max_delay=args.su_max_delay,
        styles=styles,
        paraphrase=args.paraphrase,
        visibility_model_path=args.visibility_model_path,
    )


def _report(scene_name: str, items: list[dict], path: Path) -> None:
    n_ok = sum(1 for i in items if i.get('status') == 'ok')
    n_thin = sum(1 for i in items if i.get('status') == 'thin')
    n_un = sum(1 for i in items if i.get('status') == 'unsupported')
    print(
        f'Wrote {len(items)} item(s) for {scene_name} → {path} '
        f'(ok={n_ok}, thin={n_thin}, unsupported={n_un})'
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Generate taxonomy Q&A items from nav-generator JSON. '
            '--episode_json is that command\'s --output_path '
            '(nav_data/<scene_id>/nav_<scene_id>.json).'
        )
    )
    parser.add_argument('--db_path', type=Path, default=None)
    parser.add_argument('--episode_id', type=str, default=None)
    parser.add_argument(
        '--episode_json',
        type=Path,
        default=None,
        help=(
            'Nav generator JSON root, one scene folder, or one nav_<scene>.json. '
            'Same path as ai2thor_nav_generator --output_path.'
        ),
    )
    parser.add_argument(
        '--output_path',
        type=Path,
        default=Path('src/cm_benchmark/storage/ai2thor/items'),
        help=(
            'Root directory. Each scene is written to '
            '<output_path>/<scene_id>/items_<scene_id>.json.'
        ),
    )
    parser.add_argument(
        '--constructs',
        type=str,
        default=','.join(ALL_CONSTRUCTS),
        help='Comma-separated construct ids',
    )
    parser.add_argument('--max_per_construct', type=int, default=3)
    parser.add_argument(
        '--swm_min_delay',
        type=int,
        default=2,
        help='Minimum navigation-step delay for spatial working-memory items (default: 2)',
    )
    parser.add_argument(
        '--swm_max_delay',
        type=int,
        default=None,
        help='Maximum SWM delay; omit to allow any delay through the episode end',
    )
    parser.add_argument(
        '--su_min_delay',
        type=int,
        default=2,
        help='Minimum navigation-step delay for spatial-updating items (default: 2)',
    )
    parser.add_argument(
        '--su_max_delay',
        type=int,
        default=None,
        help='Maximum spatial-updating delay; omit to allow any delay through the episode end',
    )
    parser.add_argument(
        '--styles',
        type=str,
        default='concise,verbose',
        help='Question styles: concise,verbose',
    )
    parser.add_argument(
        '--paraphrase',
        action='store_true',
        help='Optional LLM paraphrase of question text only (no-op without provider)',
    )
    parser.add_argument(
        '--visibility_model_path',
        type=str,
        default=None,
        help=(
            'Path to visibility_filter.joblib (DecisionTree). '
            'Applied to every scene under --episode_json. '
            'If omitted, searches CM_VISIBILITY_FILTER_MODEL and default '
            'analysis/dt_tune/ paths; falls back to static thresholds if none found.'
        ),
    )
    args = parser.parse_args(argv)
    if args.episode_json is None and not (args.db_path and args.episode_id):
        parser.error('Provide --episode_json (nav_data root) or both --db_path and --episode_id')
    if args.swm_min_delay < 1:
        parser.error('--swm_min_delay must be at least 1')
    if args.swm_max_delay is not None and args.swm_max_delay < args.swm_min_delay:
        parser.error('--swm_max_delay must be greater than or equal to --swm_min_delay')
    if args.su_min_delay < 1:
        parser.error('--su_min_delay must be at least 1')
    if args.su_max_delay is not None and args.su_max_delay < args.su_min_delay:
        parser.error('--su_max_delay must be greater than or equal to --su_min_delay')

    constructs = [c.strip() for c in args.constructs.split(',') if c.strip()]
    styles = tuple(s.strip() for s in args.styles.split(',') if s.strip())

    if args.episode_json is None:
        episode = load_episode(db_path=args.db_path, episode_id=args.episode_id)
        scene_name = scene_id_of(episode)
        items = _generate_one(args, episode, constructs, styles)
        path = write_items(items, items_output_file(args.output_path, scene_name))
        _report(scene_name, items, path)
        return

    sources = list_nav_episode_jsons(args.episode_json)
    failures = []
    built = []
    for source in sources:
        scene_name = scene_id_from_nav_json(source)
        try:
            episode = load_episode(episode_json=source)
            items = _generate_one(args, episode, constructs, styles)
            path = write_items(items, items_output_file(args.output_path, scene_name))
            _report(scene_name, items, path)
            built.append(scene_name)
        except Exception as exc:
            print(f'Failed {source}: {exc}')
            failures.append((source, exc))
    print(f'Built items for {len(built)} scene(s): {", ".join(built) or "(none)"}')
    if failures:
        raise SystemExit(
            f'{len(failures)} scene(s) failed under {args.episode_json}'
        )


if __name__ == '__main__':
    main()
