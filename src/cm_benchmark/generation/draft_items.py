"""CLI: draft taxonomy-aligned Q&A candidates from episode GT."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from cm_benchmark.generation.episode_io import load_episode, write_draft_items
from cm_benchmark.generation.pipeline import ALL_CONSTRUCTS, draft_items_for_episode


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description='Draft taxonomy Q&A candidates from episode GT (DB or JSON).'
    )
    parser.add_argument('--db_path', type=Path, default=None)
    parser.add_argument('--episode_id', type=str, default=None)
    parser.add_argument('--episode_json', type=Path, default=None)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('src/cm_benchmark/storage/ai2thor/items/draft_items.json'),
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
    args = parser.parse_args(argv)
    if args.swm_min_delay < 1:
        parser.error('--swm_min_delay must be at least 1')
    if args.swm_max_delay is not None and args.swm_max_delay < args.swm_min_delay:
        parser.error('--swm_max_delay must be greater than or equal to --swm_min_delay')
    if args.su_min_delay < 1:
        parser.error('--su_min_delay must be at least 1')
    if args.su_max_delay is not None and args.su_max_delay < args.su_min_delay:
        parser.error('--su_max_delay must be greater than or equal to --su_min_delay')

    episode = load_episode(
        db_path=args.db_path,
        episode_id=args.episode_id,
        episode_json=args.episode_json,
    )
    constructs = [c.strip() for c in args.constructs.split(',') if c.strip()]
    styles = tuple(s.strip() for s in args.styles.split(',') if s.strip())

    items = draft_items_for_episode(
        episode,
        constructs=constructs,
        max_per_construct=args.max_per_construct,
        swm_min_delay=args.swm_min_delay,
        swm_max_delay=args.swm_max_delay,
        su_min_delay=args.su_min_delay,
        su_max_delay=args.su_max_delay,
        styles=styles,
        paraphrase=args.paraphrase,
    )
    path = write_draft_items(items, args.output)
    n_ok = sum(1 for i in items if i.get('status') == 'ok')
    n_thin = sum(1 for i in items if i.get('status') == 'thin')
    n_un = sum(1 for i in items if i.get('status') == 'unsupported')
    print(
        f'Wrote {len(items)} draft item(s) → {path} '
        f'(ok={n_ok}, thin={n_thin}, unsupported={n_un})'
    )


if __name__ == '__main__':
    main()
