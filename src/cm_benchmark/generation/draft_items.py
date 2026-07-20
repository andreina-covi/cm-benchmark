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
