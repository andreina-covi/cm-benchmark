#!/usr/bin/env python3
"""Replay recorded actions on nav JSON episode(s) through the class-4 scorer."""

from __future__ import annotations

import argparse
import json
import sys

from cm_benchmark.generation.episode_io import list_nav_episode_jsons, load_episode_from_json
from cm_benchmark.generation.route_replay import replay_recorded_actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'episode_json',
        help='nav JSON file or directory of per-scene nav_*.json',
    )
    args = parser.parse_args(argv)
    episodes = [load_episode_from_json(p) for p in list_nav_episode_jsons(args.episode_json)]
    mismatches = []
    for ep in episodes:
        rec = replay_recorded_actions(ep)
        scene = rec.get('scene') or ep.get('scene')
        status = 'ok' if rec.get('match') else 'MISMATCH'
        print(
            f"{status}  {scene}  actions={rec.get('n_actions')}  "
            f"end_d={rec.get('end_d_m')} end_dh={rec.get('end_dh_deg')} "
            f"blocked={rec.get('blocked_moves')}  reason={rec.get('reason')}"
        )
        if not rec.get('match'):
            mismatches.append(rec)
    print(f'{len(episodes) - len(mismatches)}/{len(episodes)} matched')
    if mismatches:
        print(json.dumps(mismatches, indent=2, default=str))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
