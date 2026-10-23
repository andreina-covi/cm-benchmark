"""CLI: generate taxonomy-aligned Q&A items from episode GT."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Optional, Union

from cm_benchmark.generation.config import load_generation_config
from cm_benchmark.generation.episode_io import (
    items_output_file,
    list_nav_episode_jsons,
    load_episode,
    scene_id_from_nav_json,
    scene_id_of,
    write_items,
)
from cm_benchmark.generation.pipeline import ALL_CONSTRUCTS, generate_items_for_episode
from cm_benchmark.generation.planner import validate_delay_window


def _dedupe_items(items: list[dict]) -> list[dict]:
    """Keep the first item for each id. A repeated id is the same question."""
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        iid = item.get('item_id')
        if iid and iid in seen:
            continue
        if iid:
            seen.add(iid)
        out.append(item)
    return out


def _answer_label_and_letter(item: dict) -> tuple[str, Optional[str]]:
    """``(answer text, MC letter)``. Free-form items have no letter."""
    options = item.get('options') or {}
    answer = item.get('answer')
    if isinstance(options, dict) and answer in options:
        return str(options[answer]), str(answer)
    if answer is None:
        return '', None
    return str(answer), None


def _is_selection_unit(item: dict, scene_ids: set[str]) -> bool:
    """Concise item, or a verbose item that has no concise partner in the scene."""
    if item.get('question_style') != 'verbose':
        return True
    return item.get('paired_item_id') not in scene_ids


def _trim_label_share(chosen: list[tuple[str, dict]], share: float) -> None:
    """Drop extra copies so no label exceeds ``share`` of the final set.

    A label that appears once is always kept, so a free-form action
    sequence is not removed for being unique.
    """
    while len(chosen) > 1:
        labels = Counter(_answer_label_and_letter(item)[0] for _name, item in chosen)
        n = len(chosen)
        limit = max(1, int(float(share) * n))
        over = [label for label, count in labels.items() if count > limit]
        if not over:
            break
        worst = max(over, key=lambda label: (labels[label], label))
        letters = Counter(
            _answer_label_and_letter(item)[1] or ''
            for _name, item in chosen
            if _answer_label_and_letter(item)[0] == worst
        )
        drop_at = max(
            (
                index
                for index, (_name, item) in enumerate(chosen)
                if _answer_label_and_letter(item)[0] == worst
            ),
            key=lambda index: (
                letters[_answer_label_and_letter(chosen[index][1])[1] or ''],
                index,
            ),
        )
        chosen.pop(drop_at)


def _round_robin(
    pool: list[tuple[str, dict]],
    *,
    per_scene: int,
    slots: int,
    rng: random.Random,
) -> list[tuple[str, dict]]:
    """Keep at most ``per_scene`` items from each scene, then alternate scenes."""
    by_scene: dict[str, list[dict]] = {}
    scene_order: list[str] = []
    for name, item in pool:
        if name not in by_scene:
            by_scene[name] = []
            scene_order.append(name)
        by_scene[name].append(item)
    queues: dict[str, list[dict]] = {}
    for name in scene_order:
        items = list(by_scene[name])
        rng.shuffle(items)
        queues[name] = items[:per_scene]
    order = list(scene_order)
    rng.shuffle(order)
    chosen: list[tuple[str, dict]] = []
    while len(chosen) < slots and any(queues.values()):
        for name in order:
            if len(chosen) >= slots or not queues[name]:
                continue
            chosen.append((name, queues[name].pop(0)))
    return chosen


def balance_scene_items(
    scenes: list[tuple[str, list[dict]]],
    *,
    max_per_construct: int,
    max_per_scene: int,
    max_label_share: float,
    seed: int,
) -> list[tuple[str, list[dict]]]:
    """Pick a whole-set sample, then cap answer labels.

    Each scene contributes at most ``max_per_scene`` concise facts per
    construct. Those lists are shuffled with ``seed`` and drawn round-robin
    up to ``max_per_construct`` times the number of scenes. The label cap
    runs after that draw. Verbose partners of a kept concise item are kept.
    A scene that produced no ok fact for a construct keeps its placeholder.
    """
    scenes = [(name, _dedupe_items(items)) for name, items in scenes]
    n_scenes = max(1, len(scenes))
    target = max(1, max_per_construct * n_scenes)
    by_id = {
        item['item_id']: item
        for _name, items in scenes
        for item in items
        if item.get('item_id')
    }
    units: dict[str, list[tuple[str, dict]]] = {}
    order: list[str] = []
    unsupported: dict[tuple[str, str], dict] = {}
    had_ok: set[tuple[str, str]] = set()
    for name, items in scenes:
        scene_ids = {item.get('item_id') for item in items}
        for item in items:
            construct = item.get('construct') or ''
            if construct not in units:
                units[construct] = []
                order.append(construct)
            if item.get('status') != 'ok':
                unsupported.setdefault((name, construct), item)
                continue
            had_ok.add((name, construct))
            if _is_selection_unit(item, scene_ids):
                units[construct].append((name, item))

    rng = random.Random(seed)
    keep: dict[str, set[str]] = {name: set() for name, _items in scenes}
    for construct in order:
        chosen = _round_robin(
            units.get(construct) or [],
            per_scene=max_per_scene,
            slots=target,
            rng=rng,
        )
        _trim_label_share(chosen, max_label_share)
        for name, item in chosen:
            keep[name].add(item['item_id'])
            partner = item.get('paired_item_id')
            if partner in by_id:
                keep[name].add(partner)
    for (name, construct), placeholder in unsupported.items():
        if (name, construct) in had_ok:
            continue
        pid = placeholder.get('item_id')
        if pid:
            keep[name].add(pid)
    return [
        (name, [item for item in items if item.get('item_id') in keep[name]])
        for name, items in scenes
    ]


def _balance_generated(
    scenes: list[tuple[str, list[dict]]], cfg
) -> list[tuple[str, list[dict]]]:
    """Balance every scene together, then write."""
    if not scenes:
        return []
    return balance_scene_items(
        scenes,
        max_per_construct=cfg.budget.max_per_construct,
        max_per_scene=cfg.budget.max_per_scene,
        max_label_share=cfg.options.max_label_share,
        seed=cfg.options.shuffle_seed,
    )


def _generate_one(
    args, episode, constructs, styles, marked_root: Optional[Union[str, Path]] = None
) -> list[dict]:
    """Items for one episode with the CLI's construct, delay and style settings."""
    return generate_items_for_episode(
        episode,
        constructs=constructs,
        max_per_construct=args._plan_cap,
        swm_min_delay=args.swm_min_delay,
        swm_max_delay=args.swm_max_delay,
        su_min_delay=args.su_min_delay,
        su_max_delay=args.su_max_delay,
        styles=styles,
        paraphrase=args.paraphrase,
        visibility_model_path=args.visibility_model_path,
        cfg=args._generation_cfg,
        marked_root=Path(marked_root) if marked_root is not None else None,
    )


def _report(scene_name: str, items: list[dict], path: Path) -> None:
    """Print the item count per status for one written scene."""
    n_ok = sum(1 for i in items if i.get('status') == 'ok')
    n_thin = sum(1 for i in items if i.get('status') == 'thin')
    n_un = sum(1 for i in items if i.get('status') == 'unsupported')
    print(
        f'Wrote {len(items)} item(s) for {scene_name} → {path} '
        f'(ok={n_ok}, thin={n_thin}, unsupported={n_un})'
    )


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry: one DB episode, or every ``nav_<scene>.json`` under ``--episode_json``."""
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
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help=(
            'Path to generation.yaml (quality parameters). Default: '
            'configs/generation.yaml. CLI flags below override this file.'
        ),
    )
    parser.add_argument(
        '--max_per_construct',
        type=int,
        default=None,
        help='Override budget.max_per_construct from --config (default in generation.yaml: 3)',
    )
    parser.add_argument(
        '--swm_min_delay',
        type=int,
        default=None,
        help='Override spatial_working_memory.min_delay from --config',
    )
    parser.add_argument(
        '--swm_max_delay',
        type=int,
        default=None,
        help='Override spatial_working_memory.max_delay; omit to use the config (null = no cap)',
    )
    parser.add_argument(
        '--su_min_delay',
        type=int,
        default=None,
        help='Override spatial_updating.min_delay from --config',
    )
    parser.add_argument(
        '--su_max_delay',
        type=int,
        default=None,
        help='Override spatial_updating.max_delay; omit to use the config (null = no cap)',
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
    try:
        cfg = load_generation_config(args.config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f'invalid --config: {exc}')
    cfg = cfg.with_overrides(
        max_per_construct=args.max_per_construct,
        swm_min_delay=args.swm_min_delay,
        swm_max_delay=args.swm_max_delay,
        su_min_delay=args.su_min_delay,
        su_max_delay=args.su_max_delay,
        visibility_model_path=args.visibility_model_path,
    )
    args._generation_cfg = cfg
    args._plan_cap = cfg.budget.candidate_pool
    try:
        validate_delay_window(
            'SWM',
            cfg.spatial_working_memory.min_delay,
            cfg.spatial_working_memory.max_delay,
        )
        validate_delay_window(
            'spatial_updating',
            cfg.spatial_updating.min_delay,
            cfg.spatial_updating.max_delay,
        )
    except ValueError as exc:
        parser.error(str(exc))

    constructs = [c.strip() for c in args.constructs.split(',') if c.strip()]
    styles = tuple(s.strip() for s in args.styles.split(',') if s.strip())

    if args.episode_json is None:
        episode = load_episode(db_path=args.db_path, episode_id=args.episode_id)
        scene_name = scene_id_of(episode)
        items = _generate_one(
            args,
            episode,
            constructs,
            styles,
            marked_root=args.output_path / scene_name / 'marked',
        )
        balanced = _balance_generated(
            [(scene_name, items)],
            cfg,
        )
        for scene_name, items in balanced:
            path = write_items(items, items_output_file(args.output_path, scene_name))
            _report(scene_name, items, path)
        return

    sources = list_nav_episode_jsons(args.episode_json)
    failures = []
    gathered: list[tuple[str, list[dict]]] = []
    for source in sources:
        scene_name = scene_id_from_nav_json(source)
        try:
            episode = load_episode(episode_json=source)
            items = _generate_one(
                args,
                episode,
                constructs,
                styles,
                marked_root=args.output_path / scene_name / 'marked',
            )
            gathered.append((scene_name, items))
        except Exception as exc:
            print(f'Failed {source}: {exc}')
            failures.append((source, exc))
    built = []
    for scene_name, items in _balance_generated(gathered, cfg):
        path = write_items(items, items_output_file(args.output_path, scene_name))
        _report(scene_name, items, path)
        built.append(scene_name)
    print(f'Built items for {len(built)} scene(s): {", ".join(built) or "(none)"}')
    if failures:
        raise SystemExit(
            f'{len(failures)} scene(s) failed under {args.episode_json}'
        )


if __name__ == '__main__':
    main()
