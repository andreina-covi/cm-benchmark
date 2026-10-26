#!/usr/bin/env python3
"""Build a manual VLM chat kit from the same items as the human review package.

For each sampled study item write two chat folders:

* ``items/<item_id>/vision`` — frames renamed ``frame_01.png`` … plus ``prompt.txt``
* ``items/<item_id>/blind`` — ``prompt.txt`` only (one line: no images available)

``prompt.txt`` uses the same framing as evaluation, but MC answers may be
A–D, ``Can't tell``, plus an optional comment (aligned with human review).
Gold never appears in those folders.

Also writes ``results_template.csv`` (fill ``model`` / ``raw_reply`` /
``comment`` by hand), ``answers.csv``, ``construct_batches/``, and
``kit_manifest.json``.

Selection matches ``build_review_package.py`` when ``--seed`` and
``--per-construct`` match. Default ``--items-per-construct 2`` so each
construct has at least two chat examples. Pass ``--review-package`` to reuse
an existing human study list. ``--max-items`` caps the whole kit (smoke test).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from cm_benchmark.evaluation.protocol import ACTION_VOCAB

REPO = Path(__file__).resolve().parent
NAV_CONSTRUCTS = frozenset({
    'route_knowledge',
    'survey_based_route_planning',
})
DEFAULT_ITEMS_PER_CONSTRUCT = 2
_ACTION_LIST = ', '.join(ACTION_VOCAB)

# Pilot chat instruction: same task framing as SYSTEM_INSTRUCTION, but MC
# matches human review (Can't tell + optional comment). Final cm-evaluation
# still uses protocol.SYSTEM_INSTRUCTION (forced A–D).

CHAT_SYSTEM_INSTRUCTION = (
    'You are a robot exploring a house with a front-facing camera. The images are '
    'frames from your camera in chronological order: the first image is the earliest '
    'and the last image is your current view ("now"). One step is one action you took. '
    'Answer using only the images and text in this message.'
    'Directions ("left", "right", "ahead", "behind") are relative to where you are '
    'facing in the last image, unless the question asks you to imagine another position. '
    'An object "marked with a <colour> dot" is the object under that dot in the images. '
    'Your actions: move_ahead (0.2 m forward), move_back (0.2 m backward), '
    'rotate_left (turn 45° left), rotate_right (turn 45° right). '
    'Multiple-choice questions: on the first line, answer with only the option letter '
    '(A, B, C or D). Always choose one option, even if unsure. '
    'Route questions: on the first line, answer with the actions in order, '
    'comma-separated, using only the action names above. Always give your best attempt. '
    'Optionally, add one short comment on a second line (for route questions you may '
    'describe the route using landmarks). The comment is not scored.'
)

def _load_review():
    path = REPO / 'build_review_package.py'
    spec = importlib.util.spec_from_file_location('build_review_package', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _safe(item_id: str) -> str:
    return item_id.replace('/', '_').replace('\\', '_')


def is_nav(item: dict) -> bool:
    return item.get('construct') in NAV_CONSTRUCTS


def gold_for_answers(item: dict) -> str:
    """Letter for MC; reference action string for class-4."""
    if is_nav(item):
        answer = item.get('answer')
        if answer is None:
            ctx = item.get('context') or {}
            seq = ctx.get('action_sequence') or []
            return ', '.join(str(a) for a in seq)
        return str(answer)
    return '' if item.get('answer') is None else str(item.get('answer'))


def build_prompt(item: dict, *, blind: bool) -> str:
    """Exact text to paste into a chat app. Never includes the gold answer."""
    parts = [CHAT_SYSTEM_INSTRUCTION.strip(), '']
    if blind:
        parts.append(
            'No images are available for this item. Answer from the question text alone.'
        )
        parts.append('')
    question = (item.get('question') or '').strip()
    parts.append(question)
    if not is_nav(item):
        options = item.get('options') or {}
        if isinstance(options, dict) and options:
            parts.append('')
            for letter in sorted(options):
                parts.append(f'{letter}. {options[letter]}')
            parts.append("Can't tell")
    parts.append('')
    return '\n'.join(parts)


def copy_frames_ordered(
    item: dict,
    dest: Path,
    items_root: Path,
    resolve_image,
) -> list[str]:
    """Copy item frames as ``frame_01.png``, ``frame_02.png``, … Return names."""
    names: list[str] = []
    dest.mkdir(parents=True, exist_ok=True)
    scene_id = str(item.get('scene_id') or '')
    for i, raw in enumerate(item.get('image_paths') or [], start=1):
        src = resolve_image(str(raw), items_root, scene_id)
        if src is None:
            print(f'missing frame for {item["item_id"]}: {raw}', file=sys.stderr)
            continue
        suffix = src.suffix.lower() or '.png'
        name = f'frame_{i:02d}{suffix}'
        shutil.copy2(src, dest / name)
        names.append(name)
    return names


def _cap_per_construct(pool: list[dict], items_per_construct: int) -> list[dict]:
    return pool[: max(0, int(items_per_construct))]


def study_items_from_sample(
    items: list[dict],
    *,
    per_construct: int,
    seed: int,
    items_per_construct: int,
    max_items: int | None,
) -> list[dict]:
    """Same study sample as the human package, then keep N per construct."""
    review = _load_review()
    sampled = review.sample_items(items, per_construct, random.Random(seed))
    ordered: list[dict] = []
    for construct in review.CONSTRUCTS:
        ordered.extend(
            _cap_per_construct(list(sampled['study'][construct]), items_per_construct)
        )
    if max_items is not None:
        ordered = ordered[: max(0, int(max_items))]
    return ordered


def study_items_from_review_package(
    items_by_id: dict[str, dict],
    review_package: Path,
    *,
    items_per_construct: int,
    max_items: int | None,
) -> list[dict]:
    """Study items listed in a human ``manifest.json``, in construct then id order."""
    review = _load_review()
    manifest_path = (
        review_package / 'manifest.json' if review_package.is_dir() else review_package
    )
    manifest = json.loads(manifest_path.read_text())
    by_construct: dict[str, list[dict]] = {c: [] for c in review.CONSTRUCTS}
    for item_id, meta in (manifest.get('items') or {}).items():
        if meta.get('role') != 'review':
            continue
        item = items_by_id.get(item_id)
        if item is None:
            print(f'missing item {item_id} from review package', file=sys.stderr)
            continue
        construct = meta.get('construct') or item.get('construct')
        if construct in by_construct:
            by_construct[construct].append(item)
    ordered: list[dict] = []
    for construct in review.CONSTRUCTS:
        ordered.extend(_cap_per_construct(by_construct[construct], items_per_construct))
    if max_items is not None:
        ordered = ordered[: max(0, int(max_items))]
    return ordered


def write_item_folders(
    item: dict,
    kit_root: Path,
    items_root: Path,
    resolve_image,
) -> tuple[str, str]:
    """Write vision and blind folders. Return relative folder paths."""
    base = kit_root / 'items' / _safe(str(item['item_id']))
    vision = base / 'vision'
    blind = base / 'blind'
    if vision.exists():
        shutil.rmtree(vision)
    if blind.exists():
        shutil.rmtree(blind)
    vision.mkdir(parents=True)
    blind.mkdir(parents=True)
    copy_frames_ordered(item, vision, items_root, resolve_image)
    (vision / 'prompt.txt').write_text(build_prompt(item, blind=False), encoding='utf-8')
    (blind / 'prompt.txt').write_text(build_prompt(item, blind=True), encoding='utf-8')
    rel_vision = f'items/{_safe(str(item["item_id"]))}/vision'
    rel_blind = f'items/{_safe(str(item["item_id"]))}/blind'
    return rel_vision, rel_blind


def build_kit(
    items_root: Path,
    output: Path,
    *,
    per_construct: int = 25,
    seed: int = 0,
    items_per_construct: int = DEFAULT_ITEMS_PER_CONSTRUCT,
    max_items: int | None = None,
    review_package: Path | None = None,
) -> dict:
    """Write the chat kit. Return the kit manifest dict."""
    review = _load_review()
    if per_construct < 1:
        raise ValueError('per-construct must be at least 1')
    if items_per_construct < 1:
        raise ValueError('items-per-construct must be at least 1')
    all_items = review.load_items(items_root)
    by_id = {item['item_id']: item for item in all_items}
    if review_package is not None:
        selected = study_items_from_review_package(
            by_id,
            review_package,
            items_per_construct=items_per_construct,
            max_items=max_items,
        )
    else:
        selected = study_items_from_sample(
            all_items,
            per_construct=per_construct,
            seed=seed,
            items_per_construct=items_per_construct,
            max_items=max_items,
        )
    if not selected:
        raise ValueError('no study items selected for the VLM kit')

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    private = output / 'private' / 'items'
    private.mkdir(parents=True)
    batches = output / 'construct_batches'
    batches.mkdir(parents=True)

    results_rows: list[dict] = []
    answer_rows: list[dict] = []
    item_ids: list[str] = []
    by_construct: dict[str, list[str]] = defaultdict(list)

    for item in selected:
        item_id = str(item['item_id'])
        item_ids.append(item_id)
        (private / f'{_safe(item_id)}.json').write_text(
            json.dumps(item, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        rel_vision, rel_blind = write_item_folders(
            item, output, items_root, review.resolve_image
        )
        construct = item.get('construct') or ''
        by_construct[construct].append(rel_vision)
        kind = 'nav' if is_nav(item) else 'mc'
        for condition, folder in (('vision', rel_vision), ('blind', rel_blind)):
            results_rows.append(
                {
                    'item_id': item_id,
                    'construct': construct,
                    'condition': condition,
                    'folder': folder,
                    'model': '',
                    'raw_reply': '',
                    'comment': '',
                }
            )
        answer_rows.append(
            {
                'item_id': item_id,
                'construct': construct,
                'kind': kind,
                'gold': gold_for_answers(item),
            }
        )

    for construct, folders in by_construct.items():
        (batches / f'{construct}.txt').write_text(
            '\n'.join(folders) + '\n',
            encoding='utf-8',
        )

    results_path = output / 'results_template.csv'
    with results_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'item_id',
                'construct',
                'condition',
                'folder',
                'model',
                'raw_reply',
                'comment',
            ],
        )
        writer.writeheader()
        writer.writerows(results_rows)

    answers_path = output / 'answers.csv'
    with answers_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['item_id', 'construct', 'kind', 'gold'],
        )
        writer.writeheader()
        writer.writerows(answer_rows)

    manifest = {
        'seed': seed,
        'per_construct': per_construct,
        'items_per_construct': items_per_construct,
        'max_items': max_items,
        'review_package': str(review_package) if review_package else None,
        'items_root': str(items_root),
        'n_items': len(item_ids),
        'item_ids': item_ids,
        'results_template': 'results_template.csv',
        'answers': 'answers.csv',
        'chat_allows_cant_tell': True,
    }
    (output / 'kit_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    (output / 'README.txt').write_text(
        '\n'.join(
            [
                'VLM chat kit (manual pilot).',
                '',
                'Each construct has at least two items (see construct_batches/).',
                'In one chat, run both vision folders for that construct, then',
                'optionally the blind folders.',
                '',
                'Per item:',
                '  items/<item_id>/vision/  — upload frames in order, paste prompt.txt',
                '  items/<item_id>/blind/   — paste prompt.txt only (no frames)',
                '',
                'MC answers: A–D or Can\'t tell; optional comment on the next line.',
                'Nav answers: action list or Can\'t tell; optional comment.',
                '',
                'Fill results_template.csv (or copy to results.csv):',
                '  set model, raw_reply, and optional comment for each row.',
                '',
                'Do not open answers.csv or private/ while chatting.',
                '',
                'Score with:',
                '  python scripts/analyze_review.py --vlm-kit <this folder> \\',
                '    [--human-package <review_package>]',
                '',
            ]
        ),
        encoding='utf-8',
    )
    print(f'VLM kit: {len(item_ids)} items → {output}')
    for construct in review.CONSTRUCTS:
        n = sum(1 for item in selected if item.get('construct') == construct)
        if n:
            print(f'  {construct}: {n}')
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--items', type=Path, required=True, help='Items folder or one items JSON')
    parser.add_argument('--output', type=Path, required=True, help='Kit root')
    parser.add_argument(
        '--per-construct',
        type=int,
        default=25,
        help='Study sample size per construct (same as build_review_package)',
    )
    parser.add_argument('--seed', type=int, default=0, help='Same seed as the human review package')
    parser.add_argument(
        '--items-per-construct',
        type=int,
        default=DEFAULT_ITEMS_PER_CONSTRUCT,
        help=(
            'Keep this many study items per construct after sampling '
            f'(default {DEFAULT_ITEMS_PER_CONSTRUCT}, so each construct has '
            'two chat examples)'
        ),
    )
    parser.add_argument(
        '--max-items',
        type=int,
        default=None,
        help='Absolute cap on items (use 2 for a smoke test across constructs)',
    )
    parser.add_argument(
        '--review-package',
        type=Path,
        default=None,
        help='Existing human review package; reuse its study item ids',
    )
    args = parser.parse_args(argv)
    try:
        build_kit(
            args.items,
            args.output,
            per_construct=args.per_construct,
            seed=args.seed,
            items_per_construct=args.items_per_construct,
            max_items=args.max_items,
            review_package=args.review_package,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
