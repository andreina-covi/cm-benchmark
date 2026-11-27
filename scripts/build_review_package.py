#!/usr/bin/env python3
"""Build one zip-ready review folder per reviewer.

Reads concise ``status=ok`` multiple-choice items from an items folder
(every construct, class 4 included). The class-4 free-action diagnostic
subset has no options and is evaluated on models only, so it is never
sampled; nor are templates in ``review.model_only_templates`` (SWM count). Samples up to ``--per-construct`` items per construct, stratified
across scenes. Class-4 (route / survey) study items are further capped so
each reviewer sees at most ``review.class4_per_reviewer`` of them (default 5;
CLI override): they carry up to 32 frames each.
Each sampled item is given to ``--reviews-per-item`` reviewers (default 2)
by a rotating pairing, so each reviewer sees a balanced mix of constructs.

Five extra check items, from different constructs, are added to every
reviewer. Two practice items, also from different constructs, open the
session and show the stored answer.

Every item is answered by picking an option or ``Can't tell``; the CSV
``answer`` column stores the letter.

Each reviewer folder contains ``index.html`` and the frames that item
uses (the marked copies, when the item points at them). Gold answers for
the real items stay in ``manifest.json`` at the output root, not in the
reviewer folders.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

CONSTRUCTS = (
    'egocentric_encoding',
    'allocentric_encoding',
    'spatial_working_memory',
    'invisible_displacement',
    'spatial_updating',
    'perspective_taking',
    'route_knowledge',
    'survey_based_route_planning',
)
CLASS4_CONSTRUCTS = frozenset({
    'route_knowledge',
    'survey_based_route_planning',
})
N_CHECKS = 5
N_PRACTICE = 2
FLAGS = (
    'ambiguous_reference',
    'unclear_question',
    'object_not_visible',
    'multiple_correct',
)


def load_items(root: Path) -> list[dict]:
    """Every item dict under ``root`` (a folder or one JSON file)."""
    paths = sorted(root.rglob('items_*.json')) if root.is_dir() else [root]
    items: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        blob = json.loads(path.read_text())
        rows = blob if isinstance(blob, list) else blob.get('items') or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item_id = str(row.get('item_id') or '')
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            items.append(row)
    return items


def _model_only_templates() -> frozenset[str]:
    """``review.model_only_templates``: templates never shown to reviewers."""
    from cm_benchmark.generation.config import get_generation_config

    return frozenset(get_generation_config().review.model_only_templates)


# Item context copied into the manifest so analyze_review can split results.
MANIFEST_CONTEXT_FIELDS = (
    'rotation_since_last_seen',
    'template_mode',
    'condition',
    'n_frames_shown',
    'n_turn_events',
    'total_rotation_deg',
    'displacement_m',
)


def concise_ok(items: list[dict]) -> list[dict]:
    """Concise items the generator accepted, minus model-only templates."""
    model_only = _model_only_templates()
    out = []
    for item in items:
        if item.get('question_style') == 'verbose':
            continue
        if item.get('status') != 'ok':
            continue
        if item.get('construct') not in CONSTRUCTS:
            continue
        if not item.get('options'):
            # Class-4 free-action diagnostic subset: models only.
            continue
        if (item.get('context') or {}).get('template_mode') in model_only:
            # e.g. SWM count: scored on models only (review.model_only_templates).
            continue
        out.append(item)
    return out


def stored_display(item: dict) -> str:
    """Practice-screen text for the stored benchmark answer."""
    answer = item.get('answer')
    letter = '' if answer is None else str(answer)
    text = (item.get('options') or {}).get(letter, '')
    if letter and text:
        return f'{letter}. {text}'
    return letter


def review_gold(item: dict) -> str:
    """Gold for analysis: the option letter."""
    return '' if item.get('answer') is None else str(item.get('answer'))


def take_stratified(pool: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Up to ``k`` items, round-robin across scenes."""
    if k <= 0 or not pool:
        return []
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for item in pool:
        by_scene[str(item.get('scene_id') or '')].append(item)
    scenes = list(by_scene)
    rng.shuffle(scenes)
    for scene in scenes:
        rng.shuffle(by_scene[scene])
    picked: list[dict] = []
    while len(picked) < k and any(by_scene.values()):
        for scene in scenes:
            if by_scene[scene] and len(picked) < k:
                picked.append(by_scene[scene].pop())
    return picked


def _drop(pool: list[dict], item_id: str) -> list[dict]:
    return [item for item in pool if item.get('item_id') != item_id]


def reserve_diverse(
    groups: dict[str, list[dict]],
    k: int,
    rng: random.Random,
) -> list[dict]:
    """One item from each of ``k`` constructs, preferring larger pools."""
    constructs = [c for c in CONSTRUCTS if groups.get(c)]
    rng.shuffle(constructs)
    constructs.sort(key=lambda c: len(groups[c]), reverse=True)
    picked: list[dict] = []
    for construct in constructs:
        if len(picked) >= k:
            break
        one = take_stratified(groups[construct], 1, rng)
        if not one:
            continue
        picked.append(one[0])
        groups[construct] = _drop(groups[construct], one[0]['item_id'])
    return picked


def class4_study_total(
    *,
    n_reviewers: int,
    reviews: int,
    class4_per_reviewer: int,
) -> int:
    """Max class-4 study items across route + survey combined.

    Chosen so a single balanced assignment uses exactly
    ``n_reviewers * class4_per_reviewer`` reviewer slots.
    """
    if n_reviewers < 1 or reviews < 1 or class4_per_reviewer < 0:
        return 0
    return (int(n_reviewers) * int(class4_per_reviewer)) // int(reviews)


def _take_class4_study(
    groups: dict[str, list[dict]],
    total: int,
    rng: random.Random,
) -> dict[str, list[dict]]:
    """Round-robin class-4 items across route and survey up to ``total``."""
    out: dict[str, list[dict]] = {c: [] for c in CLASS4_CONSTRUCTS}
    if total <= 0:
        return out
    pools = {
        c: take_stratified(groups[c], len(groups[c]), rng) for c in CLASS4_CONSTRUCTS
    }
    order = [c for c in CONSTRUCTS if c in CLASS4_CONSTRUCTS]
    taken = 0
    while taken < total and any(pools.values()):
        for construct in order:
            if taken >= total:
                break
            if pools[construct]:
                out[construct].append(pools[construct].pop(0))
                groups[construct] = _drop(groups[construct], out[construct][-1]['item_id'])
                taken += 1
    return out


def sample_items(
    items: list[dict],
    per_construct: int,
    rng: random.Random,
    *,
    class4_total: Optional[int] = None,
) -> dict:
    """Practice, shared checks, and the per-construct study sample."""
    groups: dict[str, list[dict]] = {c: [] for c in CONSTRUCTS}
    for item in concise_ok(items):
        groups[item['construct']].append(item)
    practice = reserve_diverse(groups, N_PRACTICE, rng)
    checks = reserve_diverse(groups, N_CHECKS, rng)
    study: dict[str, list[dict]] = {}
    if class4_total is not None:
        nav_study = _take_class4_study(groups, int(class4_total), rng)
        for construct in CONSTRUCTS:
            if construct in CLASS4_CONSTRUCTS:
                study[construct] = nav_study.get(construct) or []
            else:
                study[construct] = take_stratified(groups[construct], per_construct, rng)
    else:
        for construct in CONSTRUCTS:
            study[construct] = take_stratified(groups[construct], per_construct, rng)
    return {'practice': practice, 'checks': checks, 'study': study}


def circle_pairs(n_reviewers: int, n_items: int, offset: int = 0) -> list[tuple[int, int]]:
    """``n_items`` rotating reviewer pairs. Each round uses disjoint pairs."""
    if n_items <= 0:
        return []
    if n_reviewers < 2:
        raise ValueError('need at least 2 reviewers to pair items')
    players: list[int | None] = list(range(n_reviewers))
    if n_reviewers % 2 == 1:
        players.append(None)
    n = len(players)
    rounds = max(n - 1, 1)
    for _ in range(offset % rounds):
        players = [players[0]] + [players[-1]] + players[1:-1]
    pairs: list[tuple[int, int]] = []
    while len(pairs) < n_items:
        for i in range(n // 2):
            a, b = players[i], players[n - 1 - i]
            if a is None or b is None or a == b:
                continue
            pairs.append((int(a), int(b)))
            if len(pairs) == n_items:
                return pairs
        players = [players[0]] + [players[-1]] + players[1:-1]
    return pairs


def assign_slots(
    n_items: int,
    n_reviewers: int,
    reviews: int,
    offset: int = 0,
) -> list[tuple[int, ...]]:
    """Reviewer indexes for each item. Two reviews use the circle schedule."""
    if reviews < 1:
        raise ValueError('reviews-per-item must be at least 1')
    if reviews > n_reviewers:
        raise ValueError('reviews-per-item cannot exceed the number of reviewers')
    if reviews == 2:
        return [tuple(pair) for pair in circle_pairs(n_reviewers, n_items, offset)]
    if reviews == 1:
        return [((i + offset) % n_reviewers,) for i in range(n_items)]
    load = [0] * n_reviewers
    assigned: list[tuple[int, ...]] = []
    for i in range(n_items):
        order = sorted(
            range(n_reviewers),
            key=lambda r: (load[r], (r + offset + i) % n_reviewers),
        )
        chosen = tuple(order[:reviews])
        for reviewer in chosen:
            load[reviewer] += 1
        assigned.append(chosen)
    return assigned


def reviewer_names(n: int) -> list[str]:
    width = max(2, len(str(n)))
    return [f'reviewer_{i + 1:0{width}d}' for i in range(n)]


def weave(study: list[dict], checks: list[dict]) -> list[dict]:
    """Insert check items at even gaps through the study list."""
    if not checks:
        return list(study)
    if not study:
        return list(checks)
    out: list[dict] = []
    every = max(1, len(study) // len(checks))
    check_i = 0
    for i, item in enumerate(study):
        out.append(item)
        if (i + 1) % every == 0 and check_i < len(checks):
            out.append(checks[check_i])
            check_i += 1
    while check_i < len(checks):
        out.append(checks[check_i])
        check_i += 1
    return out


def presentation_order(
    study_items: list[dict],
    checks: list[dict],
    practice: list[dict],
    rng: random.Random,
) -> list[dict]:
    """Practice, then constructs interleaved, with checks spread through."""
    by_construct: dict[str, list[dict]] = {c: [] for c in CONSTRUCTS}
    for item in study_items:
        by_construct[item['construct']].append(item)
    columns: list[list[dict]] = []
    for construct in CONSTRUCTS:
        column = list(by_construct[construct])
        rng.shuffle(column)
        if column:
            columns.append(column)
    merged: list[dict] = []
    while any(columns):
        for column in columns:
            if column:
                merged.append(column.pop(0))
    check_copy = list(checks)
    rng.shuffle(check_copy)
    return list(practice) + weave(merged, check_copy)


def _safe(item_id: str) -> str:
    return item_id.replace('/', '_').replace('\\', '_')


def resolve_image(raw: str, items_root: Path, scene_id: str = '') -> Path | None:
    """Resolve a stored frame path without grabbing another scene's basename."""
    path = Path(str(raw))
    if path.is_absolute():
        return path if path.is_file() else None
    if path.parent != Path('.'):
        cand = (items_root / path).resolve()
        return cand if cand.is_file() else None
    name = path.name
    if not name:
        return None
    roots = []
    if scene_id:
        scene_dir = items_root / scene_id
        if scene_dir.is_dir():
            roots.append(scene_dir)
    roots.append(items_root)
    for root in roots:
        hits = sorted(p for p in root.rglob(name) if p.is_file())
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return None
    return None


def copy_frames(item: dict, folder: Path, items_root: Path) -> list[str]:
    """Copy this item's frames. Return paths relative to the reviewer folder."""
    rels: list[str] = []
    dest = folder / 'frames' / _safe(str(item['item_id']))
    for i, raw in enumerate(item.get('image_paths') or []):
        src = resolve_image(str(raw), items_root, str(item.get('scene_id') or ''))
        name = f'{i}_{Path(str(raw)).name or "frame.png"}'
        rel = f'frames/{_safe(str(item["item_id"]))}/{name}'
        if src is None:
            print(f'missing frame for {item["item_id"]}: {raw}', file=sys.stderr)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / name)
        rels.append(rel)
    return rels


def _option_rows(item: dict) -> list[dict]:
    options = item.get('options') or {}
    if not isinstance(options, dict):
        return []
    return [{'letter': str(k), 'text': str(v)} for k, v in sorted(options.items())]


# Questions that count frames back from the last one ("N frames before the last frame").
_FRAMES_AGO_CONSTRUCTS = frozenset({'spatial_working_memory', 'spatial_updating'})


def _frame_labels_for_review(item: dict, n_frames: int) -> list[str]:
    """Labels under the viewer: ``frame i of n``.

    Where the question counts frames back, each label also says how far back
    that frame is (``k frames before the last``, ``last frame``), in the question's words.
    Simulator steps, roles and object names are left out.
    """
    labels: list[str] = []
    for i in range(n_frames):
        label = f'frame {i + 1} of {n_frames}'
        in_sequence = (item.get('context') or {}).get('condition') == 'in_sequence'
        if item.get('construct') in _FRAMES_AGO_CONSTRUCTS or in_sequence:
            back = n_frames - 1 - i
            ago = 'last frame' if back == 0 else f"{back} frame{'' if back == 1 else 's'} before the last"
            label = f'{label} · {ago}'
        labels.append(label)
    return labels


def page_item(item: dict, role: str, frames: list[str]) -> dict:
    payload = {
        'item_id': item['item_id'],
        'construct': item['construct'],
        'role': role,
        'question': item.get('question') or '',
        'frames': frames,
        'frame_labels': _frame_labels_for_review(item, len(frames)),
        'options': _option_rows(item),
        'show_answer': None,
    }
    if role == 'practice':
        payload['show_answer'] = stored_display(item)
    return payload


def render_html(reviewer_id: str, payload: list[dict]) -> str:
    """One self-contained review page."""
    data = json.dumps(payload).replace('<', '\\u003c')
    flags = json.dumps(list(FLAGS))
    return (
        _PAGE.replace('__REVIEWER_ID__', reviewer_id)
        .replace('__PAYLOAD__', data)
        .replace('__FLAGS__', flags)
    )


_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review __REVIEWER_ID__</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f3f1ec; --surface: #fff; --ink: #1d1b18; --muted: #6b665e; --line: #e2ddd3;
    --accent: #2f5d8a; --accent-soft: #e8eff6; --warn-soft: #fff4d6; --frame: #0f0f10;
    --radius: 12px;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }
  main { max-width: 1600px; margin: 0 auto; padding: 20px 20px 48px; }
  h1 { font-size: 1.25rem; margin: 0; }
  h2 { font-size: 0.8rem; margin: 18px 0 6px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
  p { margin: 8px 0; }
  button, .ghost { font: inherit; }
  button { background: var(--ink); color: #fff; border: 1px solid var(--ink); border-radius: 8px; padding: 8px 16px; cursor: pointer; }
  button:hover { opacity: 0.88; }
  button.secondary { background: var(--surface); color: var(--ink); border-color: var(--line); }
  button.secondary:hover { border-color: var(--ink); opacity: 1; }
  button:focus-visible, input:focus-visible, textarea:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 16px; }
  .row .spacer { flex: 1; }
  .construct-tag { color: var(--muted); font-size: 0.7rem; font-weight: 500; letter-spacing: 0.05em; margin-left: 10px;
    padding: 2px 8px; border: 1px solid var(--line); border-radius: 999px; vertical-align: middle; }
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 18px; margin-top: 12px; }
  .top { display: flex; justify-content: space-between; gap: 16px; align-items: center; }
  .item-nav { padding: 12px 14px; }
  .item-nav .nav-head { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; justify-content: space-between; margin-bottom: 10px; }
  .item-nav .count { color: var(--muted); font-size: 0.95rem; font-variant-numeric: tabular-nums; }
  .item-nav .goto { display: flex; gap: 6px; align-items: center; font-size: 0.95rem; }
  .item-nav .goto input { width: 4.5rem; font: inherit; padding: 5px 8px; border: 1px solid var(--line); border-radius: 6px; }
  .item-nav .goto button { padding: 5px 12px; }
  .chips { display: flex; flex-wrap: wrap; gap: 5px; }
  .chips .sep { width: 1px; background: var(--line); margin: 0 6px; align-self: stretch; }
  .chip { position: relative; min-width: 34px; height: 30px; padding: 0 6px; border-radius: 6px; font-size: 0.85rem;
    font-variant-numeric: tabular-nums; background: var(--surface); color: var(--ink); border: 1px solid #c9c3b8; }
  .chip:hover { border-color: var(--ink); opacity: 1; }
  .chip.done { background: #2e7d4f; border-color: #2e7d4f; color: #fff; }
  .chip.current { outline: 3px solid var(--accent); outline-offset: 1px; }
  .chip.noted::after { content: ""; position: absolute; top: 3px; right: 3px; width: 6px; height: 6px; border-radius: 50%; background: #e0a100; }
  .legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 8px; color: var(--muted); font-size: 0.8rem; }
  .legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -1px; margin-right: 5px; border: 1px solid #c9c3b8; background: var(--surface); }
  .legend i.done { background: #2e7d4f; border-color: #2e7d4f; }
  .legend i.current { outline: 2px solid var(--accent); }
  .legend i.noted { border: 0; width: 8px; height: 8px; border-radius: 50%; background: #e0a100; vertical-align: 1px; }
  .frame-wrap { display: flex; justify-content: center; background: var(--frame); border-radius: 8px; overflow: hidden; }
  /* Frames are 396x224: scale up to the column width so small objects stay legible. */
  .frame-wrap img { width: 100%; height: auto; max-height: 80vh; object-fit: contain; display: block; }
  .frame-nav { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-top: 10px; }
  #frame-label { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 0.95rem; }
  .item-layout { display: grid; grid-template-columns: minmax(0, 2.4fr) minmax(320px, 0.85fr); gap: 16px; align-items: start; }
  .viewer { margin-top: 12px; padding: 12px; }
  .question-panel { position: sticky; top: 12px; margin-top: 12px; }
  .question-panel .card { margin-top: 0; }
  .thumbs { display: flex; gap: 6px; overflow-x: auto; margin-top: 10px; padding-bottom: 4px; }
  .thumb { padding: 0; border: 3px solid transparent; border-radius: 6px; background: var(--frame); line-height: 0; flex: 0 0 auto; opacity: 0.7; }
  .thumb:hover { opacity: 1; }
  .thumb.current { border-color: var(--accent); opacity: 1; }
  .thumb img { height: 56px; width: 99px; object-fit: cover; display: block; border-radius: 3px; }
  input[type="range"] { width: 100%; margin: 12px 0 0; accent-color: var(--accent); }
  @media (max-width: 800px) {
    .item-layout { grid-template-columns: 1fr; }
    .question-panel { position: static; }
    main { padding: 12px 12px 32px; }
  }
  .question { font-size: 1.2rem; line-height: 1.45; font-weight: 500; white-space: pre-wrap; margin: 0;
    padding: 2px 0 2px 14px; border-left: 4px solid var(--accent); }
  label.choice { display: flex; gap: 10px; align-items: center; padding: 9px 12px; margin: 6px 0;
    border: 1px solid var(--line); border-radius: 8px; cursor: pointer; }
  label.choice:hover { border-color: var(--accent); }
  label.choice:has(input:checked) { border-color: var(--accent); background: var(--accent-soft); }
  label.choice input { accent-color: var(--accent); margin: 0; }
  .flags { display: flex; flex-wrap: wrap; gap: 6px; }
  .flags label.choice { margin: 0; padding: 5px 10px; font-size: 0.9rem; border-radius: 999px; }
  textarea { width: 100%; min-height: 64px; font: inherit; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; resize: vertical; }
  .banner { background: var(--warn-soft); border: 1px solid #f0dfa6; border-radius: var(--radius); padding: 10px 14px; margin-top: 12px; }
  .banner p { margin: 4px 0; }
  .answer { font-weight: 600; }
  ul { margin: 8px 0; padding-left: 1.2rem; }
</style>
</head>
<body>
<main id="app"></main>
<script>
const REVIEWER_ID = "__REVIEWER_ID__";
const PAYLOAD = __PAYLOAD__;
const FLAGS = __FLAGS__;
const KEY = "cm-review:" + REVIEWER_ID;
const CONSTRUCT_INITIALS = {
  egocentric_encoding: "EE",
  allocentric_encoding: "AE",
  spatial_working_memory: "SWM",
  invisible_displacement: "ID",
  spatial_updating: "SU",
  perspective_taking: "PT",
  route_knowledge: "RK",
  survey_based_route_planning: "SR"
};

function blank() {
  return {
    answer: "",
    flags: [],
    comment: "",
    seconds: 0,
    frames_viewed: []
  };
}
function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
    return {index: saved.index || 0, last: saved.last || saved.index || 0, items: saved.items || {}};
  } catch (e) {
    return {index: 0, last: 0, items: {}};
  }
}
let state = loadState();
let activeId = null;
let activeItem = null;
let timerStart = null;
let frameIndex = 0;

function itemState(id) {
  if (!state.items[id]) state.items[id] = blank();
  return state.items[id];
}
function save() {
  localStorage.setItem(KEY, JSON.stringify(state));
}
function pauseTimer() {
  if (activeId == null || timerStart == null) return;
  const rec = itemState(activeId);
  rec.seconds = Math.round((Number(rec.seconds) + (Date.now() - timerStart) / 1000) * 10) / 10;
  timerStart = null;
}
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
    return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c];
  });
}
function csvEscape(s) {
  s = String(s == null ? "" : s);
  if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}
function downloadCsv() {
  pauseTimer();
  if (activeId) timerStart = Date.now();
  save();
  const header = [
    "reviewer_id", "item_id", "construct", "answer",
    "flags", "comment", "seconds", "role", "frames_viewed"
  ];
  const lines = [header.join(",")];
  PAYLOAD.forEach(function (item) {
    const rec = state.items[item.item_id] || blank();
    const flags = (rec.flags || []).filter(function (flag) { return FLAGS.indexOf(flag) >= 0; });
    const seen = [];
    (rec.frames_viewed || []).forEach(function (n) {
      if (seen.indexOf(n) < 0) seen.push(n);
    });
    const row = [
      REVIEWER_ID, item.item_id, item.construct, rec.answer || "",
      flags.join("|"), rec.comment || "", rec.seconds || 0, item.role, seen.join("|")
    ].map(csvEscape);
    lines.push(row.join(","));
  });
  const blob = new Blob([lines.join("\n") + "\n"], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = REVIEWER_ID + ".csv";
  a.click();
}
function instructionsHtml() {
  return ''
    + '<div class="top"><h1>Spatial question review</h1></div>'
    + '<div class="card"><p>You will see first-person frames and one question per item. Frames are in time order, one at a time, labeled "frame i of n". Questions name the frames they refer to by these numbers; "the last frame" is frame n. Where a question counts frames back, the label also says how far back each frame is.</p>'
    + '<p>The question is shown next to the frames. Read it, then look through the frames it refers to. When an item has more than one frame, move with Previous and Next, the left and right arrow keys, the slider, or the thumbnail strip. Which distinct frames you open is recorded. On a narrow screen the question sits under the frames.</p>'
    + '<p>For multiple choice, pick the option that matches the frames, or <strong>Can\'t tell</strong>.</p>'
    + '<p>Some questions compare two frames: which way you face in the last frame compared with an earlier one ("the same way", "turned to your left", "turned to your right", "turned around"). Look at every frame in between; one turn alone may not give the answer. Count questions ask how many different objects of a kind you saw: an object that leaves the view and comes back is still one object.</p>'
    + '<p>Route questions ask which route takes you from your position in frame i to your position in frame j. Each option lists the turns in order and the landmark where each turn happens; "walk ahead" means that turn has no landmark. "Turn sharply" is a 135° turn, "turn around" 180°. The options differ in a turn\'s direction or in where a landmark comes.</p>'
    + '<p>Also tick any problem that applies:</p><ul>'
    + '<li>ambiguous reference</li><li>unclear question</li>'
    + '<li>object not visible</li><li>more than one option looks correct</li></ul>'
    + '<p>Your answers are saved in this browser. Reopening the page returns to the last item you had open. The numbered strip above each item shows which items are answered (green), which are not (white) and which have a problem or comment (yellow dot); click a number, or type it in <strong>Go to item</strong>, to jump to any item.</p>'
    + '<p>A comment is optional. Time on each item is recorded. Two practice items come first and show the stored answer. Then the review items. Download the CSV when you finish, including if you stop early.</p>'
    + '<div class="row"><button id="start" type="button">' + (state.last > 0 ? 'Continue from item ' + state.last : 'Start') + '</button>'
    + '<button id="download" class="secondary" type="button">Download CSV</button></div></div>';
}
function noteFrame(item) {
  if (!item || !item.frames.length) return;
  const rec = itemState(item.item_id);
  if (!rec.frames_viewed) rec.frames_viewed = [];
  const n = frameIndex + 1;
  if (rec.frames_viewed.indexOf(n) < 0) rec.frames_viewed.push(n);
  save();
}
function setFrame(index0) {
  if (!activeItem || !activeItem.frames.length) return;
  const n = activeItem.frames.length;
  frameIndex = Math.max(0, Math.min(n - 1, index0));
  const img = document.getElementById("frame-img");
  if (img) img.src = activeItem.frames[frameIndex];
  const label = document.getElementById("frame-label");
  if (label && n > 1) {
    const labels = activeItem.frame_labels || [];
    label.textContent = labels[frameIndex]
      || ("frame " + (frameIndex + 1) + " of " + n);
  }
  const slider = document.getElementById("frame-slider");
  if (slider && document.activeElement !== slider) slider.value = String(frameIndex + 1);
  document.querySelectorAll(".thumb").forEach(function (el, i) {
    if (i === frameIndex) el.classList.add("current");
    else el.classList.remove("current");
  });
  const strip = document.getElementById("thumbs");
  const current = strip && strip.querySelector(".thumb.current");
  if (strip && current) {
    strip.scrollLeft = current.offsetLeft - strip.clientWidth / 2 + current.offsetWidth / 2;
  }
  noteFrame(activeItem);
}
function viewerHtml(item) {
  const n = item.frames.length;
  if (!n) return '<p>No frames for this item.</p>';
  if (frameIndex < 0 || frameIndex >= n) frameIndex = 0;
  let thumbs = "";
  item.frames.forEach(function (src, i) {
    thumbs += '<button type="button" class="thumb' + (i === frameIndex ? " current" : "") + '" data-frame="' + i + '">'
      + '<img alt="" src="' + esc(src) + '"></button>';
  });
  const image = '<div class="frame-wrap"><img id="frame-img" alt="" src="' + esc(item.frames[frameIndex]) + '"></div>';
  // A single image needs no label, navigation, slider or thumbnails.
  if (n < 2) return image;
  const labels = item.frame_labels || [];
  const startLabel = labels[frameIndex] || ('frame ' + (frameIndex + 1) + ' of ' + n);
  return image
    + '<div class="frame-nav"><button class="secondary" type="button" id="prev-frame">&#8249; Previous</button>'
    + '<span id="frame-label">' + esc(startLabel) + '</span>'
    + '<button class="secondary" type="button" id="next-frame">Next &#8250;</button></div>'
    + '<input id="frame-slider" type="range" min="1" max="' + n + '" value="' + (frameIndex + 1) + '">'
    + '<div class="thumbs" id="thumbs">' + thumbs + '</div>';
}
function questionBlock(item, answers, flags, prompt) {
  const rec = itemState(item.item_id);
  return '<div class="card"><p class="question">' + esc(item.question) + '</p>'
    + '<h2>' + prompt + '</h2>' + answers
    + '<h2>Problems</h2><div class="flags">' + flags + '</div>'
    + '<h2>Comment</h2><textarea id="comment">' + esc(rec.comment || "") + '</textarea>'
    + '</div>';
}
function isAnswered(item) {
  const rec = state.items[item.item_id];
  if (!rec) return false;
  return !!rec.answer;
}
function hasNote(item) {
  const rec = state.items[item.item_id];
  return !!(rec && ((rec.flags || []).length || (rec.comment || "").trim()));
}
function chipLabel(item, i) {
  if (item.role !== "practice") return String(i);
  let p = 0;
  for (let k = 0; k < i; k++) if (PAYLOAD[k].role === "practice") p++;
  return "P" + p;
}
function navigatorHtml(index) {
  const n = PAYLOAD.length;
  let done = 0;
  let chips = "";
  PAYLOAD.forEach(function (item, k) {
    const i = k + 1;
    const answered = isAnswered(item);
    if (answered) done++;
    if (k > 0 && PAYLOAD[k - 1].role === "practice" && item.role !== "practice") chips += '<span class="sep"></span>';
    const cls = "chip" + (answered ? " done" : "") + (i === index ? " current" : "") + (hasNote(item) ? " noted" : "");
    const state = (answered ? "answered" : "not answered") + (hasNote(item) ? ", has a problem flag or comment" : "");
    chips += '<button type="button" class="' + cls + '" data-goto="' + i + '" title="Item ' + i + ': ' + state + '"'
      + (i === index ? ' aria-current="true"' : '') + '>' + chipLabel(item, i) + '</button>';
  });
  return '<div class="nav-head"><span class="count"><strong>' + done + '</strong> of ' + n + ' answered</span>'
    + '<span class="goto"><label for="goto-n">Go to item</label>'
    + '<input id="goto-n" type="number" min="1" max="' + n + '" value="' + index + '">'
    + '<button type="button" class="secondary" id="goto-btn">Go</button></span></div>'
    + '<div class="chips">' + chips + '</div>'
    + '<div class="legend"><span><i class="done"></i>answered</span><span><i></i>not answered</span>'
    + '<span><i class="current"></i>current</span><span><i class="noted"></i>problem or comment</span></div>';
}
function goTo(index, target) {
  const t = parseInt(target, 10);
  if (!(t >= 1 && t <= PAYLOAD.length)) return;
  if (index >= 1) readForm(PAYLOAD[index - 1]);
  show(t);
}
function bindNavigator(index) {
  document.querySelectorAll("[data-goto]").forEach(function (el) {
    el.addEventListener("click", function () { goTo(index, el.getAttribute("data-goto")); });
  });
  const input = document.getElementById("goto-n");
  const btn = document.getElementById("goto-btn");
  if (btn && input) btn.addEventListener("click", function () { goTo(index, input.value); });
  if (input) input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); goTo(index, input.value); }
  });
}
function refreshNavigator() {
  const box = document.getElementById("item-nav");
  if (!box || !state.index) return;
  box.innerHTML = navigatorHtml(state.index);
  bindNavigator(state.index);
}
function itemHtml(item, index) {
  const rec = itemState(item.item_id);
  const n = PAYLOAD.length;
  const practice = item.role === "practice";
  const practiceN = PAYLOAD.filter(function (row) { return row.role === "practice"; }).length;
  const label = practice
    ? ('Practice ' + index + ' of ' + practiceN)
    : ('Item ' + index + ' of ' + n);
  let answers = '';
  (item.options || []).forEach(function (opt) {
    const checked = rec.answer === opt.letter ? " checked" : "";
    answers += '<label class="choice"><input type="radio" name="answer" value="' + esc(opt.letter) + '"' + checked + '> '
      + '<span><strong>' + esc(opt.letter) + '.</strong> ' + esc(opt.text) + '</span></label>';
  });
  const cant = rec.answer === "Can't tell" ? " checked" : "";
  answers += '<label class="choice"><input type="radio" name="answer" value="Can\'t tell"' + cant + '> Can\'t tell</label>';
  let flags = '';
  FLAGS.forEach(function (flag) {
    const checked = (rec.flags || []).indexOf(flag) >= 0 ? " checked" : "";
    const text = flag.replace(/_/g, " ");
    flags += '<label class="choice"><input type="checkbox" name="flag" value="' + flag + '"' + checked + '> ' + esc(text) + '</label>';
  });
  const prompt = "Your answer";
  let banner = '';
  if (practice) {
    banner = '<div class="banner"><p>Practice. This one is not part of the scored review.</p>'
      + '<p class="answer">Stored answer: ' + esc(item.show_answer || "") + '</p></div>';
  }
  const nav = '<div class="row"><button class="secondary" type="button" id="back">Back</button>'
    + '<button class="secondary" type="button" id="download">Download CSV</button>'
    + '<span class="spacer"></span>'
    + '<button type="button" id="next">' + (index === n ? "Done" : "Next item") + '</button></div>';
  const panel = questionBlock(item, answers, flags, prompt) + nav;
  const initials = CONSTRUCT_INITIALS[item.construct] || "";
  const tag = initials
    ? '<span class="construct-tag" title="' + esc(item.construct) + '">' + esc(initials) + '</span>'
    : '';
  return ''
    + '<div class="top"><h1>' + esc(label) + tag + '</h1>'
    + '<button class="secondary" type="button" id="instructions">Instructions</button></div>'
    + '<div class="card item-nav" id="item-nav">' + navigatorHtml(index) + '</div>'
    + banner
    + '<div class="item-layout">'
    + '<div class="card viewer">' + viewerHtml(item) + '</div>'
    + '<div class="question-panel">' + panel + '</div>'
    + '</div>';
}
function readForm(item) {
  const rec = itemState(item.item_id);
  if (!document.querySelector('input[name="answer"]')) {
    save();
    return;
  }
  const selected = document.querySelector('input[name="answer"]:checked');
  rec.answer = selected ? selected.value : "";
  rec.flags = Array.prototype.map.call(
    document.querySelectorAll('input[name="flag"]:checked'),
    function (el) { return el.value; }
  );
  const comment = document.getElementById("comment");
  rec.comment = comment ? comment.value : "";
  save();
  refreshNavigator();
}
function bindChrome(index) {
  const start = document.getElementById("start");
  if (start) start.addEventListener("click", function () { show(state.last || 1); });
  const download = document.getElementById("download");
  if (download) download.addEventListener("click", downloadCsv);
  const intro = document.getElementById("instructions");
  if (intro) intro.addEventListener("click", function () {
    readForm(PAYLOAD[index - 1]);
    show(0);
  });
  bindNavigator(index);
  const back = document.getElementById("back");
  if (back) back.addEventListener("click", function () {
    readForm(PAYLOAD[index - 1]);
    show(index - 1);
  });
  const nextBtn = document.getElementById("next");
  if (nextBtn) nextBtn.addEventListener("click", function () {
    readForm(PAYLOAD[index - 1]);
    if (index >= PAYLOAD.length) downloadCsv();
    else show(index + 1);
  });
}
function show(index) {
  pauseTimer();
  if (index < 0) index = 0;
  if (index > PAYLOAD.length) index = PAYLOAD.length;
  state.index = index;
  if (index > 0) state.last = index;
  save();
  frameIndex = 0;
  const root = document.getElementById("app");
  if (index <= 0) {
    activeId = null;
    activeItem = null;
    root.innerHTML = instructionsHtml();
    bindChrome(index);
    return;
  }
  const item = PAYLOAD[index - 1];
  activeId = item.item_id;
  activeItem = item;
  timerStart = Date.now();
  root.innerHTML = itemHtml(item, index);
  noteFrame(item);
  bindItem(item, index);
  bindChrome(index);
}
function bindItem(item, index) {
  document.querySelectorAll(
    'input[name="answer"], input[name="flag"]'
  ).forEach(function (el) {
    el.addEventListener("change", function () { readForm(item); });
  });
  const comment = document.getElementById("comment");
  if (comment) comment.addEventListener("input", function () { readForm(item); });
  const prev = document.getElementById("prev-frame");
  const next = document.getElementById("next-frame");
  if (prev) prev.addEventListener("click", function () { setFrame(frameIndex - 1); });
  if (next) next.addEventListener("click", function () { setFrame(frameIndex + 1); });
  const slider = document.getElementById("frame-slider");
  if (slider) slider.addEventListener("input", function () { setFrame(Number(slider.value) - 1); });
  document.querySelectorAll(".thumb").forEach(function (el) {
    el.addEventListener("click", function () { setFrame(Number(el.getAttribute("data-frame"))); });
  });
}
document.addEventListener("keydown", function (e) {
  if (!activeItem || activeItem.frames.length < 2 || (e.key !== "ArrowLeft" && e.key !== "ArrowRight")) return;
  const el = e.target;
  if (el && (el.tagName === "TEXTAREA" || el.tagName === "SELECT")) return;
  if (el && el.tagName === "INPUT" && el.type !== "button") return;
  e.preventDefault();
  setFrame(frameIndex + (e.key === "ArrowRight" ? 1 : -1));
});
window.addEventListener("beforeunload", function () { pauseTimer(); save(); });
setInterval(function () {
  if (activeId == null) return;
  pauseTimer();
  timerStart = Date.now();
  save();
}, 5000);
show(state.index || 0);
</script>
</body>
</html>
"""


def _default_review_settings() -> int:
    """``review.class4_per_reviewer`` from generation.yaml."""
    try:
        from cm_benchmark.generation.config import get_generation_config

        return int(get_generation_config().review.class4_per_reviewer)
    except Exception:
        return 5


def build_package(
    items_root: Path,
    output: Path,
    *,
    per_construct: int = 25,
    n_reviewers: int = 10,
    reviews: int = 2,
    seed: int = 0,
    class4_per_reviewer: Optional[int] = None,
) -> dict:
    """Write reviewer folders and ``manifest.json``. Return the manifest."""
    if per_construct < 1:
        raise ValueError('items per construct must be at least 1')
    if n_reviewers < 1:
        raise ValueError('need at least 1 reviewer')
    cfg_cap = _default_review_settings()
    c4_cap = cfg_cap if class4_per_reviewer is None else int(class4_per_reviewer)
    if c4_cap < 0:
        raise ValueError('class4-per-reviewer must be >= 0')
    class4_total = class4_study_total(
        n_reviewers=n_reviewers,
        reviews=reviews,
        class4_per_reviewer=c4_cap,
    )
    rng = random.Random(seed)
    sampled = sample_items(
        load_items(items_root),
        per_construct,
        rng,
        class4_total=class4_total,
    )
    names = reviewer_names(n_reviewers)
    study_by_reviewer: dict[int, list[dict]] = {i: [] for i in range(n_reviewers)}
    pair_by_item: dict[str, list[str]] = {}
    # Assign all class-4 study items in one balanced round so each reviewer
    # gets exactly class4_per_reviewer of them (when the pool is full).
    nav_study = [
        item
        for construct in CONSTRUCTS
        if construct in CLASS4_CONSTRUCTS
        for item in sampled['study'][construct]
    ]
    nav_slots = assign_slots(len(nav_study), n_reviewers, reviews, offset=0)
    for item, slot in zip(nav_study, nav_slots):
        pair_by_item[item['item_id']] = [names[i] for i in slot]
        for reviewer in slot:
            study_by_reviewer[reviewer].append(item)
    for offset, construct in enumerate(CONSTRUCTS):
        if construct in CLASS4_CONSTRUCTS:
            continue
        study = sampled['study'][construct]
        slots = assign_slots(len(study), n_reviewers, reviews, offset + 1)
        for item, slot in zip(study, slots):
            pair_by_item[item['item_id']] = [names[i] for i in slot]
            for reviewer in slot:
                study_by_reviewer[reviewer].append(item)

    output.mkdir(parents=True, exist_ok=True)
    manifest_items: dict[str, dict] = {}

    def remember(item: dict, role: str) -> None:
        manifest_items[item['item_id']] = {
            'construct': item['construct'],
            'scene_id': item.get('scene_id') or '',
            'role': role,
            'kind': 'mc',
            'gold': review_gold(item),
            'stored_answer': None if item.get('answer') is None else str(item.get('answer')),
            **{f: (item.get('context') or {}).get(f) for f in MANIFEST_CONTEXT_FIELDS},
        }

    for item in sampled['practice']:
        remember(item, 'practice')
    for item in sampled['checks']:
        remember(item, 'check')
    for construct in CONSTRUCTS:
        for item in sampled['study'][construct]:
            remember(item, 'review')

    assignments: dict[str, list[str]] = {}
    for index, name in enumerate(names):
        folder = output / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        order_rng = random.Random((seed + 1) * 1000 + index)
        ordered = presentation_order(
            study_by_reviewer[index],
            sampled['checks'],
            sampled['practice'],
            order_rng,
        )
        payload = []
        for item in ordered:
            role = manifest_items[item['item_id']]['role']
            frames = copy_frames(item, folder, items_root)
            payload.append(page_item(item, role, frames))
        (folder / 'index.html').write_text(render_html(name, payload))
        assignments[name] = [item['item_id'] for item in ordered]

    manifest = {
        'seed': seed,
        'per_construct': per_construct,
        'class4_per_reviewer': c4_cap,
        'class4_total': class4_total,
        'reviewers': n_reviewers,
        'reviews_per_item': reviews,
        'practice_item_ids': [item['item_id'] for item in sampled['practice']],
        'check_item_ids': [item['item_id'] for item in sampled['checks']],
        'items': manifest_items,
        'assignments': assignments,
        'reviewers_for_item': pair_by_item,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    _print_summary(sampled, names, assignments, class4_per_reviewer=c4_cap)
    return manifest


def _print_summary(
    sampled: dict,
    names: list[str],
    assignments: dict[str, list[str]],
    *,
    class4_per_reviewer: int,
) -> None:
    print(f'practice: {len(sampled["practice"])}')
    check_constructs = [item['construct'] for item in sampled['checks']]
    print(f'checks: {len(sampled["checks"])} ({", ".join(check_constructs) or "none"})')
    for construct in CONSTRUCTS:
        n = len(sampled['study'][construct])
        if n:
            print(f'  {construct}: {n}')
    print(f'class-4 cap per reviewer: {class4_per_reviewer}')
    held = {item['item_id'] for item in sampled['practice']} | {
        item['item_id'] for item in sampled['checks']
    }
    study_by_id = {
        item['item_id']: item
        for construct in CONSTRUCTS
        for item in sampled['study'][construct]
    }
    for name in names:
        study_ids = [i for i in assignments[name] if i not in held]
        n_class4 = sum(
            1 for i in study_ids if study_by_id.get(i, {}).get('construct') in CLASS4_CONSTRUCTS
        )
        print(
            f'{name}: {len(study_ids)} study ({n_class4} class-4) + checks and practice → {name}/'
        )


def main(argv: list[str] | None = None) -> int:
    cfg_cap = _default_review_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--items', type=Path, required=True, help='Items folder or one items JSON')
    parser.add_argument('--output', type=Path, required=True, help='Package root (one folder per reviewer)')
    parser.add_argument('--per-construct', type=int, default=25, help='Concise items sampled per construct')
    parser.add_argument(
        '--class4-per-reviewer',
        type=int,
        default=cfg_cap,
        metavar='N',
        help=(
            'Max route+survey study items each reviewer sees '
            f'(default: {cfg_cap} from configs/generation.yaml review.class4_per_reviewer). '
            'Pass a larger N to include more class-4 items when the pool has them.'
        ),
    )
    parser.add_argument('--reviewers', type=int, default=10)
    parser.add_argument('--reviews-per-item', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    try:
        build_package(
            args.items,
            args.output,
            per_construct=args.per_construct,
            n_reviewers=args.reviewers,
            reviews=args.reviews_per_item,
            seed=args.seed,
            class4_per_reviewer=args.class4_per_reviewer,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
