#!/usr/bin/env python3
"""Build one zip-ready review folder per reviewer.

Reads concise ``status=ok`` items from an items folder. Samples up to
``--per-construct`` items per construct, stratified across scenes. Each
sampled item is given to ``--reviews-per-item`` reviewers (default 2) by a
rotating pairing, so each reviewer sees a balanced mix of constructs.

Five extra check items, from different constructs, are added to every
reviewer. Two practice items, also from different constructs, open the
session and show the stored answer.

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
NAV_CONSTRUCTS = frozenset({
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


def concise_ok(items: list[dict]) -> list[dict]:
    """Concise items the generator accepted."""
    out = []
    for item in items:
        if item.get('question_style') == 'verbose':
            continue
        if item.get('status') != 'ok':
            continue
        if item.get('construct') not in CONSTRUCTS:
            continue
        out.append(item)
    return out


def is_nav(item: dict) -> bool:
    return item.get('construct') in NAV_CONSTRUCTS


def stored_display(item: dict) -> str:
    """Practice-screen text for the stored benchmark answer."""
    answer = item.get('answer')
    if is_nav(item):
        return '' if answer is None else str(answer)
    letter = '' if answer is None else str(answer)
    text = (item.get('options') or {}).get(letter, '')
    if letter and text:
        return f'{letter}. {text}'
    return letter


def review_gold(item: dict) -> str:
    """What a reviewer should submit. Navigation items are a yes/no judgment."""
    if is_nav(item):
        return 'yes'
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


def sample_items(
    items: list[dict],
    per_construct: int,
    rng: random.Random,
) -> dict:
    """Practice, shared checks, and the per-construct study sample."""
    groups: dict[str, list[dict]] = {c: [] for c in CONSTRUCTS}
    for item in concise_ok(items):
        groups[item['construct']].append(item)
    practice = reserve_diverse(groups, N_PRACTICE, rng)
    checks = reserve_diverse(groups, N_CHECKS, rng)
    study = {
        construct: take_stratified(groups[construct], per_construct, rng)
        for construct in CONSTRUCTS
    }
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


def resolve_image(raw: str, items_root: Path) -> Path | None:
    path = Path(str(raw))
    if path.is_file():
        return path
    if path.is_absolute():
        return None
    for base in (items_root, Path.cwd()):
        cand = base / path
        if cand.is_file():
            return cand
    return None


def copy_frames(item: dict, folder: Path, items_root: Path) -> list[str]:
    """Copy this item's frames. Return paths relative to the reviewer folder."""
    rels: list[str] = []
    dest = folder / 'frames' / _safe(str(item['item_id']))
    for i, raw in enumerate(item.get('image_paths') or []):
        src = resolve_image(str(raw), items_root)
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


def page_item(item: dict, role: str, frames: list[str]) -> dict:
    payload = {
        'item_id': item['item_id'],
        'construct': item['construct'],
        'role': role,
        'kind': 'nav' if is_nav(item) else 'mc',
        'question': item.get('question') or '',
        'frames': frames,
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
    return _PAGE.replace('__REVIEWER_ID__', reviewer_id).replace(
        '__PAYLOAD__', data
    ).replace('__FLAGS__', flags)


_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review __REVIEWER_ID__</title>
<style>
  :root { color-scheme: light; }
  body { margin: 0; font: 16px/1.45 system-ui, sans-serif; background: #f4f1ea; color: #1c1a16; }
  main { max-width: 1120px; margin: 0 auto; padding: 24px 20px 48px; }
  h1 { font-size: 1.4rem; margin: 0 0 8px; }
  h2 { font-size: 1.05rem; margin: 20px 0 8px; }
  p { margin: 8px 0; }
  button, .ghost { font: inherit; }
  button { background: #1c1a16; color: #fff; border: 0; border-radius: 6px; padding: 8px 14px; cursor: pointer; }
  button.secondary { background: #fff; color: #1c1a16; border: 1px solid #1c1a16; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 16px; }
  .progress { color: #5c564c; font-size: 0.9rem; }
  .card { background: #fff; border-radius: 10px; padding: 16px; margin-top: 12px; }
  .frame-wrap { display: flex; justify-content: center; background: #111; border-radius: 8px; min-height: 180px; }
  .frame-wrap img { max-width: 100%; max-height: 520px; object-fit: contain; }
  .frame-nav { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }
  .item-layout { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(280px, 0.8fr); gap: 12px; align-items: start; }
  .viewer { margin-top: 12px; }
  .question-panel { position: sticky; top: 12px; }
  .question-panel .card { margin-top: 0; }
  .thumbs { display: flex; gap: 6px; overflow-x: auto; margin-top: 8px; }
  .thumb { padding: 0; border: 3px solid transparent; border-radius: 4px; background: #111; line-height: 0; flex: 0 0 auto; }
  .thumb.current { border-color: #e6a817; }
  .thumb img { height: 64px; width: 96px; object-fit: cover; display: block; }
  input[type="range"] { width: 100%; margin: 10px 0 0; }
  @media (max-width: 800px) {
    .item-layout { grid-template-columns: 1fr; }
    .question-panel { position: static; }
  }
  .question { font-size: 1.15rem; white-space: pre-wrap; }
  label.choice { display: flex; gap: 8px; align-items: flex-start; padding: 6px 0; }
  textarea { width: 100%; min-height: 72px; font: inherit; padding: 8px; box-sizing: border-box; }
  .banner { background: #fff4cc; border-radius: 8px; padding: 10px 12px; }
  .answer { font-weight: 600; }
  ul { margin: 8px 0; padding-left: 1.2rem; }
  .top { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
</style>
</head>
<body>
<main id="app"></main>
<script>
const REVIEWER_ID = "__REVIEWER_ID__";
const PAYLOAD = __PAYLOAD__;
const FLAGS = __FLAGS__;
const KEY = "cm-review:" + REVIEWER_ID;

function blank() {
  return {answer: "", flags: [], comment: "", seconds: 0, frames_viewed: []};
}
function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
    return {index: saved.index || 0, items: saved.items || {}};
  } catch (e) {
    return {index: 0, items: {}};
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
  const header = ["reviewer_id", "item_id", "construct", "answer", "flags", "comment", "seconds", "role", "frames_viewed"];
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
    + '<div class="top"><h1>Spatial question review</h1><span class="progress">' + esc(REVIEWER_ID) + '</span></div>'
    + '<div class="card"><p>You will see first-person frames and one question per item. Frames are in time order, one at a time, labeled only "frame i of n".</p>'
    + '<p>Move with Previous and Next, the left and right arrow keys, the slider, or the thumbnail strip. The question and the answer choices stay on the right. On a narrow screen they sit under the frame. Which distinct frames you open is recorded.</p>'
    + '<p>For multiple choice, pick the option that matches the frames, or <strong>Can\'t tell</strong>.</p>'
    + '<p>For route and survey questions, do not list actions. Answer only: <strong>Is this question clear and answerable from the frames?</strong> Yes or no.</p>'
    + '<p>Also tick any problem that applies:</p><ul>'
    + '<li>ambiguous reference</li><li>unclear question</li>'
    + '<li>object not visible</li><li>more than one option looks correct</li></ul>'
    + '<p>A comment is optional. Time on each item is recorded. Two practice items come first and show the stored answer. Then the review items. Download the CSV when you finish, including if you stop early.</p>'
    + '<div class="row"><button id="start" type="button">Start</button>'
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
  if (label) label.textContent = "frame " + (frameIndex + 1) + " of " + n;
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
  return ''
    + '<div class="frame-wrap"><img id="frame-img" alt="" src="' + esc(item.frames[frameIndex]) + '"></div>'
    + '<div class="frame-nav"><button class="secondary" type="button" id="prev-frame">Previous</button>'
    + '<span id="frame-label">frame ' + (frameIndex + 1) + ' of ' + n + '</span>'
    + '<button class="secondary" type="button" id="next-frame">Next</button></div>'
    + '<input id="frame-slider" type="range" min="1" max="' + n + '" value="' + (frameIndex + 1) + '"'
    + (n < 2 ? " disabled" : "") + '>'
    + '<div class="thumbs" id="thumbs">' + thumbs + '</div>';
}
function questionBlock(item, answers, flags, prompt) {
  const rec = itemState(item.item_id);
  return '<div class="card"><p class="question">' + esc(item.question) + '</p>'
    + '<h2>' + prompt + '</h2>' + answers
    + '<h2>Problems</h2>' + flags
    + '<h2>Comment</h2><textarea id="comment">' + esc(rec.comment || "") + '</textarea>'
    + '</div>';
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
  if (item.kind === "nav") {
    ["yes", "no"].forEach(function (value) {
      const checked = rec.answer === value ? " checked" : "";
      answers += '<label class="choice"><input type="radio" name="answer" value="' + value + '"' + checked + '> '
        + (value === "yes" ? "Yes" : "No") + '</label>';
    });
  } else {
    (item.options || []).forEach(function (opt) {
      const checked = rec.answer === opt.letter ? " checked" : "";
      answers += '<label class="choice"><input type="radio" name="answer" value="' + esc(opt.letter) + '"' + checked + '> '
        + '<span><strong>' + esc(opt.letter) + '.</strong> ' + esc(opt.text) + '</span></label>';
    });
    const cant = rec.answer === "Can't tell" ? " checked" : "";
    answers += '<label class="choice"><input type="radio" name="answer" value="Can\'t tell"' + cant + '> Can\'t tell</label>';
  }
  let flags = '';
  FLAGS.forEach(function (flag) {
    const checked = (rec.flags || []).indexOf(flag) >= 0 ? " checked" : "";
    const text = flag.replace(/_/g, " ");
    flags += '<label class="choice"><input type="checkbox" name="flag" value="' + flag + '"' + checked + '> ' + esc(text) + '</label>';
  });
  const prompt = item.kind === "nav"
    ? "Is this question clear and answerable from the frames?"
    : "Your answer";
  const banner = practice
    ? ('<div class="banner"><p>Practice. The stored answer is shown. This one is not part of the scored review.</p>'
      + '<p class="answer">Stored answer: ' + esc(item.show_answer || "") + '</p></div>')
    : '';
  const nav = '<div class="row"><button class="secondary" type="button" id="back">Back</button>'
    + '<button type="button" id="next">' + (index === n ? "Done" : "Next") + '</button>'
    + '<button class="secondary" type="button" id="download">Download CSV</button></div>';
  return ''
    + '<div class="top"><h1>' + esc(label) + '</h1><span class="progress">' + esc(REVIEWER_ID) + '</span></div>'
    + banner
    + '<div class="item-layout">'
    + '<div class="card viewer">' + viewerHtml(item) + '</div>'
    + '<div class="question-panel">' + questionBlock(item, answers, flags, prompt) + nav + '</div>'
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
}
function bindChrome(index) {
  const start = document.getElementById("start");
  if (start) start.addEventListener("click", function () { show(1); });
  const download = document.getElementById("download");
  if (download) download.addEventListener("click", downloadCsv);
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
  document.querySelectorAll('input[name="answer"], input[name="flag"]').forEach(function (el) {
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
  if (!activeItem || (e.key !== "ArrowLeft" && e.key !== "ArrowRight")) return;
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


def build_package(
    items_root: Path,
    output: Path,
    *,
    per_construct: int = 25,
    n_reviewers: int = 10,
    reviews: int = 2,
    seed: int = 0,
) -> dict:
    """Write reviewer folders and ``manifest.json``. Return the manifest."""
    if per_construct < 1:
        raise ValueError('items per construct must be at least 1')
    if n_reviewers < 1:
        raise ValueError('need at least 1 reviewer')
    rng = random.Random(seed)
    sampled = sample_items(load_items(items_root), per_construct, rng)
    names = reviewer_names(n_reviewers)
    study_by_reviewer: dict[int, list[dict]] = {i: [] for i in range(n_reviewers)}
    pair_by_item: dict[str, list[str]] = {}
    for offset, construct in enumerate(CONSTRUCTS):
        study = sampled['study'][construct]
        slots = assign_slots(len(study), n_reviewers, reviews, offset)
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
            'kind': 'nav' if is_nav(item) else 'mc',
            'gold': review_gold(item),
            'stored_answer': None if item.get('answer') is None else str(item.get('answer')),
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
        'reviewers': n_reviewers,
        'reviews_per_item': reviews,
        'practice_item_ids': [item['item_id'] for item in sampled['practice']],
        'check_item_ids': [item['item_id'] for item in sampled['checks']],
        'items': manifest_items,
        'assignments': assignments,
        'reviewers_for_item': pair_by_item,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    _print_summary(sampled, names, assignments)
    return manifest


def _print_summary(sampled: dict, names: list[str], assignments: dict[str, list[str]]) -> None:
    print(f'practice: {len(sampled["practice"])}')
    check_constructs = [item['construct'] for item in sampled['checks']]
    print(f'checks: {len(sampled["checks"])} ({", ".join(check_constructs) or "none"})')
    for construct in CONSTRUCTS:
        n = len(sampled['study'][construct])
        if n:
            print(f'  {construct}: {n}')
    for name in names:
        n_study = sum(
            1 for item_id in assignments[name]
            if item_id not in {item['item_id'] for item in sampled['practice']}
            and item_id not in {item['item_id'] for item in sampled['checks']}
        )
        print(f'{name}: {n_study} study items + checks and practice → {name}/')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--items', type=Path, required=True, help='Items folder or one items JSON')
    parser.add_argument('--output', type=Path, required=True, help='Package root (one folder per reviewer)')
    parser.add_argument('--per-construct', type=int, default=25, help='Concise items sampled per construct')
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
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
