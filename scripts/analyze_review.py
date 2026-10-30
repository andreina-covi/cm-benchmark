#!/usr/bin/env python3
"""Merge reviewer CSVs and report agreement against the package manifest.

Per construct (study items only): human accuracy, nominal Krippendorff's
alpha, and the share of reviews with any problem flag. Check-item accuracy
is reported per reviewer. Items that both reviewers flagged, or that both
answered wrongly, are listed.

Class-4 (route / survey) answers are expanded action sequences scored with
the same simulator as models (success, SPL; nDTW/SDTW for route). The
yes/no ``answerable`` column is reported separately and does not gate
success. Practice rows are ignored. Gold / private items come from
``manifest.json`` and ``private/items/`` written by ``build_review_package.py``.

With ``--vlm-kit``, score a filled VLM chat-kit results CSV against
``answers.csv`` (MC letter match; class-4 via the route/survey scorers) and
optionally compare to human accuracy on the same items.

``--json-out`` writes a machine-readable summary for plotting
(``scripts/plot_review_report.py``). ``--txt-out`` saves the printed text.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / 'src'
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cm_benchmark.evaluation.score import (
    score_route_knowledge,
    score_survey_based_route_planning,
)
from cm_benchmark.generation.nav_graph import parse_nav_actions

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
MC_CHANCE = 0.25
BLIND_ABOVE_CHANCE = 0.40
VISION_NEAR_CEILING = 0.95
CANT_TELL = "Can't tell"
_MC_LETTER = re.compile(r'\b([A-Da-d])\b')
_CANT_TELL_RE = re.compile(
    r"^\s*(can\s*'?t\s+tell|cannot\s+tell|cant\s+tell)\s*[.\s]*$",
    re.IGNORECASE,
)


def krippendorff_alpha(units: list[list[str]]) -> float | None:
    """Nominal Krippendorff's alpha. Each unit is one item's rater labels."""
    values = [[v for v in unit if v is not None and v != ''] for unit in units]
    values = [unit for unit in values if len(unit) >= 2]
    if len(values) < 2:
        return None
    levels = sorted({v for unit in values for v in unit})
    if len(levels) == 1:
        return 1.0
    index = {level: i for i, level in enumerate(levels)}
    q = len(levels)
    coinc = [[0.0] * q for _ in range(q)]
    n = 0
    for unit in values:
        m = len(unit)
        n += m
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                coinc[index[unit[i]]][index[unit[j]]] += 1.0 / (m - 1)
    if n < 2:
        return None
    diag = sum(coinc[c][c] for c in range(q))
    do = 1.0 - diag / n
    nc = [sum(coinc[c]) for c in range(q)]
    expected_agreement = sum(nc[c] * (nc[c] - 1) for c in range(q)) / (n * (n - 1))
    de = 1.0 - expected_agreement
    if de == 0:
        return 1.0 if do == 0 else None
    return 1.0 - do / de


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    if 'items' not in data:
        raise ValueError(f'{path} has no items table')
    return data


def _parse_flags(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in str(text).replace(',', '|').split('|') if part.strip()]


def load_rows(paths: list[Path]) -> list[dict]:
    """CSV rows. A later file overrides the same reviewer and item."""
    merged: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for path in paths:
        with path.open(newline='') as handle:
            for raw in csv.DictReader(handle):
                reviewer = (raw.get('reviewer_id') or '').strip()
                item_id = (raw.get('item_id') or '').strip()
                if not reviewer or not item_id:
                    continue
                key = (reviewer, item_id)
                if key not in merged:
                    order.append(key)
                # Legacy CSVs stored yes/no in answer and free text in route.
                answer = (raw.get('answer') or '').strip()
                answerable = (raw.get('answerable') or '').strip()
                legacy_route = (raw.get('route') or '').strip()
                if (
                    not answerable
                    and answer.casefold() in {'yes', 'no'}
                    and (raw.get('construct') or '').strip() in NAV_CONSTRUCTS
                ):
                    answerable = answer
                    answer = legacy_route
                merged[key] = {
                    'reviewer_id': reviewer,
                    'item_id': item_id,
                    'construct': (raw.get('construct') or '').strip(),
                    'answer': answer,
                    'answerable': answerable,
                    'flags': _parse_flags(raw.get('flags') or ''),
                    'comment': raw.get('comment') or '',
                    'seconds': (raw.get('seconds') or '').strip(),
                    'role': (raw.get('role') or '').strip(),
                }
    return [merged[key] for key in order]


def _role(row: dict, manifest: dict) -> str:
    meta = (manifest.get('items') or {}).get(row['item_id']) or {}
    return meta.get('role') or row.get('role') or 'review'


def _load_package_item(package: Path | None, item_id: str) -> dict | None:
    if package is None:
        return None
    return _load_private_item(package, item_id)


def score_nav_answer(
    *,
    package: Path | None,
    item_id: str,
    construct: str,
    answer: str,
) -> dict | None:
    """Score a human/model action sequence; None when unparseable or missing item."""
    parsed = parse_nav_actions(answer)
    if parsed is None:
        return None
    item = _load_package_item(package, item_id)
    if item is None:
        return None
    if construct == 'survey_based_route_planning':
        return score_survey_based_route_planning(item, None, parsed)
    return score_route_knowledge(item, None, parsed)


def _correct_mc(meta: dict, answer: str) -> bool:
    gold = (meta.get('gold') or '').strip()
    given = (answer or '').strip()
    if not gold or not given:
        return False
    return given.casefold() == gold.casefold()


def _nav_success(
    meta: dict,
    row: dict,
    *,
    package: Path | None,
) -> bool | None:
    """True/False success, or None when the answer cannot be scored."""
    construct = meta.get('construct') or row.get('construct') or ''
    scored = score_nav_answer(
        package=package,
        item_id=row['item_id'],
        construct=construct,
        answer=row.get('answer') or '',
    )
    if scored is None:
        return None
    return bool(scored.get('success'))


def _row_correct(
    meta: dict,
    row: dict,
    *,
    package: Path | None,
) -> bool:
    construct = meta.get('construct') or row.get('construct') or ''
    if meta.get('kind') == 'nav' or construct in NAV_CONSTRUCTS:
        ok = _nav_success(meta, row, package=package)
        return bool(ok)
    return _correct_mc(meta, row.get('answer') or '')


def collect_human_analysis(
    manifest: dict,
    rows: list[dict],
    *,
    package: Path | None = None,
) -> dict:
    """Structured human-review summary (JSON-serializable)."""
    items = manifest.get('items') or {}
    checks = list(manifest.get('check_item_ids') or [])
    by_key = {(row['reviewer_id'], row['item_id']): row for row in rows}

    study_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if _role(row, manifest) != 'review':
            continue
        meta = items.get(row['item_id'])
        if not meta:
            continue
        study_rows[meta.get('construct') or row['construct']].append(row)

    constructs: list[dict] = []
    for construct in CONSTRUCTS:
        group = study_rows.get(construct) or []
        if not group and not any(
            meta.get('role') == 'review' and meta.get('construct') == construct
            for meta in items.values()
        ):
            continue
        is_nav = construct in NAV_CONSTRUCTS
        answered = [row for row in group if row.get('answer')]
        scored_rows: list[tuple[dict, dict]] = []
        n_correct = 0
        alpha_labels: dict[str, list[str]] = defaultdict(list)
        for row in answered:
            meta = items[row['item_id']]
            if is_nav:
                scored = score_nav_answer(
                    package=package,
                    item_id=row['item_id'],
                    construct=construct,
                    answer=row.get('answer') or '',
                )
                if scored is None:
                    continue
                scored_rows.append((row, scored))
                ok = bool(scored.get('success'))
                n_correct += int(ok)
                alpha_labels[row['item_id']].append('success' if ok else 'fail')
            else:
                ok = _correct_mc(meta, row['answer'])
                n_correct += int(ok)
                alpha_labels[row['item_id']].append(row['answer'])
        n_scored = len(scored_rows) if is_nav else len(answered)
        alpha = krippendorff_alpha(list(alpha_labels.values()))
        flagged = sum(1 for row in group if row['flags'])
        entry: dict = {
            'construct': construct,
            'kind': 'nav' if is_nav else 'mc',
            'n_reviews': len(group),
            'n_scored': n_scored,
            'n_correct': n_correct,
            'accuracy': (n_correct / n_scored) if n_scored else None,
            'alpha': alpha,
            'flag_rate': (flagged / len(group)) if group else None,
            'n_flagged': flagged,
            'mean_spl': None,
            'mean_ndtw': None,
            'mean_sdtw': None,
        }
        if is_nav and scored_rows:
            entry['mean_spl'] = sum(
                float(s['efficiency']) for _, s in scored_rows
            ) / len(scored_rows)
            entry['mean_ndtw'] = sum(
                float(s.get('ndtw') or 0.0) for _, s in scored_rows
            ) / len(scored_rows)
            entry['mean_sdtw'] = sum(
                float(s.get('sdtw') or 0.0) for _, s in scored_rows
            ) / len(scored_rows)
            if construct != 'route_knowledge':
                entry['mean_ndtw'] = None
                entry['mean_sdtw'] = None
        constructs.append(entry)

    answerable: list[dict] = []
    for construct in ('route_knowledge', 'survey_based_route_planning'):
        group = study_rows.get(construct) or []
        judged = [row for row in group if row.get('answerable')]
        n_yes = sum(1 for row in judged if row['answerable'].casefold() == 'yes')
        answerable.append(
            {
                'construct': construct,
                'n_judged': len(judged),
                'n_yes': n_yes,
                'yes_rate': (n_yes / len(judged)) if judged else None,
            }
        )

    check_accuracy: list[dict] = []
    reviewers = sorted({row['reviewer_id'] for row in rows})
    for reviewer in reviewers:
        hits = 0
        for item_id in checks:
            row = by_key.get((reviewer, item_id))
            meta = items.get(item_id) or {}
            if row and _row_correct(meta, row, package=package):
                hits += 1
        check_accuracy.append(
            {
                'reviewer_id': reviewer,
                'n_correct': hits,
                'n_checks': len(checks),
                'accuracy': (hits / len(checks)) if checks else None,
            }
        )

    problems: list[dict] = []
    by_item_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if _role(row, manifest) != 'review':
            continue
        if row['item_id'] in items:
            by_item_rows[row['item_id']].append(row)
    for item_id in sorted(by_item_rows):
        group = by_item_rows[item_id]
        if len(group) < 2:
            continue
        meta = items[item_id]
        both_flagged = all(row['flags'] for row in group)
        answered = [row for row in group if row.get('answer')]
        both_wrong = (
            len(answered) >= 2
            and len(answered) == len(group)
            and all(not _row_correct(meta, row, package=package) for row in answered)
        )
        if both_flagged or both_wrong:
            problems.append(
                {
                    'item_id': item_id,
                    'construct': meta.get('construct') or '',
                    'both_flagged': both_flagged,
                    'both_wrong': both_wrong,
                }
            )

    return {
        'kind': 'human',
        'constructs': constructs,
        'answerable': answerable,
        'check_accuracy': check_accuracy,
        'problems': problems,
        'n_check_items': len(checks),
    }


def format_human_analysis(data: dict) -> str:
    """Text table for a ``collect_human_analysis`` payload."""
    lines = [
        f'{"construct":<32} {"reviews":>8} {"accuracy":>16} {"alpha":>8} '
        f'{"flag_rate":>10} {"mean_spl":>10} {"mean_ndtw":>10} {"mean_sdtw":>10}',
    ]
    for entry in data.get('constructs') or []:
        n_scored = int(entry.get('n_scored') or 0)
        n_correct = int(entry.get('n_correct') or 0)
        if n_scored:
            accuracy = f'{n_correct / n_scored:.3f} ({n_correct}/{n_scored})'
        else:
            accuracy = 'n/a'
        alpha = entry.get('alpha')
        alpha_s = 'n/a' if alpha is None else f'{float(alpha):.3f}'
        flag_rate = entry.get('flag_rate')
        flag_s = 'n/a' if flag_rate is None else f'{float(flag_rate):.3f}'
        spl = entry.get('mean_spl')
        ndtw = entry.get('mean_ndtw')
        sdtw = entry.get('mean_sdtw')
        spl_s = 'n/a' if spl is None else f'{float(spl):.3f}'
        ndtw_s = 'n/a' if ndtw is None else f'{float(ndtw):.3f}'
        sdtw_s = 'n/a' if sdtw is None else f'{float(sdtw):.3f}'
        lines.append(
            f'{entry["construct"]:<32} {int(entry.get("n_reviews") or 0):>8} '
            f'{accuracy:>16} {alpha_s:>8} {flag_s:>10} '
            f'{spl_s:>10} {ndtw_s:>10} {sdtw_s:>10}'
        )

    lines.append('')
    lines.append('class-4 answerable (yes/no; not a success gate)')
    for entry in data.get('answerable') or []:
        judged = int(entry.get('n_judged') or 0)
        if not judged:
            lines.append(f'  {entry["construct"]}: n/a')
            continue
        n_yes = int(entry.get('n_yes') or 0)
        lines.append(
            f'  {entry["construct"]}: yes {n_yes}/{judged} '
            f'({n_yes / judged:.3f})'
        )

    lines.append('')
    lines.append('check-item accuracy')
    checks = int(data.get('n_check_items') or 0)
    check_rows = data.get('check_accuracy') or []
    if not checks:
        lines.append('  no check items')
    elif not check_rows:
        lines.append('  no reviews')
    else:
        width = max(len(row['reviewer_id']) for row in check_rows)
        for row in check_rows:
            lines.append(
                f'  {row["reviewer_id"]:<{width}}  '
                f'{int(row["n_correct"])}/{int(row["n_checks"])}'
            )

    lines.append('')
    lines.append('flagged or wrong by both reviewers')
    problems = data.get('problems') or []
    if not problems:
        lines.append('  none')
    else:
        lines.append(
            f'  {"item_id":<44} {"construct":<32} '
            f'{"both_flagged":>13} {"both_wrong":>11}'
        )
        for problem in problems:
            lines.append(
                f'  {problem["item_id"]:<44} {problem["construct"]:<32} '
                f'{str(problem["both_flagged"]):>13} {str(problem["both_wrong"]):>11}'
            )
    return '\n'.join(lines)


def analyze(
    manifest: dict,
    rows: list[dict],
    *,
    package: Path | None = None,
) -> str:
    """Text report for one merged set of reviews."""
    return format_human_analysis(
        collect_human_analysis(manifest, rows, package=package)
    )


def _csv_paths(package: Path, extra: list[Path]) -> list[Path]:
    paths = list(extra)
    if not paths:
        paths = sorted(
            path
            for path in package.rglob('*.csv')
            if path.name not in {'answers.csv', 'results_template.csv'}
        )
    return paths


def is_cant_tell(raw_reply: str) -> bool:
    """True when the first line is a Can't tell abstention."""
    text = (raw_reply or '').strip()
    if not text:
        return False
    first = text.splitlines()[0].strip()
    if first.casefold() == CANT_TELL.casefold():
        return True
    return bool(_CANT_TELL_RE.match(first))


def parse_mc_letter(raw_reply: str) -> str | None:
    """First option letter A–D in the model reply, else None.

    Ignores a Can't tell abstention. Uses the first non-empty line so an
    optional comment on later lines does not change the letter.
    """
    text = (raw_reply or '').strip()
    if not text or is_cant_tell(text):
        return None
    first = text.splitlines()[0].strip()
    if len(first) == 1 and first.upper() in 'ABCD':
        return first.upper()
    match = _MC_LETTER.search(first)
    return match.group(1).upper() if match else None


def load_answers_csv(path: Path) -> dict[str, dict]:
    """``item_id`` → {construct, kind, gold}."""
    out: dict[str, dict] = {}
    with path.open(newline='', encoding='utf-8') as handle:
        for raw in csv.DictReader(handle):
            item_id = (raw.get('item_id') or '').strip()
            if not item_id:
                continue
            out[item_id] = {
                'construct': (raw.get('construct') or '').strip(),
                'kind': (raw.get('kind') or '').strip(),
                'gold': (raw.get('gold') or '').strip(),
            }
    return out


def load_vlm_results(path: Path) -> list[dict]:
    """Rows from a filled results CSV (template or ``results.csv``)."""
    rows: list[dict] = []
    with path.open(newline='', encoding='utf-8') as handle:
        for raw in csv.DictReader(handle):
            item_id = (raw.get('item_id') or '').strip()
            if not item_id:
                continue
            rows.append(
                {
                    'item_id': item_id,
                    'construct': (raw.get('construct') or '').strip(),
                    'condition': (raw.get('condition') or '').strip().casefold(),
                    'folder': (raw.get('folder') or '').strip(),
                    'model': (raw.get('model') or '').strip(),
                    'raw_reply': (raw.get('raw_reply') or '').strip(),
                    'comment': (raw.get('comment') or '').strip(),
                }
            )
    return rows


def _load_private_item(kit: Path, item_id: str) -> dict | None:
    safe = item_id.replace('/', '_').replace('\\', '_')
    path = kit / 'private' / 'items' / f'{safe}.json'
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def score_vlm_reply(
    *,
    kit: Path,
    item_id: str,
    construct: str,
    kind: str,
    gold: str,
    raw_reply: str,
) -> tuple[bool, str, bool]:
    """Return ``(correct, parsed, cant_tell)``.

    ``Can't tell`` is never counted correct against gold. ``parsed`` is the
    letter, action string, or ``Can't tell``.
    """
    if not raw_reply:
        return False, '', False
    if is_cant_tell(raw_reply):
        return False, CANT_TELL, True
    if kind == 'nav' or construct in NAV_CONSTRUCTS:
        # Score only the first line so an optional comment does not break parse.
        first = raw_reply.splitlines()[0].strip()
        parsed = parse_nav_actions(first)
        if parsed is None:
            return False, '', False
        item = _load_private_item(kit, item_id)
        if item is None:
            gold_actions = parse_nav_actions(gold) or []
            ok = parsed == gold_actions
            return ok, ', '.join(parsed), False
        if construct == 'survey_based_route_planning':
            scored = score_survey_based_route_planning(item, None, parsed)
        else:
            scored = score_route_knowledge(item, None, parsed)
        # Validity is diagnostic only; binary correct = success (pose within goal).
        ok = bool(scored.get('success'))
        return bool(ok), ', '.join(parsed), False
    letter = parse_mc_letter(raw_reply)
    if letter is None:
        return False, '', False
    return letter.casefold() == gold.casefold(), letter, False


def human_accuracy_by_construct(
    human_package: Path,
    item_ids: set[str],
) -> dict[str, float]:
    """Study-item human accuracy per construct, restricted to ``item_ids``."""
    manifest_path = (
        human_package / 'manifest.json' if human_package.is_dir() else human_package
    )
    manifest = load_manifest(manifest_path)
    root = manifest_path.parent if manifest_path.is_file() else human_package
    rows = load_rows(_csv_paths(root, []))
    items = manifest.get('items') or {}
    tallies: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if _role(row, manifest) != 'review':
            continue
        if row['item_id'] not in item_ids:
            continue
        meta = items.get(row['item_id'])
        if not meta or not row['answer']:
            continue
        construct = meta.get('construct') or row['construct']
        tallies[construct].append(_row_correct(meta, row, package=root))
    return {
        construct: sum(hits) / len(hits)
        for construct, hits in tallies.items()
        if hits
    }


def collect_vlm_analysis(
    kit: Path,
    *,
    results_path: Path | None = None,
    answers_path: Path | None = None,
    human_package: Path | None = None,
) -> dict:
    """Structured VLM-kit summary (JSON-serializable)."""
    answers_file = answers_path or (kit / 'answers.csv')
    if results_path is None:
        cand = kit / 'results.csv'
        results_file = cand if cand.is_file() else kit / 'results_template.csv'
    else:
        results_file = results_path
    answers = load_answers_csv(answers_file)
    rows = load_vlm_results(results_file)
    if not rows:
        raise ValueError(f'no VLM result rows in {results_file}')

    # (construct, condition) → answered outcomes
    tallies: dict[tuple[str, str], list[bool]] = defaultdict(list)
    cant_tell: dict[tuple[str, str], int] = defaultdict(int)
    item_ids: set[str] = set()
    for row in rows:
        item_id = row['item_id']
        item_ids.add(item_id)
        meta = answers.get(item_id)
        if meta is None:
            continue
        if not row['raw_reply']:
            continue
        construct = meta['construct'] or row['construct']
        ok, _parsed, abstain = score_vlm_reply(
            kit=kit,
            item_id=item_id,
            construct=construct,
            kind=meta['kind'],
            gold=meta['gold'],
            raw_reply=row['raw_reply'],
        )
        key = (construct, row['condition'])
        tallies[key].append(ok)
        if abstain:
            cant_tell[key] += 1

    human = (
        human_accuracy_by_construct(human_package, item_ids)
        if human_package is not None
        else {}
    )

    results: list[dict] = []
    flags: list[str] = []
    for construct in CONSTRUCTS:
        for condition in ('vision', 'blind'):
            group = tallies.get((construct, condition)) or []
            if not group:
                continue
            n_ok = sum(1 for hit in group if hit)
            rate = n_ok / len(group)
            n_ct = cant_tell.get((construct, condition), 0)
            results.append(
                {
                    'construct': construct,
                    'condition': condition,
                    'n': len(group),
                    'n_correct': n_ok,
                    'accuracy': rate,
                    'cant_tell_rate': n_ct / len(group),
                    'n_cant_tell': n_ct,
                    'chance': None if construct in NAV_CONSTRUCTS else MC_CHANCE,
                    'human_accuracy': human.get(construct),
                }
            )
        blind = None
        vision = None
        for row in results:
            if row['construct'] != construct:
                continue
            if row['condition'] == 'blind':
                blind = row['accuracy']
            elif row['condition'] == 'vision':
                vision = row['accuracy']
        if (
            blind is not None
            and construct not in NAV_CONSTRUCTS
            and blind >= BLIND_ABOVE_CHANCE
        ):
            flags.append(
                f'{construct}: blind accuracy {blind:.3f} is well above chance '
                f'({MC_CHANCE:.2f}) — question may be solvable without images'
            )
        if vision is not None and vision >= VISION_NEAR_CEILING:
            flags.append(
                f'{construct}: vision accuracy {vision:.3f} is near ceiling — '
                'items may be too easy'
            )

    return {
        'kind': 'vlm',
        'results': results,
        'flags': flags,
        'mc_chance': MC_CHANCE,
    }


def format_vlm_analysis(data: dict) -> str:
    """Text table for a ``collect_vlm_analysis`` payload."""
    lines = [
        f'{"construct":<32} {"cond":>6} {"n":>4} {"accuracy":>16} '
        f'{"cant_tell":>10} {"chance":>8} {"human":>8}',
    ]
    for row in data.get('results') or []:
        n = int(row['n'])
        n_ok = int(row['n_correct'])
        n_ct = int(row['n_cant_tell'])
        ct = f'{n_ct / n:.3f} ({n_ct})'
        chance = 'n/a' if row.get('chance') is None else f'{float(row["chance"]):.2f}'
        human = row.get('human_accuracy')
        human_s = 'n/a' if human is None else f'{float(human):.3f}'
        acc = f'{n_ok / n:.3f} ({n_ok}/{n})'
        lines.append(
            f'{row["construct"]:<32} {row["condition"]:>6} {n:>4} {acc:>16} '
            f'{ct:>10} {chance:>8} {human_s:>8}'
        )
    lines.append('')
    lines.append('flags')
    flags = data.get('flags') or []
    if not flags:
        lines.append('  none')
    else:
        for flag in flags:
            lines.append(f'  {flag}')
    return '\n'.join(lines)


def analyze_vlm(
    kit: Path,
    *,
    results_path: Path | None = None,
    answers_path: Path | None = None,
    human_package: Path | None = None,
) -> str:
    """Per construct × condition accuracy, chance, human comparison, flags."""
    return format_vlm_analysis(
        collect_vlm_analysis(
            kit,
            results_path=results_path,
            answers_path=answers_path,
            human_package=human_package,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--package',
        type=Path,
        default=None,
        help='Human review folder that contains manifest.json',
    )
    parser.add_argument(
        '--vlm-kit',
        type=Path,
        default=None,
        help='VLM chat kit folder (answers.csv + results CSV)',
    )
    parser.add_argument(
        '--vlm-results',
        type=Path,
        default=None,
        help='Filled VLM results CSV (default: results.csv or results_template.csv in the kit)',
    )
    parser.add_argument(
        '--human-package',
        type=Path,
        default=None,
        help='Optional human review package for side-by-side accuracy on the same items',
    )
    parser.add_argument(
        'csvs',
        nargs='*',
        type=Path,
        help='Reviewer CSVs for --package. Default: every CSV under the package',
    )
    parser.add_argument(
        '--json-out',
        type=Path,
        default=None,
        help='Write structured analysis JSON (for scripts/plot_review_report.py)',
    )
    parser.add_argument(
        '--txt-out',
        type=Path,
        default=None,
        help='Also write the printed text report to this file',
    )
    args = parser.parse_args(argv)
    if args.vlm_kit is None and args.package is None:
        parser.error('provide --package and/or --vlm-kit')
    try:
        chunks: list[str] = []
        payload: dict = {'sections': []}
        if args.package is not None:
            manifest_path = (
                args.package / 'manifest.json' if args.package.is_dir() else args.package
            )
            manifest = load_manifest(manifest_path)
            package_root = manifest_path.parent if manifest_path.is_file() else args.package
            paths = _csv_paths(package_root, list(args.csvs))
            if not paths:
                print('no CSV files', file=sys.stderr)
                return 1
            human = collect_human_analysis(
                manifest, load_rows(paths), package=package_root
            )
            payload['sections'].append(human)
            chunks.append(format_human_analysis(human))
        if args.vlm_kit is not None:
            vlm = collect_vlm_analysis(
                args.vlm_kit,
                results_path=args.vlm_results,
                human_package=args.human_package or args.package,
            )
            payload['sections'].append(vlm)
            chunks.append(format_vlm_analysis(vlm))
        text = '\n\n'.join(chunks)
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
            print(f'Wrote {args.json_out}', file=sys.stderr)
        if args.txt_out is not None:
            args.txt_out.parent.mkdir(parents=True, exist_ok=True)
            args.txt_out.write_text(text + '\n', encoding='utf-8')
            print(f'Wrote {args.txt_out}', file=sys.stderr)
    except (OSError, ValueError, json.JSONDecodeError, csv.Error, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
