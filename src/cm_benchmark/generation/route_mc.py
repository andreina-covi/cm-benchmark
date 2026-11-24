"""Class-4 multiple choice: landmark-and-turn routes built from an action sequence.

A route is a start turn, then one leg per mid turn, then a last leg to the end
point: ``turn left, walk to the sofa, turn right, walk ahead, turn left, walk to
the end point``. A turn with no landmark is reached by ``walk ahead``.
Every option of an item has the same slots, so the correct route never stands
out by its length. Distractors change one thing, named by their tag:

Options are a 2x2 design (user decision 2026-11-24): two single changes X and Y
at different steps, and the options {correct, X, Y, X+Y}. Every option is then
the same total distance from the others, so "pick the option most like the
others" is at chance. A change is one of:

* ``mirrored_turn``: one turn flipped left <-> right ("sharply" is kept, so
  options never differ only by a turn's size);
* ``swapped_landmarks``: two turn slots trade landmarks (one may be unnamed).

X+Y is tagged with both reasons (``mirrored_turn+swapped_landmarks``).
"""

from __future__ import annotations

import math
import random
from itertools import combinations
from typing import Optional, Sequence

TURN_SIGN = {'rotate_left': -1, 'rotate_right': 1}
TURN_TEXT = {
    'left': 'turn left',
    'right': 'turn right',
    'sharp_left': 'turn sharply left',
    'sharp_right': 'turn sharply right',
    'around': 'turn around',
}
FLIP = {
    'left': 'right',
    'right': 'left',
    'sharp_left': 'sharp_right',
    'sharp_right': 'sharp_left',
    'around': 'around',
    'straight': 'straight',
}
END_LEG = 'walk to the end point'
UNNAMED_LEG = 'walk ahead'
ROUTE_DISTRACTOR_TAGS = ('mirrored_turn', 'swapped_landmarks')


def _norm_deg(deg: float) -> float:
    """Signed angle in (-180, 180]."""
    d = (float(deg) + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def turn_clusters(
    actions: Sequence[str],
    poses_xz: Sequence[Sequence[float]],
    *,
    rotation_deg: float,
    merge_m: float,
) -> tuple[float, list[dict]]:
    """``(start_turn_deg, mid_turns)`` from an action list and its replayed poses.

    ``poses_xz[i]`` is the pose before ``actions[i]`` (one more pose than actions).
    Turns before the first move form the start turn; turns after the last move
    do not change where the route ends and are dropped. Mid turns closer than
    ``merge_m`` of travel are summed. Each mid turn has ``net_deg`` (right is
    positive), ``points`` (the xz poses where it turned) and ``action_index``.
    """
    acts = list(actions)
    if len(poses_xz) < len(acts) + 1:
        raise ValueError('poses_xz needs one pose per action plus the start')
    travel = [0.0]
    for i in range(len(acts)):
        a, b = poses_xz[i], poses_xz[i + 1]
        travel.append(travel[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    moves = [i for i, a in enumerate(acts) if a not in TURN_SIGN]
    if not moves:
        return 0.0, []
    first_move, last_move = moves[0], moves[-1]

    runs: list[dict] = []
    i = 0
    while i < len(acts):
        if acts[i] not in TURN_SIGN:
            i += 1
            continue
        j = i
        net = 0
        while j < len(acts) and acts[j] in TURN_SIGN:
            net += TURN_SIGN[acts[j]]
            j += 1
        runs.append(
            {
                'action_index': i,
                'net_deg': net * float(rotation_deg),
                'travel_m': travel[i],
                'points': [tuple(poses_xz[k]) for k in range(i, j + 1)],
            }
        )
        i = j

    start_deg = sum(r['net_deg'] for r in runs if r['action_index'] < first_move)
    mids: list[dict] = []
    for run in runs:
        if run['action_index'] < first_move or run['action_index'] > last_move:
            continue
        if mids and run['travel_m'] - mids[-1]['travel_m'] < float(merge_m):
            prev = mids[-1]
            prev['net_deg'] += run['net_deg']
            prev['points'].extend(run['points'])
            prev['travel_m'] = run['travel_m']
            continue
        mids.append(dict(run, points=list(run['points'])))
    for m in mids:
        m['net_deg'] = _norm_deg(m['net_deg'])
    return _norm_deg(start_deg), mids


def turn_label(net_deg: float, labels_deg: Sequence[float]) -> Optional[str]:
    """Exact label for a net turn; None when it matches no label.

    ``labels_deg`` is ``(turn, sharp, around)``, e.g. ``(90, 135, 180)`` for
    SPOC's 45-degree steps (``class4.mc.turn_labels_deg``). Anything smaller than
    the plain turn is a heading correction (``straight``). Labels are exact, so
    no answer depends on judging a turn's size near a boundary.
    """
    a = abs(_norm_deg(net_deg))
    turn, sharp, around = (float(x) for x in labels_deg)
    tol = 1e-6
    if a < turn - tol:
        return 'straight'
    side = 'right' if net_deg > 0 else 'left'
    if abs(a - turn) <= tol:
        return side
    if abs(a - sharp) <= tol:
        return f'sharp_{side}'
    if abs(a - around) <= tol:
        return 'around'
    return None


def route_parts(
    start_turn: str, turns: Sequence[str], names: Sequence[Optional[str]]
) -> list[str]:
    """Ordered step phrases of one route; a None name is ``walk ahead``."""
    parts: list[str] = []
    if start_turn != 'straight':
        parts.append(TURN_TEXT[start_turn])
    for turn, name in zip(turns, names):
        parts.append(f'walk to the {name}' if name else UNNAMED_LEG)
        parts.append(TURN_TEXT[turn])
    parts.append(END_LEG)
    return parts


def format_route(start_turn: str, turns: Sequence[str], names: Sequence[Optional[str]]) -> str:
    text = ', '.join(route_parts(start_turn, turns, names))
    return text[:1].upper() + text[1:]


def route_distractors(
    start_turn: str,
    turns: Sequence[str],
    names: Sequence[Optional[str]],
    *,
    key: str,
) -> tuple[str, list[tuple[str, str]]]:
    """``(correct_text, [(tag, text) x 3])`` in the 2x2 design, or ``(correct, [])``.

    X is a mirrored turn; Y is a landmark swap when one exists, else a second
    mirrored turn at another step. Which turn and which pair are drawn with a
    seeded RNG on ``key``. Empty when the route has no two independent changes.
    """
    turns = list(turns)
    names = list(names)
    correct = format_route(start_turn, turns, names)
    rng = random.Random(key)
    all_turns = [start_turn, *turns]
    mirrors = [('mirrored_turn', i) for i, t in enumerate(all_turns) if FLIP.get(t, t) != t]
    # Unnamed slots may trade with named ones (the landmark moves to another
    # turn); two unnamed slots never form a pair.
    swaps = [
        ('swapped_landmarks', (i, j))
        for i, j in combinations(range(len(names)), 2)
        if names[i] != names[j] and (names[i] or names[j])
    ]
    if mirrors and swaps:
        pairs = [(x, y) for x in mirrors for y in swaps]
    else:
        pairs = list(combinations(mirrors, 2))
    if not pairs:
        return correct, []
    changes = rng.choice(pairs)

    def apply(chosen) -> str:
        seq, labels = list(all_turns), list(names)
        for kind, where in chosen:
            if kind == 'mirrored_turn':
                seq[where] = FLIP[seq[where]]
            else:
                i, j = where
                labels[i], labels[j] = labels[j], labels[i]
        return format_route(seq[0], seq[1:], labels)

    x, y = changes
    return correct, [
        (x[0], apply([x])),
        (y[0], apply([y])),
        (f'{x[0]}+{y[0]}', apply([x, y])),
    ]


def consensus_pick(texts: Sequence[str], key: str) -> Optional[int]:
    """Index of the option closest to all others (step-wise mismatches).

    A text-only shortcut: if distractors each change one step of the answer,
    the answer is the "average" option. Ties are broken with a seeded RNG on
    ``key``, so in the 2x2 design the pick is right 1 time in 4.
    """
    parts = [t.split(', ') for t in texts]
    totals = []
    for i, a in enumerate(parts):
        total = 0
        for j, b in enumerate(parts):
            if i == j:
                continue
            total += sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
        totals.append(total)
    if not totals:
        return None
    best = min(totals)
    tied = [i for i, t in enumerate(totals) if t == best]
    return random.Random(f'consensus:{key}').choice(tied)


def point_box_gap_xz(point: Sequence[float], amin, amax) -> float:
    """Floor-plane distance from an xz point to an AABB (0 inside)."""
    x, z = float(point[0]), float(point[1])
    dx = max(float(amin[0]) - x, 0.0, x - float(amax[0]))
    dz = max(float(amin[2]) - z, 0.0, z - float(amax[2]))
    return math.hypot(dx, dz)
