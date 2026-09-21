"""Eval harness: one system/output-format contract for every item.

Per-item ``question`` text stays construct-specific. Do not repeat this
contract inside templates or verbose preambles — the Model Runner prepends
``SYSTEM_INSTRUCTION`` once per call.
"""

from __future__ import annotations

from typing import Any, Optional

from cm_benchmark.generation.nav_graph import (
    NAV_ACTION_MOVE_AHEAD,
    NAV_ACTION_MOVE_BACK,
    NAV_ACTION_ROTATE_LEFT,
    NAV_ACTION_ROTATE_RIGHT,
)

# Collected-export names. Scorer aliases (MoveAhead, rotate right, …) still parse.
ACTION_VOCAB = (
    NAV_ACTION_MOVE_AHEAD,
    NAV_ACTION_ROTATE_LEFT,
    NAV_ACTION_ROTATE_RIGHT,
    NAV_ACTION_MOVE_BACK,
)

_ACTION_LIST = ', '.join(ACTION_VOCAB)

SYSTEM_INSTRUCTION = (
    'You are answering spatial-cognition questions from first-person views.\n'
    '\n'
    'Multiple-choice items: reply with exactly one option letter (A, B, C, or D).\n'
    '\n'
    'Navigation-action items (route knowledge and survey-based route planning): '
    'reply with an ordered sequence using only these action names: '
    f'{_ACTION_LIST}. Separate actions with commas or arrows. '
    'Do not add explanation.'
)


def wrap_item_for_eval(
    item: Optional[dict[str, Any]],
    *,
    system: str = SYSTEM_INSTRUCTION,
) -> dict[str, str]:
    """Bundle the global format contract with the item question.

    Does not mutate ``item['question']``. The Model Runner sends ``system``
    once (or as the system turn) and ``question`` as the user turn.
    """
    rec = item or {}
    return {
        'system': system,
        'question': rec.get('question') or '',
    }
