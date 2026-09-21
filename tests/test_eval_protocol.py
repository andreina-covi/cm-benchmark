"""Eval harness: format contract lives in SYSTEM_INSTRUCTION, not item text."""

from cm_benchmark.evaluation.protocol import (
    ACTION_VOCAB,
    SYSTEM_INSTRUCTION,
    wrap_item_for_eval,
)
from cm_benchmark.generation.constructs import select_template


def test_system_instruction_states_action_vocab_once():
    assert ACTION_VOCAB == (
        'move_ahead',
        'rotate_left',
        'rotate_right',
        'move_back',
    )
    for name in ACTION_VOCAB:
        assert name in SYSTEM_INSTRUCTION
    assert 'A, B, C, or D' in SYSTEM_INSTRUCTION


def test_class4_templates_do_not_repeat_action_vocab():
    for construct, n in (('route_knowledge', 1), ('survey_based_route_planning', 2)):
        for i in range(n):
            text = select_template(construct, index=i).lower()
            for name in ACTION_VOCAB:
                assert name not in text, f'{construct}[{i}] restates {name}'


def test_wrap_item_for_eval_does_not_mutate_question():
    item = {
        'construct': 'route_knowledge',
        'question': 'List, in order, the navigation actions you took to travel from the Chair to the Table.',
    }
    bundle = wrap_item_for_eval(item)
    assert bundle['question'] == item['question']
    assert 'move_ahead' not in bundle['question']
    assert 'move_ahead' in bundle['system']
    assert bundle['system'] == SYSTEM_INSTRUCTION
