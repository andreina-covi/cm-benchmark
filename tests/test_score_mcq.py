"""MCQ exact-match scoring and distractor (failure-mode) choice rates."""

from __future__ import annotations

import pytest

from cm_benchmark.evaluation.score import distractor_choice_rates, score_mcq


def _id_item(item_id):
    return {
        'item_id': item_id,
        'construct': 'invisible_displacement',
        'options': {'A': 'to your left', 'B': 'behind you', 'C': 'ahead of you', 'D': 'to your right'},
        'answer': 'A',
        'distractor_rationale': {
            'A': 'correct',
            'B': 'original_location',
            'C': 'ego_bank_filler',
            'D': 'opposite_direction',
        },
    }


def test_score_mcq_exact_letter_and_chosen_mode():
    item = _id_item('i1')
    assert score_mcq(item, 'a') == {'correct': True, 'chosen_mode': 'correct'}
    assert score_mcq(item, 'B') == {'correct': False, 'chosen_mode': 'original_location'}
    assert score_mcq(item, 'behind you') == {'correct': False, 'chosen_mode': None}


def test_perseveration_rate_against_chance():
    items = [_id_item(f'i{n}') for n in range(4)]
    choices = {'i0': 'A', 'i1': 'B', 'i2': 'B', 'i3': 'D'}
    row = distractor_choice_rates(items, choices)['invisible_displacement']
    assert (row['n'], row['n_correct'], row['n_wrong']) == (4, 1, 3)
    assert row['rates']['original_location'] == pytest.approx(2 / 3)
    assert row['chance']['original_location'] == pytest.approx(1 / 3)
