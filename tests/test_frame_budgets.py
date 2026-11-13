"""Image budgets: one frame list per budget (frames.budgets) for multi-frame items."""

from cm_benchmark.generation.config import get_generation_config
from cm_benchmark.generation.planner import (
    PlannedFact,
    _budget_frame_lists,
    _expand_frame_budgets,
)
from cm_benchmark.generation.templates import _stable_item_id


def test_short_span_is_one_list_for_every_budget():
    lists = _budget_frame_lists(range(10), [0, 9])
    assert lists == [(list(range(10)), list(get_generation_config().frames.budgets))]


def test_span_between_budgets_is_sampled_for_the_small_budget_only():
    lists = _budget_frame_lists(range(20), [0, 19])
    assert len(lists) == 2
    (small, small_budgets), (full, full_budgets) = lists
    assert small_budgets == [16] and full_budgets == [24, 32]
    assert len(small) == 16 and small[0] == 0 and small[-1] == 19
    max_gap = get_generation_config().frames.max_gap_steps
    assert all(b - a <= max_gap for a, b in zip(small, small[1:]))
    assert full == list(range(20))


def test_required_frames_are_kept_in_every_list():
    for shown, _budgets in _budget_frame_lists(range(40), [0, 13, 14, 39]):
        assert {0, 13, 14, 39} <= set(shown)


def test_span_too_long_for_the_gap_rule_offers_no_budget():
    # 200 steps with at most 5 between shown frames needs ~40 frames > 32.
    assert _budget_frame_lists(range(200), [0, 199]) == []


def _fact(**extra):
    return PlannedFact(
        construct='route_knowledge',
        status='ok',
        query_step=9,
        encoding_step=0,
        queried_object_id='Box|1',
        answer_label='m,r',
        image_paths=['a.png'],
        extra=dict(extra),
    )


def test_expand_makes_one_fact_per_frame_list_with_distinct_ids():
    variants = [
        {'budgets': [16], 'shown_frame_steps': [0, 9], 'image_paths': ['a', 'b'],
         'image_roles': ['r0', 'r1'], 'source_frame': 1, 'goal_frame': 2},
        {'budgets': [24, 32], 'shown_frame_steps': [0, 5, 9], 'image_paths': ['a', 'c', 'b'],
         'image_roles': ['r0', 'r2', 'r1'], 'source_frame': 1, 'goal_frame': 3},
    ]
    facts = _expand_frame_budgets({}, [_fact(frame_variants=variants)])
    assert [f.extra['frame_budgets'] for f in facts] == [[16], [24, 32]]
    assert [f.extra['frame_variant'] for f in facts] == ['f16', 'f24']
    assert facts[1].extra['goal_frame'] == 3 and facts[1].image_paths == ['a', 'c', 'b']
    assert all('frame_variants' not in f.extra for f in facts)
    ids = {_stable_item_id('ep', f, 'concise') for f in facts}
    groups = {_stable_item_id('ep', f, 'concise', with_variant=False) for f in facts}
    assert len(ids) == 2 and len(groups) == 1


def test_fact_with_no_budget_becomes_unsupported():
    facts = _expand_frame_budgets({}, [_fact(frame_variants=[])])
    assert [(f.status, f.reason) for f in facts] == [('unsupported', 'frames_exceed_budgets')]
