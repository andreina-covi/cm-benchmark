"""Tests for first-draft taxonomy Q&A generation."""

from copy import deepcopy
from pathlib import Path

import pytest

from cm_benchmark.generator.ai2thor_nav_generator import Ai2ThorNavGenerator
from cm_benchmark.generation.constructs import select_template
from cm_benchmark.generation.pipeline import draft_items_for_episode
from cm_benchmark.generation.planner import plan_episode
from cm_benchmark.generation.templates import _core_question, build_verbose_preamble

FIXTURES = Path(__file__).parent / 'fixtures'
NAV_CSV = FIXTURES / 'navigation_tiny.csv'
OBJ_CSV = FIXTURES / 'objects_tiny.csv'
EPISODE_DIR = FIXTURES / 'episode_tiny'


@pytest.fixture
def tiny_episode(tmp_path):
    gen = Ai2ThorNavGenerator(
        path_navigation=str(NAV_CSV),
        path_objects=str(OBJ_CSV),
        output_path=str(tmp_path),
        output_filename='episode.json',
    )
    return gen.collect_episode_data(extra_data={'scene': 'TinyScene'})


@pytest.fixture
def folder_episode(tmp_path):
    gen = Ai2ThorNavGenerator(
        csv_path_folder=str(EPISODE_DIR),
        output_path=str(tmp_path),
        output_filename='folder_episode.json',
    )
    return gen.collect_episode_data(extra_data={'scene': 'ignore'})


@pytest.fixture
def delayed_episode(tiny_episode):
    """Tiny episode extended so SWM can exercise a multi-step delay."""
    episode = deepcopy(tiny_episode)
    for step_idx in (2, 3):
        step = deepcopy(episode['steps'][-1])
        step['step'] = step_idx
        step['image_path'] = f'/img_{step_idx}.png'
        step['action'] = 'MoveAhead'
        episode['steps'].append(step)

        pose = deepcopy(episode['agent_trajectory'][-1])
        pose.update(step=step_idx, image_path=step['image_path'])
        episode['agent_trajectory'].append(pose)
        episode['agent_actions'].append(
            {'step': step_idx, 'action': 'MoveAhead', 'degrees': None}
        )
    return episode


def test_egocentric_item_has_answer_source(tiny_episode):
    items = draft_items_for_episode(
        tiny_episode,
        constructs=['egocentric_encoding'],
        max_per_construct=1,
        styles=('concise',),
    )
    assert items
    item = items[0]
    assert item['status'] == 'ok'
    assert item['construct'] == 'egocentric_encoding'
    assert item['answer'] in item['options']
    assert item['answer_source']
    assert item['answer_source'][0].startswith('steps[')
    step = tiny_episode['steps'][item['query_step']]
    edge = next(
        e
        for e in step['edges_egocentric']
        if e['target'] == item['queried_object_id']
    )
    from cm_benchmark.generation.constructs import angle_relation_to_ego_label

    assert item['options'][item['answer']] == angle_relation_to_ego_label(
        edge['angle_relation']
    )


def test_concise_verbose_pair_share_answer(tiny_episode):
    items = draft_items_for_episode(
        tiny_episode,
        constructs=['egocentric_encoding'],
        max_per_construct=1,
        styles=('concise', 'verbose'),
    )
    assert len(items) == 2
    a, b = items
    assert {a['question_style'], b['question_style']} == {'concise', 'verbose'}
    assert a['answer'] == b['answer']
    assert a['answer_source'] == b['answer_source']
    assert a['paired_item_id'] == b['item_id'] or b['paired_item_id'] == a['item_id']
    verbose = a if a['question_style'] == 'verbose' else b
    answer_text = verbose['options'][verbose['answer']]
    assert answer_text.lower() not in verbose['question'].lower()
    assert len(verbose['question']) > len(
        (a if a['question_style'] == 'concise' else b)['question']
    )


def test_invisible_displacement_requires_visible_to_hidden(folder_episode):
    facts = plan_episode(
        folder_episode, constructs=['invisible_displacement'], max_per_construct=1
    )
    assert facts
    # Fixture has Cup visible at step 0 before move at t=1 → ok when proven
    if facts[0].status == 'ok':
        items = draft_items_for_episode(
            folder_episode,
            constructs=['invisible_displacement'],
            max_per_construct=1,
            styles=('concise',),
        )
        item = items[0]
        assert item['queried_object_id'] == 'Cup|1'
        assert 'to_receptacle' in item['answer_source'][0]
        assert item['options'][item['answer']].startswith('on/in the')
        assert 'Where is the' in item['question']
    else:
        assert facts[0].status == 'unsupported'
        assert 'visible' in (facts[0].reason or '') or 'hidden' in (facts[0].reason or '')


def test_perspective_taking_unsupported(tiny_episode):
    items = draft_items_for_episode(
        tiny_episode,
        constructs=['perspective_taking'],
        styles=('concise',),
    )
    assert len(items) == 1
    assert items[0]['status'] == 'unsupported'
    assert items[0]['question'] == ''
    assert items[0]['answer'] is None


def test_allocentric_encoding_unsupported_without_facing(tiny_episode):
    items = draft_items_for_episode(
        tiny_episode,
        constructs=['allocentric_encoding'],
        styles=('concise',),
    )
    assert len(items) == 1
    assert items[0]['status'] == 'unsupported'
    assert 'facing' in items[0]['distractor_rationale'].get('reason', '')


def test_swm_question_states_explicit_delay_k(delayed_episode):
    facts = plan_episode(
        delayed_episode, constructs=['spatial_working_memory'], max_per_construct=2
    )
    relation = [f for f in facts if (f.extra or {}).get('template_mode') == 'recall_relation']
    assert relation, 'expected at least one recall_relation SWM fact'
    fact = relation[0]
    assert fact.status == 'ok'
    assert fact.queried_object_id == 'Cup|1'
    k = fact.extra['k']
    assert k == fact.query_step - fact.encoding_step
    assert k >= 2
    assert len(fact.image_paths) == k + 1
    q = _core_question(fact)
    assert str(k) in q
    assert 'steps' in q.lower() or 'ago' in q.lower()


def test_swm_delay_range_is_configurable(delayed_episode):
    items = draft_items_for_episode(
        delayed_episode,
        constructs=['spatial_working_memory'],
        max_per_construct=1,
        swm_min_delay=3,
        swm_max_delay=3,
        styles=('concise',),
    )
    ok = [item for item in items if item.get('status') == 'ok']
    assert ok, 'expected an SWM item at the requested delay'
    item = ok[0]
    assert item['query_step'] - item['encoding_step'] == 3
    assert len(item['image_paths']) == 4
    assert len(item['agent_trajectory']) == 4
    assert len(item['agent_actions']) == 3


def test_spatial_updating_mentions_now(delayed_episode):
    items = draft_items_for_episode(
        delayed_episode,
        constructs=['spatial_updating'],
        max_per_construct=1,
        styles=('concise',),
    )
    if not items or items[0].get('status') == 'unsupported':
        pytest.skip('no spatial_updating fact on delayed episode')
    item = items[0]
    q = item['question'].lower()
    assert 'relative to you now' in q or 'now' in q
    assert len(item.get('image_paths') or []) >= 3
    assert item['query_step'] - item['encoding_step'] >= 2
    assert 'time order' in q or 'in order' in q or 'earlier' in q


def test_spatial_updating_delay_range_is_configurable(delayed_episode):
    items = draft_items_for_episode(
        delayed_episode,
        constructs=['spatial_updating'],
        max_per_construct=1,
        su_min_delay=3,
        su_max_delay=3,
        styles=('concise',),
    )
    ok = [item for item in items if item.get('status') == 'ok']
    assert ok, 'expected a spatial_updating item at the requested delay'
    item = ok[0]
    assert item['query_step'] - item['encoding_step'] == 3
    assert len(item['image_paths']) == 4
    assert len(item['agent_trajectory']) == 4
    assert len(item['agent_actions']) == 3


def test_route_knowledge_is_retrace_not_plan(folder_episode):
    """Class-4 route items retrace walked A→B; not full-episode turn dump."""
    facts = plan_episode(
        folder_episode, constructs=['route_knowledge'], max_per_construct=2
    )
    if facts and facts[0].status == 'unsupported':
        assert 'source_goal' in (facts[0].reason or '') or 'region' in (facts[0].reason or '')
        return
    assert facts
    fact = facts[0]
    assert fact.extra.get('source')
    assert fact.extra.get('goal')
    assert fact.encoding_step is not None and fact.query_step is not None
    assert fact.query_step - fact.encoding_step < 100
    assert ' → ' in (fact.answer_label or '')
    assert len((fact.answer_label or '').split(' → ')) <= 10
    q = _core_question(fact).lower()
    assert 'retrace' in q
    assert 'plan a route' not in q


def test_survey_knowledge_unsupported_without_novel_path(folder_episode):
    facts = plan_episode(
        folder_episode, constructs=['survey_knowledge'], max_per_construct=1
    )
    assert facts
    assert facts[0].status == 'unsupported'
    reason = facts[0].reason or ''
    assert (
        'connectivity' in reason
        or 'novel' in reason
        or 'regions' in reason
        or 'untraversed' in reason
    )


def test_object_type_skips_undefined_category():
    from cm_benchmark.generation.constructs import object_type_from_id

    assert (
        object_type_from_id(
            'ObjaScooter|4|5', {'ObjaScooter|4|5': {'category': 'Undefined'}}
        )
        == 'ObjaScooter'
    )
    assert (
        object_type_from_id('FloorLamp|4|2', {'FloorLamp|4|2': {'category': 'FloorLamp'}})
        == 'FloorLamp'
    )
    assert object_type_from_id('Cup|1') == 'Cup'


def test_verbose_preamble_does_not_leak_answer(tiny_episode):
    facts = plan_episode(tiny_episode, constructs=['egocentric_encoding'], max_per_construct=1)
    fact = facts[0]
    preamble = build_verbose_preamble(tiny_episode, fact)
    assert fact.answer_label.lower() not in preamble.lower()


def test_core_question_covers_active_template_placeholders():
    """Guard against KeyError when formatting templates."""
    from cm_benchmark.generation.planner import PlannedFact

    for construct, modes in (
        ('egocentric_encoding', [None]),
        ('invisible_displacement', [None]),
        ('spatial_updating', [None]),
        ('route_knowledge', [None]),
        ('survey_knowledge', [None]),
        ('spatial_working_memory', ['recall_relation', 'recall_count']),
    ):
        for mode in modes:
            tmpl = select_template(construct, template_mode=mode)
            fact = PlannedFact(
                construct=construct,
                status='ok',
                query_step=5,
                encoding_step=2,
                queried_object_id='Cup|1',
                reference_object_id='Table|1',
                answer_label='to your left',
                extra={
                    'object_type': 'Cup',
                    'object_category': 'Cup',
                    'reference_object': 'Table',
                    'source': 'Kitchen',
                    'goal': 'LivingRoom',
                    'k': 3,
                    'template_mode': mode,
                },
            )
            q = _core_question(fact)
            assert '{object' not in q
            assert '{k}' not in q
            assert tmpl.split('{')[0] in q or 'Cup' in q or 'Kitchen' in q


@pytest.mark.parametrize(
    'construct', ['spatial_working_memory', 'spatial_updating']
)
def test_trajectory_hooks_are_scoped_to_item_frames(delayed_episode, construct):
    items = draft_items_for_episode(
        delayed_episode,
        constructs=[construct],
        max_per_construct=1,
        styles=('concise',),
    )
    ok = [i for i in items if i.get('status') == 'ok']
    if not ok:
        pytest.skip(f'no ok {construct} item')
    item = ok[0]

    poses = item.get('agent_trajectory')
    assert poses, 'expected pose per item frame'
    assert len(poses) == len(item['image_paths'])
    assert [p.get('image_path') for p in poses] == item['image_paths']
    assert len(poses) <= len(delayed_episode['agent_trajectory'])

    actions = item.get('agent_actions') or []
    steps = [a['step'] for a in actions]
    assert all(item['encoding_step'] < s <= item['query_step'] for s in steps)
    assert len(actions) <= item['query_step'] - item['encoding_step']


def test_route_knowledge_does_not_leak_action_list(folder_episode):
    items = draft_items_for_episode(
        folder_episode,
        constructs=['route_knowledge'],
        max_per_construct=1,
        styles=('concise',),
    )
    ok = [i for i in items if i.get('status') == 'ok']
    if not ok:
        pytest.skip('no ok route item')
    assert ok[0].get('agent_actions') is None


def test_write_draft_json(folder_episode, tmp_path):
    from cm_benchmark.generation.episode_io import write_draft_items

    items = draft_items_for_episode(
        folder_episode,
        constructs=['egocentric_encoding', 'invisible_displacement', 'perspective_taking'],
        max_per_construct=1,
        styles=('concise',),
    )
    out = write_draft_items(items, tmp_path / 'draft.json')
    assert out.is_file()
    import json

    data = json.loads(out.read_text())
    assert data['n_items'] == len(items)
