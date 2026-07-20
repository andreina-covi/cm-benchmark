"""Tests for first-draft taxonomy Q&A generation."""

from pathlib import Path

import pytest

from cm_benchmark.generator.ai2thor_nav_generator import Ai2ThorNavGenerator
from cm_benchmark.generation.pipeline import draft_items_for_episode
from cm_benchmark.generation.planner import plan_episode
from cm_benchmark.generation.templates import build_verbose_preamble

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
    return gen.collect_episode_data(extra_data={'scene': 'TinyScene', 'all_distances': []})


@pytest.fixture
def folder_episode(tmp_path):
    gen = Ai2ThorNavGenerator(
        csv_path_folder=str(EPISODE_DIR),
        output_path=str(tmp_path),
        output_filename='folder_episode.json',
    )
    return gen.collect_episode_data(extra_data={'scene': 'ignore', 'all_distances': []})


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
    # Mini-validator: answer label matches ego edge
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
    # Preamble should mention another visible category when available
    # (step 0 only has Cup — step with Plate+Cup memory may vary)
    assert len(verbose['question']) > len(
        (a if a['question_style'] == 'concise' else b)['question']
    )


def test_invisible_displacement_from_folder(folder_episode):
    items = draft_items_for_episode(
        folder_episode,
        constructs=['invisible_displacement'],
        max_per_construct=1,
        styles=('concise',),
    )
    assert items
    item = items[0]
    assert item['status'] == 'ok'
    assert item['queried_object_id'] == 'Cup|1'
    assert 'to_receptacle' in item['answer_source'][0]
    assert item['options'][item['answer']].startswith('on/in the')


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


def test_swm_item_when_object_static(tiny_episode):
    facts = plan_episode(tiny_episode, constructs=['spatial_working_memory'], max_per_construct=2)
    assert facts
    assert facts[0].status == 'ok'
    assert facts[0].queried_object_id == 'Cup|1'


def test_spatial_updating_mentions_frame_order(tiny_episode):
    items = draft_items_for_episode(
        tiny_episode,
        constructs=['spatial_updating'],
        max_per_construct=1,
        styles=('concise',),
    )
    if not items or items[0].get('status') == 'unsupported':
        pytest.skip('no spatial_updating fact on tiny episode')
    item = items[0]
    if len(item.get('image_paths') or []) < 2:
        pytest.skip('spatial_updating item has fewer than 2 images')
    q = item['question'].lower()
    assert 'time order' in q or 'in order' in q
    assert 'last image' in q or 'last' in q


def test_route_knowledge_is_short_source_goal_not_full_episode(folder_episode):
    """Class-4 route items plan A→B segments; they must not dump all episode turns."""
    facts = plan_episode(
        folder_episode, constructs=['route_knowledge'], max_per_construct=2
    )
    # Tiny fixture may lack multi-region segments → unsupported is acceptable
    if facts and facts[0].status == 'unsupported':
        assert 'source_goal' in (facts[0].reason or '') or 'region' in (facts[0].reason or '')
        return
    assert facts
    fact = facts[0]
    assert fact.extra.get('source')
    assert fact.extra.get('goal')
    assert fact.encoding_step is not None and fact.query_step is not None
    assert fact.query_step - fact.encoding_step < 100
    # Answer should be a compressed plan, not hundreds of raw steps
    assert ' → ' in (fact.answer_label or '')
    assert len((fact.answer_label or '').split(' → ')) <= 10


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
