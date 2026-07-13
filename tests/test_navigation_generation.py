"""Simple end-to-end tests for Ai2ThorNavGenerator with tiny CSV fixtures."""

import json
from pathlib import Path

import pytest

from cm_benchmark.generator.ai2thor_nav_generator import Ai2ThorNavGenerator

FIXTURES = Path(__file__).parent / 'fixtures'
NAV_CSV = FIXTURES / 'navigation_tiny.csv'
OBJ_CSV = FIXTURES / 'objects_tiny.csv'


@pytest.fixture
def episode(tmp_path):
    gen = Ai2ThorNavGenerator(
        path_navigation=str(NAV_CSV),
        path_objects=str(OBJ_CSV),
        output_path=str(tmp_path),
        output_filename='episode.json',
    )
    data = gen.collect_episode_data(extra_data={'scene': 'TinyScene', 'all_distances': []})
    gen.export_to_json(data, 'episode.json')
    return data, tmp_path / 'episode.json'


def test_episode_has_taxonomy_level_fields(episode):
    data, _ = episode
    assert data['scene'] == 'TinyScene'
    assert data['object_state_track'] is None
    assert data['displacement_events'] == []
    assert len(data['agent_trajectory']) == 2
    assert len(data['agent_actions']) == 2
    assert 'landmarks' in data['route']
    assert 'turns' in data['route']


def test_step0_cup_is_visible_and_egocentric_edge_exists(episode):
    data, _ = episode
    step0 = data['steps'][0]
    assert 'Cup|1' in step0['visible_objects']
    assert step0['non_visible_objects'] == {}
    assert step0['edges_object_frame'] == []

    ego_targets = [e['target'] for e in step0['edges_egocentric']]
    assert 'Cup|1' in ego_targets

    cup_edge = next(e for e in step0['edges_egocentric'] if e['target'] == 'Cup|1')
    # Cup is ahead on +Z → front, no left/right
    assert cup_edge['angle_relation'][2] == 'front'
    assert cup_edge['inferred'] is False


def test_step1_cup_is_remembered_as_non_visible(episode):
    data, _ = episode
    step1 = data['steps'][1]
    assert 'Plate|1' in step1['visible_objects']
    assert 'Cup|1' in step1['non_visible_objects']
    assert step1['non_visible_objects']['Cup|1']['last_seen_step'] == 0

    inferred_targets = [e['target'] for e in step1['edges_inferred']]
    assert 'Cup|1' in inferred_targets
    cup_inf = next(e for e in step1['edges_inferred'] if e['target'] == 'Cup|1')
    assert cup_inf['inferred'] is True
    assert cup_inf['last_seen'] == 0
    assert cup_inf['visible'] is False


def test_route_turns_and_landmark_from_nearby_cup(episode):
    data, _ = episode
    turns = data['route']['turns']
    assert turns[0]['action'] == 'rotate_right'
    assert turns[1]['degrees'] == 90

    landmark_ids = [lm['object_id'] for lm in data['route']['landmarks']]
    # Cup at 0.8m → nearby; should appear as a landmark
    assert 'Cup|1' in landmark_ids


def test_exported_json_is_readable(episode):
    _, json_path = episode
    loaded = json.loads(json_path.read_text())
    assert loaded['scene'] == 'TinyScene'
    assert len(loaded['steps']) == 2
