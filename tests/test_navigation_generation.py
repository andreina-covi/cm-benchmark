"""End-to-end tests: tiny CSVs and folder-based SPOC-style episodes."""

import json
from pathlib import Path

import pytest

from cm_benchmark.generator.ai2thor_nav_generator import Ai2ThorNavGenerator
from cm_benchmark.generator.episode_paths import resolve_episode_paths

FIXTURES = Path(__file__).parent / 'fixtures'
NAV_CSV = FIXTURES / 'navigation_tiny.csv'
OBJ_CSV = FIXTURES / 'objects_tiny.csv'
EPISODE_DIR = FIXTURES / 'episode_tiny'


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
    assert 'Cup|1' in landmark_ids


def test_exported_json_is_readable(episode):
    _, json_path = episode
    loaded = json.loads(json_path.read_text())
    assert loaded['scene'] == 'TinyScene'
    assert len(loaded['steps']) == 2


def test_resolve_episode_paths_from_folder():
    paths = resolve_episode_paths(EPISODE_DIR)
    assert paths['scene_id'] == 'house_tiny'
    assert paths['navigation'].name == 'navigation-house_tiny.csv'
    assert paths['displacement_events'].name.startswith('displacement_events')


@pytest.fixture
def folder_episode(tmp_path):
    gen = Ai2ThorNavGenerator(
        csv_path_folder=str(EPISODE_DIR),
        output_path=str(tmp_path),
        output_filename='folder_episode.json',
    )
    data = gen.collect_episode_data(extra_data={'scene': 'ignore', 'all_distances': []})
    return gen, data


def test_folder_loads_displacement_and_survey_fields(folder_episode):
    _gen, data = folder_episode
    assert data['scene'] == 'house_tiny'
    assert len(data['displacement_events']) == 1
    assert data['displacement_events'][0]['obj_id'] == 'Cup|1'
    assert data['displacement_events'][0]['hidden_during'] is True

    assert data['object_state_track'] is not None
    assert 'Cup|1' in data['object_state_track']
    entries = data['object_state_track']['Cup|1']['entries']
    # Sparse: step 0 (first) + step 1 (pose/fov change) — no duplicate unchanged rows
    assert [e['step'] for e in entries] == [0, 1]
    assert entries[0]['in_camera_fov'] is True
    assert entries[1]['in_camera_fov'] is False
    assert entries[1]['position'] == [0.2, 1.0, 0.5]

    from cm_benchmark.generator.ai2thor_nav_generator import state_at_step

    # Carry-forward: querying step with no new entry uses the previous state
    assert state_at_step(entries, 0)['in_camera_fov'] is True
    mid = state_at_step(entries, 1)
    assert mid['position'] == [0.2, 1.0, 0.5]
    assert state_at_step(entries, 5)['position'] == [0.2, 1.0, 0.5]

    assert data['world_layout']['regions'][0]['region_id'] == 'room|1'
    assert data['region_trajectory'][0]['region_id'] == 'room|1'
    # Sparse: fixture has same room on steps 0 and 1 → only first kept
    assert len(data['region_trajectory']) == 1
    assert data['passage_state'][0]['is_open'] is True
    assert data['episode_meta']['episode_id'] == 'house_tiny_test'


def test_folder_episode_exports_to_db(folder_episode, tmp_path):
    gen, data = folder_episode
    db_path = tmp_path / 'ep.db'
    gen.export_to_db(data, db_path=str(db_path), episode_id='folder_1')

    from cm_benchmark.storage import EpisodeStore

    with EpisodeStore(db_path) as store:
        loaded = store.load_episode('folder_1')
    assert loaded['displacement_events'][0]['obj_id'] == 'Cup|1'
    assert 'Cup|1' in loaded['object_state_track']
    assert loaded['world_layout'] is not None
