"""Simple tests for SQLite EpisodeStore."""

from pathlib import Path

import pytest

from cm_benchmark.generator.ai2thor_nav_generator import Ai2ThorNavGenerator
from cm_benchmark.storage.episode_store import EpisodeStore

FIXTURES = Path(__file__).parent / 'fixtures'
NAV_CSV = FIXTURES / 'navigation_tiny.csv'
OBJ_CSV = FIXTURES / 'objects_tiny.csv'


@pytest.fixture
def tiny_episode(tmp_path):
    gen = Ai2ThorNavGenerator(
        path_navigation=str(NAV_CSV),
        path_objects=str(OBJ_CSV),
        output_path=str(tmp_path),
        output_filename='unused.json',
    )
    data = gen.collect_episode_data(extra_data={'scene': 'TinyScene', 'all_distances': []})
    return gen, data, tmp_path / 'episodes.db'


def test_save_and_load_roundtrip(tiny_episode):
    gen, data, db_path = tiny_episode
    gen.export_to_db(data, db_path=str(db_path), episode_id='tiny_1')

    with EpisodeStore(db_path) as store:
        loaded = store.load_episode('tiny_1')
        listed = store.list_episodes()

    assert listed[0]['episode_id'] == 'tiny_1'
    assert listed[0]['n_steps'] == 2
    assert loaded['scene'] == 'TinyScene'
    assert len(loaded['steps']) == 2
    assert 'Cup|1' in loaded['steps'][0]['visible_objects']
    assert 'Cup|1' in loaded['steps'][1]['non_visible_objects']
    assert loaded['route']['landmarks']


def test_query_inferred_edges_without_full_load(tiny_episode):
    gen, data, db_path = tiny_episode
    gen.export_to_db(data, db_path=str(db_path), episode_id='tiny_2')

    with EpisodeStore(db_path) as store:
        edges = store.get_edges('tiny_2', step=1, edge_type='inferred', target='Cup|1')

    assert len(edges) == 1
    assert edges[0]['inferred'] is True
    assert edges[0]['last_seen'] == 0


def test_replace_episode_keeps_single_copy(tiny_episode):
    gen, data, db_path = tiny_episode
    gen.export_to_db(data, db_path=str(db_path), episode_id='tiny_3')
    gen.export_to_db(data, db_path=str(db_path), episode_id='tiny_3')

    with EpisodeStore(db_path) as store:
        assert len(store.list_episodes()) == 1
        assert len(store.load_episode('tiny_3')['steps']) == 2
