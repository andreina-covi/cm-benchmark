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
    data = gen.collect_episode_data(extra_data={'scene': 'TinyScene'})
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
    data = gen.collect_episode_data(extra_data={'scene': 'ignore'})
    return gen, data


def test_folder_loads_displacement_and_survey_fields(folder_episode):
    _gen, data = folder_episode
    assert data['scene'] == 'house_tiny'
    assert len(data['displacement_events']) == 3
    disp_0 = data['displacement_events'][0]
    assert disp_0['obj_id'] == 'Cup|1'
    assert disp_0['hidden_during'] is True
    assert disp_0['moved_via'] == 'direct'
    swap_rows = [e for e in data['displacement_events'] if e.get('moved_via') == 'swap']
    assert len(swap_rows) == 2
    assert {e['obj_id'] for e in swap_rows} == {'Cup|1', 'Plate|1'}
    assert swap_rows[0]['swap_partner_id'] in {'Cup|1', 'Plate|1'}

    assert data['object_state_track'] is not None
    assert 'Cup|1' in data['object_state_track']
    entries = data['object_state_track']['Cup|1']['entries']
    # Sparse: step 0 (first) + steps 1–2 (pose/fov changes) — no duplicate unchanged rows
    assert [e['step'] for e in entries] == [0, 1, 2]
    assert entries[0]['in_camera_fov'] is True
    assert entries[1]['in_camera_fov'] is False
    assert entries[1]['position'] == [0.2, 1.0, 0.5]
    assert entries[2]['in_camera_fov'] is False
    assert entries[2]['position'] == [0.8, 1.0, 0.0]

    from cm_benchmark.generator.ai2thor_nav_generator import state_at_step

    # Carry-forward: querying step with no new entry uses the previous state
    assert state_at_step(entries, 0)['in_camera_fov'] is True
    mid = state_at_step(entries, 1)
    assert mid['position'] == [0.2, 1.0, 0.5]
    assert state_at_step(entries, 5)['position'] == [0.8, 1.0, 0.0]

    assert data['world_layout']['regions'][0]['region_id'] == 'room|1'
    # Self-loop connectivity rows are dropped at ingest.
    assert all(
        c['from_region'] != c['to_region']
        for c in data['world_layout']['connectivity']
    )
    assert data.get('nav_graph') is not None
    assert data['nav_graph']['scene_id'] == 'house_tiny'
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


def _clone_episode(src: Path, dest: Path, scene_id: str) -> None:
    """Copy a flat fixture into <dest>/annotations with the scene id renamed."""
    ann = dest / 'annotations'
    ann.mkdir(parents=True)
    (dest / 'images').mkdir()
    for path in src.iterdir():
        if not path.is_file():
            continue
        text = path.read_text().replace('house_tiny', scene_id)
        ann.joinpath(path.name.replace('house_tiny', scene_id)).write_text(text)


def test_main_builds_every_child_episode_with_one_visibility_model(tmp_path):
    """A root of timestamp folders yields one episode per child, same model."""
    import joblib
    import numpy as np
    from argparse import Namespace
    from sklearn.tree import DecisionTreeClassifier

    from cm_benchmark.generator.ai2thor_nav_generator import main
    from cm_benchmark.storage import EpisodeStore

    root = tmp_path / 'navigation'
    _clone_episode(EPISODE_DIR, root / '09_23_2026_16_31_04_526137', 'house_007514')
    _clone_episode(EPISODE_DIR, root / '09_23_2026_16_35_09_464480', 'house_001030')

    rng = np.random.default_rng(0)
    side = np.concatenate([rng.uniform(1, 8, 20), rng.uniform(20, 40, 20)])
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(np.column_stack([side * side, side]), (side >= 15).astype(int))
    model_path = tmp_path / 'visibility_filter.joblib'
    joblib.dump(
        {
            'model': clf,
            'features': ['bbox-area', 'min-side'],
            'low': 0.3,
            'high': 0.7,
            'ambiguous_proba_stats': {'n': 0, 'min': 0.3, 'max': 0.7, 'mean': 0.5},
        },
        model_path,
    )

    out = tmp_path / 'nav_data'
    db = tmp_path / 'episodes'
    main(
        Namespace(
            csv_path_folder=str(root),
            scene_id='should_be_ignored',
            episode_id='should_be_ignored',
            output_path=str(out),
            output_filename=None,
            db_path=str(db),
            environment='ai2thor',
            export_json=True,
            visibility_model_path=str(model_path),
            file_navigation=None,
            file_objects=None,
            file_object_state=None,
            file_displacement_events=None,
            file_displacement_candidates=None,
        )
    )

    assert sorted(p.relative_to(out).as_posix() for p in out.rglob('*.json')) == [
        'house_001030/nav_house_001030.json',
        'house_007514/nav_house_007514.json',
    ]
    for scene in ('house_007514', 'house_001030'):
        payload = json.loads((out / scene / f'nav_{scene}.json').read_text())
        assert payload['scene'] == scene
        assert payload['visibility_filter_model']['path'] == str(model_path.resolve())
        with EpisodeStore(db / scene / 'episodes.db') as store:
            rows = store.list_episodes()
        assert [row['scene'] for row in rows] == [scene]
