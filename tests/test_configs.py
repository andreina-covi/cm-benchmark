"""Config Registry coherence: the YAML must load and agree with the code.

``configs/taxonomy.yaml`` is the declared single source of truth for the eight
construct configs. Nothing imports it yet (the Config Registry node has no
code), so nothing else would notice if it stopped parsing or drifted away from
the constants the planner actually enforces.
"""

from pathlib import Path

import pytest
import yaml

CONFIGS = Path(__file__).parent.parent / 'configs'
TAXONOMY = CONFIGS / 'taxonomy.yaml'
CONSTRUCT_DIR = CONFIGS / 'constructs'

CONSTRUCTS = (
    'egocentric_encoding',
    'allocentric_encoding',
    'spatial_working_memory',
    'invisible_displacement',
    'spatial_updating',
    'perspective_taking',
    'route_knowledge',
    'survey_based_route_planning',
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _taxonomy_constructs() -> dict:
    taxonomy = _load(TAXONOMY)
    out = {}
    for cls in ('class_1', 'class_2', 'class_3', 'class_4'):
        out.update(taxonomy[cls]['constructs'])
    return out


@pytest.mark.parametrize(
    'path', [TAXONOMY] + sorted(CONSTRUCT_DIR.glob('*.yaml')), ids=lambda p: p.name
)
def test_config_yaml_parses(path):
    assert isinstance(_load(path), dict)


def test_taxonomy_defines_all_eight_constructs():
    assert set(_taxonomy_constructs()) == set(CONSTRUCTS)


def test_class4_configs_cite_the_reference_builder_actually_used():
    """Class-4 answers come from follow_path_actions(), not the per-hop converter."""
    for name in ('route_knowledge', 'survey_based_route_planning'):
        text = (CONSTRUCT_DIR / f'{name}.yaml').read_text()
        method = _taxonomy_constructs()[name]['ground_truth_method']
        for blob in (text, method):
            assert 'follow_path_actions()' in blob, name
            assert 'stored answer is path_to_nav_actions()' not in blob, name


def test_class4_configs_match_planner_gate_constants():
    """Numbers quoted in the configs must be the ones the planner enforces."""
    from cm_benchmark.generation import planner

    route = _taxonomy_constructs()['route_knowledge']['ground_truth_method']
    survey = _taxonomy_constructs()['survey_based_route_planning']['ground_truth_method']

    assert planner.MIN_PAIR_GEODESIC_M == 1.0
    assert planner.MAX_PAIR_GEODESIC_M == 30.0
    assert f'{planner.ROUTE_MIN_GEODESIC_EUCLIDEAN_RATIO:g}' in route
    assert f'{planner.SURVEY_MIN_GEODESIC_EUCLIDEAN_RATIO:g}' in survey
    turn_band = f'({planner.ROUTE_MIN_TURN_COUNT}–{planner.ROUTE_MAX_TURN_COUNT})'
    assert turn_band in route
    assert f'ROUTE_MIN_HOP_COUNT ({planner.ROUTE_MIN_HOP_COUNT}' in route


def test_route_does_not_require_agent_actions():
    """Route rebuilds actions from the graph path plus trajectory yaw.

    Class-4 items also ship without ``agent_actions``, so requiring the field
    here would contradict both the generator and the item schema.
    """
    taxonomy = _load(TAXONOMY)
    route = _taxonomy_constructs()['route_knowledge']
    assert 'agent_actions' not in route['required_temporal_fields']
    assert 'agent_trajectory' in route['required_temporal_fields']
    assert 'nav_graph' in route['required_temporal_fields']

    hooks = taxonomy['schema_hooks']
    assert 'route_knowledge' not in hooks['agent_actions']['required_for']
    assert 'route_knowledge' in hooks['nav_graph']['required_for']


def test_class4_is_free_action_sequence_not_mcq():
    """Invariant: class 4 has no MCQ options and no distractor patterns."""
    for name in ('route_knowledge', 'survey_based_route_planning'):
        cfg = _taxonomy_constructs()[name]
        assert cfg['class'] == 4
        assert cfg['distractor_pattern'] == []
        assert 'Not MCQ' in cfg['answer_type'] or 'not MCQ' in cfg['answer_type']
