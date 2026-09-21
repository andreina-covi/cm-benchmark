"""Config YAML must load and name the eight constructs.

Numeric gates live in planner.py, not in the YAML. The YAML states what each
construct tests; tests below only lock schema facts that would silently
contradict the generator.
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


def test_route_does_not_require_agent_actions():
    taxonomy = _load(TAXONOMY)
    route = _taxonomy_constructs()['route_knowledge']
    assert 'agent_actions' not in route['required_temporal_fields']
    assert 'agent_trajectory' in route['required_temporal_fields']
    assert 'nav_graph' in route['required_temporal_fields']
    hooks = taxonomy['schema_hooks']
    assert 'route_knowledge' not in hooks['agent_actions']['required_for']
    assert 'route_knowledge' in hooks['nav_graph']['required_for']


def test_class4_is_free_action_sequence_not_mcq():
    for name in ('route_knowledge', 'survey_based_route_planning'):
        cfg = _taxonomy_constructs()[name]
        assert cfg['class'] == 4
        assert cfg['distractor_pattern'] == []
        assert 'Not MCQ' in cfg['answer_type'] or 'not MCQ' in cfg['answer_type']
