"""Generation quality numbers live in configs/generation.yaml, not in source."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cm_benchmark.generation.config import (
    default_config_path,
    get_generation_config,
    load_generation_config,
)

REPO = Path(__file__).resolve().parent.parent
SCAN_ROOTS = (
    REPO / 'src' / 'cm_benchmark' / 'generation',
    REPO / 'src' / 'cm_benchmark' / 'generator' / 'visibility_filters.py',
    REPO / 'src' / 'cm_benchmark' / 'utils' / 'spatial_relations.py',
)
SKIP_FILES = frozenset({'config.py', 'slide_copy.py', 'paraphrase.py', 'episode_io.py', 'schema.py'})
# Structural integers / circle math / tiny epsilons may stay in code.
ALLOW = {
    0,
    1,
    -1,
    2,
    3,
    4,
    90,
    180,
    360,
    0.0,
    0.5,
    1.0,
    2.0,
    90.0,
    180.0,
    360.0,
}


def _is_tiny_epsilon(value: float) -> bool:
    return 0 < abs(value) <= 1e-5


def _yaml_quality_numbers() -> set[float]:
    cfg = load_generation_config()
    blob = cfg.content_hash()  # force construction
    del blob
    raw = default_config_path().read_text()
    import yaml

    def walk(node, out: set[float]) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            out.add(float(node))
        elif isinstance(node, dict):
            for v in node.values():
                walk(v, out)
        elif isinstance(node, list):
            for v in node:
                walk(v, out)

    found: set[float] = set()
    walk(yaml.safe_load(raw), found)
    return found


def _iter_scan_files():
    for root in SCAN_ROOTS:
        if root.is_file():
            yield root
            continue
        for path in sorted(root.glob('*.py')):
            if path.name in SKIP_FILES:
                continue
            yield path


def test_default_config_loads_and_matches_phase0_values():
    cfg = get_generation_config()
    assert cfg.name == 'default'
    assert cfg.directions.ahead_half_width_fov_deg == 20.0
    assert cfg.directions.ahead_half_width_full_deg == 45.0
    assert cfg.directions.pt_back_deg == 135.0
    assert cfg.directions.pt_boundary_margin_deg == 10.0
    assert cfg.visibility.query.min_bbox_area == 1200.0
    assert cfg.visibility.id_encode.min_bbox_area == 100.0
    assert cfg.disambiguation.active_proximity().margin_m == 0.6
    assert cfg.class4.min_pair_geodesic_m == 1.0
    assert cfg.class4.max_pair_geodesic_m == 30.0
    assert cfg.class4.route.max_ratio_floor == 1.1
    assert cfg.class4.survey.max_ratio_floor == 1.05
    assert cfg.budget.max_per_construct == 3
    assert cfg.content_hash()
    assert len(cfg.content_hash()) == 64


def test_ablation_overlay_merges_onto_default():
    cfg = load_generation_config(REPO / 'configs' / 'ablations' / 'sr3d.yaml')
    assert cfg.name == 'sr3d'
    assert cfg.disambiguation.proximity_setting == 'sr3d'
    assert cfg.disambiguation.active_proximity().margin_m == 0.20
    assert cfg.disambiguation.active_proximity().ratio is None
    assert cfg.disambiguation.all_marks is False
    assert cfg.directions.ahead_half_width_fov_deg == 20.0

    marks = load_generation_config(REPO / 'configs' / 'ablations' / 'all_marks.yaml')
    assert marks.name == 'all_marks'
    assert marks.disambiguation.all_marks is True
    assert marks.disambiguation.proximity_setting == 'current'


def test_legacy_constant_names_read_the_config():
    from cm_benchmark.generation import constructs, planner

    assert constructs.AHEAD_HALF_WIDTH_FOV == 20.0
    assert planner.ROUTE_MIN_HOP_COUNT == 2
    assert planner.MIN_PAIR_GEODESIC_M == 1.0


@pytest.mark.parametrize('path', list(_iter_scan_files()), ids=lambda p: p.name)
def test_no_quality_literals_outside_config(path: Path):
    """Fail if a generation.yaml number is copied back into generation code."""
    quality = {v for v in _yaml_quality_numbers() if v not in ALLOW and not _is_tiny_epsilon(v)}
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            continue
        value = float(node.value)
        if value in ALLOW or _is_tiny_epsilon(value):
            continue
        if value in quality or (isinstance(node.value, int) and float(node.value) in quality):
            hits.append(f'{path.name}:{node.lineno} {node.value!r}')
    assert not hits, 'quality-affecting literals must live in configs/generation.yaml:\n' + '\n'.join(
        hits
    )
