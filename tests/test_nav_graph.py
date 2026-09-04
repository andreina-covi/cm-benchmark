"""Unit tests for offline nav_graph helpers."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from cm_benchmark.generation.nav_graph import (
    build_nav_graph,
    derive_turns,
    filter_valid_connectivity,
    format_turn_sequence,
    is_valid_untraversed_shortcut,
    perturb_turn_sequence,
    sanitize_world_layout,
    shortest_path,
    snap_position_to_graph,
    snap_trajectory_to_graph,
    was_traversed,
)

FIXTURE = Path(__file__).parent / 'fixtures' / 'episode_tiny' / 'nav_graph-house_tiny.json'


@pytest.fixture
def tiny_graph():
    return build_nav_graph(json.loads(FIXTURE.read_text()))


def test_build_nav_graph_synthesizes_8_connected_edges(tiny_graph):
    assert tiny_graph.number_of_nodes() == 9
    assert tiny_graph.number_of_edges() > 0
    # Orthogonal neighbor
    assert tiny_graph.has_edge('n0', 'n1')
    # Diagonal neighbor within grid_size * sqrt(2)
    assert tiny_graph.has_edge('n0', 'n4')


def test_snap_landmark_allows_larger_radius(tiny_graph):
    from cm_benchmark.generation.nav_graph import snap_landmark_to_graph

    # Slightly off the grid but within landmark radius
    assert snap_landmark_to_graph(tiny_graph, (0.9, 1.0, 0.0)) is not None
    assert snap_landmark_to_graph(tiny_graph, (50.0, 1.0, 50.0)) is None


def test_snap_to_nearest_of_prefers_candidates(tiny_graph):
    from cm_benchmark.generation.nav_graph import snap_to_nearest_of

    # Pose near n0; restrict candidates to a farther node still within radius
    nodes = list(tiny_graph.nodes())
    assert nodes
    chosen = snap_to_nearest_of(tiny_graph, (0.0, 1.0, 0.0), [nodes[-1], nodes[0]])
    assert chosen in (nodes[0], nodes[-1])
    assert snap_to_nearest_of(tiny_graph, (0.0, 1.0, 0.0), []) is None


def test_snap_trajectory_collapses_duplicates_and_drops_far(tiny_graph):
    traj = [
        {'step': 0, 'position': (0.0, 1.0, 0.0)},
        {'step': 1, 'position': (0.02, 1.0, 0.0)},  # same node
        {'step': 2, 'position': (0.25, 1.0, 0.0)},
        {'step': 3, 'position': (99.0, 1.0, 99.0)},  # drop
        {'step': 4, 'position': (0.5, 1.0, 0.0)},
    ]
    snapped = snap_trajectory_to_graph(tiny_graph, traj)
    assert [r['node_id'] for r in snapped] == ['n0', 'n1', 'n2']


def test_was_traversed_contiguous_subsequence():
    trav = ['a', 'b', 'c', 'd']
    assert was_traversed(['b', 'c'], trav)
    assert not was_traversed(['a', 'c'], trav)
    assert not was_traversed(['c', 'b'], trav)


def test_is_valid_untraversed_shortcut(tiny_graph):
    trav = ['n0', 'n1', 'n2']
    # Experienced segment
    assert not is_valid_untraversed_shortcut(['n0', 'n1', 'n2'], trav, tiny_graph)
    # Alternate path via z>0 corridor
    alt = shortest_path(tiny_graph, 'n0', 'n8')
    assert is_valid_untraversed_shortcut(alt, trav, tiny_graph)


def test_derive_turns_bins_to_rotation_granularity(tiny_graph):
    # L-shaped path: n0 -> n1 -> n4 (turn)
    path = ['n0', 'n1', 'n4', 'n7']
    turns = derive_turns(path, tiny_graph, rotation_deg=45.0)
    assert turns
    labels = [t['label'] for t in turns]
    assert all(lab in {
        'straight', 'turn left', 'turn right',
        'sharp turn left', 'sharp turn right', 'turn around',
    } for lab in labels)
    text = format_turn_sequence(turns)
    assert ' → ' in text or text in labels


def test_format_turn_sequence_compresses_straights():
    turns = [
        {'node_id': 'a', 'label': 'straight', 'landmark': None},
        {'node_id': 'b', 'label': 'straight', 'landmark': None},
        {'node_id': 'c', 'label': 'turn left', 'landmark': 'Door'},
        {'node_id': 'd', 'label': 'straight', 'landmark': None},
        {'node_id': 'e', 'label': 'straight', 'landmark': None},
    ]
    assert format_turn_sequence(turns) == 'straight → turn left @ Door → straight'


def test_perturb_reversed_sequence():
    turns = [
        {'node_id': 'n1', 'label': 'turn left', 'landmark': 'Door'},
        {'node_id': 'n2', 'label': 'turn right', 'landmark': None},
    ]
    pert = perturb_turn_sequence(turns, 'reversed_sequence')
    assert [t['label'] for t in pert] == ['turn right', 'turn left']
    # Alias still accepted
    alias = perturb_turn_sequence(turns, 'opposite_direction')
    assert [t['label'] for t in alias] == ['turn right', 'turn left']


def test_filter_self_loop_connectivity():
    rows = [
        {'from_region': 'a', 'to_region': 'b', 'passage_id': 'ok'},
        {'from_region': 'a', 'to_region': 'a', 'passage_id': 'bad'},
    ]
    valid = filter_valid_connectivity(rows, log=False)
    assert len(valid) == 1
    assert valid[0]['passage_id'] == 'ok'


def test_sanitize_world_layout_fixture():
    layout = json.loads(
        (Path(__file__).parent / 'fixtures' / 'episode_tiny' / 'world_layout-house_tiny.json').read_text()
    )
    clean = sanitize_world_layout(layout)
    assert all(c['from_region'] != c['to_region'] for c in clean['connectivity'])


def test_shortest_path_uses_exported_or_built_edges(tiny_graph):
    p = shortest_path(tiny_graph, 'n0', 'n8')
    assert p[0] == 'n0' and p[-1] == 'n8'
    assert nx.is_simple_path(tiny_graph, p)
