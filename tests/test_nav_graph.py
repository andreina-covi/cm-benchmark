"""Unit tests for offline nav_graph helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import pytest

from cm_benchmark.generation.nav_graph import (
    _spl_ratio,
    action_sequence_reaches_goal,
    build_nav_graph,
    calibrate_view_radius_m,
    derive_turns,
    filter_traversed_to_exported,
    filter_valid_connectivity,
    format_nav_actions,
    format_turn_sequence,
    is_valid_untraversed_shortcut,
    parse_nav_actions,
    path_exists,
    path_to_nav_actions,
    perturb_turn_sequence,
    sanitize_world_layout,
    score_route_action_sequence,
    score_survey_action_sequence,
    shortest_path,
    snap_position_to_graph,
    snap_trajectory_to_graph,
    trajectory_snap_tolerance,
    subgraph_from_traversed_edges,
    traversed_edges_from_snapped,
    viewed_edges_from_trajectory,
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


def test_trajectory_snap_tolerance_covers_export_grid_mismatch():
    """0.20 m unsnapped steps on a 0.15 m grid exceed grid/2 (0.075 m)."""
    raw = {
        'snapshots': {
            'episode_start': {
                'params': {
                    'grid_size': 0.15,
                    'agent_move_m': 0.2,
                    'agent_rotation_deg': 45.0,
                },
                'nodes': [
                    {'node_id': 'n0', 'x': 0.0, 'y': 1.0, 'z': 0.0},
                    {'node_id': 'n1', 'x': 0.15, 'y': 1.0, 'z': 0.0},
                ],
                'edges': [],
            }
        }
    }
    graph = build_nav_graph(raw)
    assert graph.graph['agent_move_m'] == 0.2
    # 0.10 m from n0: half-grid (0.075) rejects; trajectory snap accepts.
    assert snap_position_to_graph(graph, (0.0, 1.0, 0.10)) is None
    snapped = snap_trajectory_to_graph(
        graph,
        [
            {'step': 0, 'position': (0.0, 1.0, 0.10)},  # 0.10 m > 0.075
            {'step': 1, 'position': (0.15, 1.0, 0.22)},  # 0.22 m, still one step
            {'step': 2, 'position': (5.0, 1.0, 5.0)},  # far
        ],
    )
    assert [r['node_id'] for r in snapped] == ['n0', 'n1']
    assert trajectory_snap_tolerance(graph) > 0.15 / 2.0


def test_traversed_edges_are_undirected_set(tiny_graph):
    traj = [
        {'step': 0, 'position': (0.0, 1.0, 0.0)},
        {'step': 1, 'position': (0.25, 1.0, 0.0)},
        {'step': 2, 'position': (0.5, 1.0, 0.0)},
        {'step': 3, 'position': (0.25, 1.0, 0.0)},  # backtrack
        {'step': 4, 'position': (0.0, 1.0, 0.0)},
    ]
    snapped = snap_trajectory_to_graph(tiny_graph, traj)
    edges = traversed_edges_from_snapped(snapped)
    # Backtracking collapses: only n0–n1 and n1–n2 once each.
    assert edges == {('n0', 'n1'), ('n1', 'n2')}
    sub = subgraph_from_traversed_edges(tiny_graph, edges)
    assert path_exists(sub, 'n0', 'n2')
    assert not path_exists(sub, 'n0', 'n8')
    assert shortest_path(sub, 'n0', 'n2') == ['n0', 'n1', 'n2']


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


def test_parse_nav_actions_accepts_collected_and_prose():
    assert parse_nav_actions('move_ahead → rotate_left → move_ahead') == [
        'move_ahead',
        'rotate_left',
        'move_ahead',
    ]
    assert parse_nav_actions(
        [{'action': 'MoveAhead'}, {'action': 'rotate_right', 'degrees': 45}]
    ) == ['move_ahead', 'rotate_right']
    assert parse_nav_actions('look_up, move ahead, then rotate right') == [
        'move_ahead',
        'rotate_right',
    ]


def test_spl_ratio_is_success_gated_and_capped():
    assert _spl_ratio(2.0, 2.0, success=True) == 1.0
    assert _spl_ratio(2.0, 4.0, success=True) == 0.5
    assert _spl_ratio(2.0, 1.0, success=True) == 1.0
    assert _spl_ratio(2.0, 4.0, success=False) == 0.0
    assert _spl_ratio(None, 4.0, success=True) == 0.0
    assert _spl_ratio(0.0, 0.0, success=True) == 1.0


def test_count_direction_changes_counts_turns_not_rotation_tokens():
    """A 135° turn is three rotate tokens but one decision the model must make."""
    from cm_benchmark.generation.nav_graph import count_direction_changes

    assert count_direction_changes([]) == 0
    assert count_direction_changes(['move_ahead'] * 5) == 0
    assert count_direction_changes(['rotate_right'] * 3) == 1
    assert (
        count_direction_changes(
            ['move_ahead', 'rotate_right', 'rotate_right', 'move_ahead', 'rotate_left']
        )
        == 2
    )


def test_simplify_polyline_collapses_a_lattice_staircase():
    """A staircase around a straight line is one segment, not one per hop."""
    from cm_benchmark.generation.nav_graph import simplify_polyline_xz

    staircase = []
    for k in range(9):
        staircase.append((k * 0.15, 0.0, k * 0.075))
        staircase.append((k * 0.15, 0.0, k * 0.075 + 0.075))
    assert len(simplify_polyline_xz(staircase, 0.2)) == 2
    # A real 90° corner survives: tolerance must not erase genuine turns.
    corner = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0), (2.0, 0.0, 2.0)]
    assert len(simplify_polyline_xz(corner, 0.2)) == 3


def test_follow_path_actions_emits_runs_not_per_hop_rotations():
    """Greedy follower on a staircase yields one straight run of move_ahead."""
    import networkx as nx
    from cm_benchmark.generation.nav_graph import follow_path_actions

    g = nx.Graph()
    g.graph.update({'grid_size': 0.15, 'agent_move_m': 0.15, 'agent_rotation_deg': 45.0})
    path = []
    for k in range(13):
        nid = f's{k}'
        # 45° diagonal: on an 8-connected lattice this is a clean straight run.
        g.add_node(nid, pos=(k * 0.15, 0.0, k * 0.15))
        path.append(nid)
    for a, b in zip(path, path[1:]):
        g.add_edge(a, b, weight=0.15 * math.sqrt(2))

    followed = follow_path_actions(g, path, start_heading_deg=45.0)
    assert followed is not None
    assert followed['turn_count'] == 0
    assert set(followed['actions']) == {'move_ahead'}
    end_x, end_z = followed['end_xz']
    assert math.hypot(end_x - 12 * 0.15, end_z - 12 * 0.15) < 0.2


def test_path_to_nav_actions_roundtrip_reaches_goal(tiny_graph):
    from cm_benchmark.generation.nav_graph import path_start_heading_deg

    path = ['n0', 'n1', 'n2']
    heading = path_start_heading_deg(tiny_graph, path)
    actions = path_to_nav_actions(path, tiny_graph, start_heading_deg=heading)
    assert actions[0] == 'move_ahead'
    assert format_nav_actions(actions)
    scored = action_sequence_reaches_goal(
        tiny_graph, 'n0', 'n2', actions, start_heading_deg=heading
    )
    assert scored['parse_ok'] is True
    assert scored['reached_goal'] is True
    assert scored['success'] is True
    assert scored['end_node'] == 'n2'
    fail = action_sequence_reaches_goal(
        tiny_graph, 'n0', 'n2', 'rotate_left, rotate_right', start_heading_deg=heading
    )
    assert fail['reached_goal'] is False
    assert fail['success'] is False
    assert fail['efficiency'] == 0.0
    assert fail['route_efficiency'] == 0.0


def test_score_route_action_sequence_metric_not_node_walk(tiny_graph):
    from cm_benchmark.evaluation.score import score_route_knowledge
    from cm_benchmark.generation.nav_graph import path_start_heading_deg

    heading = path_start_heading_deg(tiny_graph, ['n0', 'n1', 'n2'])
    actions = path_to_nav_actions(
        ['n0', 'n1', 'n2'], tiny_graph, start_heading_deg=heading
    )
    walked = [('n0', 'n1'), ('n1', 'n2')]
    scored = score_route_action_sequence(
        tiny_graph, 'n0', 'n2', actions, walked, start_heading_deg=heading
    )
    assert scored['success'] is True
    assert scored['success_raw'] is True
    assert scored['validity'] is True
    assert scored['route_success'] is True
    assert scored['outcome'] == 'valid_success'
    assert scored['illegal_edges'] == []
    assert 0.0 < scored['efficiency'] <= 1.0
    assert scored['route_efficiency'] == scored['efficiency']
    assert scored['simulated_length_m'] > 0

    # Fake snap-only edge is dropped before validity / efficiency.
    mixed = [('n0', 'n1'), ('n1', 'n2'), ('n0', 'ghost')]
    assert ('n0', 'ghost') not in filter_traversed_to_exported(tiny_graph, mixed)

    # Off-graph walk rejects at the first pose that does not snap.
    off = score_route_action_sequence(
        tiny_graph,
        'n0',
        'n2',
        ['move_ahead'] * 8,
        walked,
        start_heading_deg=heading,
    )
    assert off['success'] is False
    assert off['validity'] is False
    assert off['efficiency'] == 0.0
    assert off['route_efficiency'] == 0.0
    assert off['outcome'] == 'invalid_fail'

    # Success is a goal-distance check, not exact node equality.
    near = score_route_action_sequence(
        tiny_graph,
        'n0',
        'n2',
        ['move_ahead'],
        walked,
        start_heading_deg=heading,
        goal_tolerance=0.3,
    )
    assert near['end_node'] == 'n1'
    assert near['success'] is True
    assert near['validity'] is True
    assert 0.0 < near['efficiency'] <= 1.0

    # Validity is independent: reaching via an untraversed edge is invalid.
    # SPL given success stays; construct SPL (route_efficiency) is 0.
    invalid = score_route_action_sequence(
        tiny_graph,
        'n0',
        'n2',
        actions,
        [('n0', 'n1')],
        start_heading_deg=heading,
    )
    assert invalid['success'] is True
    assert invalid['validity'] is False
    assert invalid['route_success'] is False
    assert invalid['outcome'] == 'invalid_success'
    assert ('n1', 'n2') in invalid['illegal_edges']
    assert 0.0 < invalid['efficiency'] <= 1.0
    assert invalid['route_efficiency'] == 0.0

    item = {
        'context': {
            'source_node': 'n0',
            'goal_node': 'n2',
            'traversed_edges': walked,
            'start_heading_deg': heading,
        }
    }
    wrapped = score_route_knowledge(item, tiny_graph, actions)
    assert wrapped['success'] is True
    assert wrapped['validity'] is True
    assert wrapped['route_success'] is True
    assert wrapped['efficiency'] == wrapped['route_efficiency']
    assert wrapped['illegal_edges'] == []


def test_survey_route_efficiency_is_spl_not_a_constant_zero(tiny_graph):
    """Survey's headline SPL is computed on viewed_edges, like route's on traversed."""
    from cm_benchmark.generation.nav_graph import (
        path_start_heading_deg,
        path_to_nav_actions,
        score_survey_action_sequence,
    )

    path = ['n0', 'n1', 'n2']
    heading = path_start_heading_deg(tiny_graph, path)
    actions = path_to_nav_actions(path, tiny_graph, start_heading_deg=heading)
    viewed = [('n0', 'n1'), ('n1', 'n2')]

    # Whole path is viewed but never walked: a valid success must score > 0.
    scored = score_survey_action_sequence(
        tiny_graph, 'n0', 'n2', actions, viewed, [], start_heading_deg=heading
    )
    assert scored['outcome'] == 'valid_success'
    assert 0.0 < scored['route_efficiency'] <= 1.0
    assert scored['route_efficiency'] == scored['efficiency']

    # Nothing novel (every edge already walked) is a success but not valid,
    # so the construct SPL drops to 0 while plain SPL given success stays.
    not_novel = score_survey_action_sequence(
        tiny_graph, 'n0', 'n2', actions, viewed, viewed, start_heading_deg=heading
    )
    assert not_novel['success'] is True
    assert not_novel['validity'] is False
    assert 0.0 < not_novel['efficiency'] <= 1.0
    assert not_novel['route_efficiency'] == 0.0


def test_viewed_edges_are_superset_of_exported_traversed(tiny_graph):
    traj = [
        {'step': 0, 'position': (0.0, 1.0, 0.0)},
        {'step': 1, 'position': (0.25, 1.0, 0.0)},
        {'step': 2, 'position': (0.5, 1.0, 0.0)},
    ]
    snapped = snap_trajectory_to_graph(tiny_graph, traj)
    traversed = traversed_edges_from_snapped(snapped)
    exported = filter_traversed_to_exported(tiny_graph, traversed)
    radius = calibrate_view_radius_m(
        tiny_graph,
        [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (3.0, 1.0, 0.0), (10.0, 1.0, 0.0)],
    )
    assert radius >= trajectory_snap_tolerance(tiny_graph)
    viewed, nodes = viewed_edges_from_trajectory(
        tiny_graph, traj, radius, traversed_edges=traversed
    )
    assert exported <= viewed
    for nid in (row['node_id'] for row in snapped):
        assert nid in nodes
    for a, b in viewed:
        assert tiny_graph.has_edge(a, b)
        assert a in nodes or (a, b) in exported or (b, a) in exported


def test_calibrate_view_radius_uses_scene_spacing_not_a_constant(tiny_graph):
    tight = calibrate_view_radius_m(
        tiny_graph,
        [(0.0, 1.0, 0.0), (0.8, 1.0, 0.0), (1.6, 1.0, 0.0)],
    )
    wide = calibrate_view_radius_m(
        tiny_graph,
        [(0.0, 1.0, 0.0), (4.0, 1.0, 0.0), (8.0, 1.0, 0.0)],
    )
    assert wide > tight
    assert tight >= trajectory_snap_tolerance(tiny_graph)


def test_score_survey_action_sequence_viewed_and_novel(tiny_graph):
    from cm_benchmark.evaluation.score import score_survey_based_route_planning
    from cm_benchmark.generation.nav_graph import path_start_heading_deg

    traversed = [('n0', 'n1'), ('n1', 'n2')]
    viewed = list(tiny_graph.edges())
    heading_walked = path_start_heading_deg(tiny_graph, ['n0', 'n1', 'n2'])
    walked_actions = path_to_nav_actions(
        ['n0', 'n1', 'n2'], tiny_graph, start_heading_deg=heading_walked
    )
    only_walked = score_survey_action_sequence(
        tiny_graph,
        'n0',
        'n2',
        walked_actions,
        viewed,
        traversed,
        start_heading_deg=heading_walked,
    )
    assert only_walked['success'] is True
    assert only_walked['validity'] is False
    assert only_walked['novel_edges'] == []
    assert 0.0 < only_walked['efficiency'] <= 1.0

    novel_path = ['n0', 'n3', 'n6', 'n7', 'n8']
    heading_novel = path_start_heading_deg(tiny_graph, novel_path)
    novel_actions = path_to_nav_actions(
        novel_path, tiny_graph, start_heading_deg=heading_novel
    )
    planned = score_survey_action_sequence(
        tiny_graph,
        'n0',
        'n8',
        novel_actions,
        viewed,
        traversed,
        start_heading_deg=heading_novel,
    )
    assert planned['success'] is True
    assert planned['validity'] is True
    assert planned['novel_edges']
    assert 0.0 < planned['efficiency'] <= 1.0

    unseen = score_survey_action_sequence(
        tiny_graph,
        'n0',
        'n8',
        novel_actions,
        traversed,
        traversed,
        start_heading_deg=heading_novel,
    )
    assert unseen['success'] is True
    assert unseen['validity'] is False
    assert unseen['illegal_edges']

    item = {
        'context': {
            'source_node': 'n0',
            'goal_node': 'n8',
            'viewed_edges': viewed,
            'traversed_edges': traversed,
            'start_heading_deg': heading_novel,
        }
    }
    wrapped = score_survey_based_route_planning(item, tiny_graph, novel_actions)
    assert wrapped['success'] is True
    assert wrapped['validity'] is True
