"""[CODE] deterministic scorers. An LLM must never judge correctness."""

from __future__ import annotations

from typing import Any, Optional

from cm_benchmark.generation.nav_graph import (
    load_episode_nav_graph,
    score_route_action_sequence,
    score_survey_action_sequence,
    snap_trajectory_to_graph,
    traversed_edges_from_snapped,
    viewed_edges_from_trajectory,
)


def _item_extra(item: Optional[dict[str, Any]]) -> dict:
    rec = item or {}
    extra = rec.get('context') or rec.get('extra') or {}
    return extra if isinstance(extra, dict) else {}


def _nav_endpoints(item: Optional[dict[str, Any]]) -> tuple[Any, Any, float, Any, Any]:
    extra = _item_extra(item)
    rec = item or {}
    source_node = extra.get('source_node') or rec.get('source_node')
    goal_node = extra.get('goal_node') or rec.get('goal_node')
    heading = extra.get('start_heading_deg', rec.get('start_heading_deg', 0.0))
    start_pos = extra.get('start_pos') or rec.get('start_pos')
    goal_pos = extra.get('goal_pos') or rec.get('goal_pos')
    return source_node, goal_node, float(heading or 0.0), start_pos, goal_pos


def _item_edges(item: Optional[dict[str, Any]], key: str) -> list:
    extra = _item_extra(item)
    rec = item or {}
    edges = extra.get(key)
    if edges is None:
        edges = rec.get(key) or []
    return edges


def _graph_for_item(item, graph, episode):
    """Use the caller graph, else load episode_start nodes and edges from the SPOC file."""
    if graph is not None:
        return graph
    if episode is not None:
        return load_episode_nav_graph(episode)
    path = _item_extra(item).get('nav_graph_path')
    if not path:
        return None
    return load_episode_nav_graph({'nav_graph_path': path})


def _rebuild_edges(item, graph, episode, key: str) -> list:
    """Traversed / viewed edges from the episode trajectory and the SPOC graph.

    Slim items do not store those dumps. ``view_radius_m`` on the item keeps
    the survey radius that generation calibrated.
    """
    if episode is None or graph is None:
        return []
    snapped = snap_trajectory_to_graph(graph, episode.get('agent_trajectory') or [])
    traversed = traversed_edges_from_snapped(snapped)
    if key == 'traversed_edges':
        return [list(edge) for edge in traversed]
    radius = _item_extra(item).get('view_radius_m')
    if radius is None:
        return []
    viewed, _nodes = viewed_edges_from_trajectory(
        graph,
        episode.get('agent_trajectory') or [],
        float(radius),
        traversed_edges=traversed,
    )
    return [list(edge) for edge in viewed]


def _edges_for_score(item, graph, episode, key: str) -> list:
    stored = _item_edges(item, key)
    if stored:
        return stored
    return _rebuild_edges(item, graph, episode, key)


def _reference_actions(item: Optional[dict[str, Any]]):
    """Stored gold action sequence for nDTW, when present."""
    extra = _item_extra(item)
    rec = item or {}
    seq = extra.get('action_sequence')
    if seq is None:
        seq = rec.get('action_sequence')
    return seq


def score_route_knowledge(
    item: Optional[dict[str, Any]],
    graph,
    raw_actions,
    *,
    episode: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict:
    """Score a route_knowledge reply: success, SPL, nDTW/SDTW; validity diagnostic.

    When the item has no ``traversed_edges``, pass ``episode`` so the walked
    edges are rebuilt from the trajectory and the SPOC graph.
    """
    source_node, goal_node, heading, start_pos, goal_pos = _nav_endpoints(item)
    graph = _graph_for_item(item, graph, episode)
    kwargs.setdefault('reference_actions', _reference_actions(item))
    return score_route_action_sequence(
        graph,
        source_node,
        goal_node,
        raw_actions,
        _edges_for_score(item, graph, episode, 'traversed_edges'),
        start_heading_deg=heading,
        start_pos=start_pos,
        goal_pos=goal_pos,
        **kwargs,
    )


def score_survey_based_route_planning(
    item: Optional[dict[str, Any]],
    graph,
    raw_actions,
    *,
    episode: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict:
    """Score a survey reply: success (headline) and SPL; validity is diagnostic.

    When the item omits edge dumps, pass ``episode`` to rebuild them from the
    trajectory, ``view_radius_m``, and the SPOC graph.
    """
    source_node, goal_node, heading, start_pos, goal_pos = _nav_endpoints(item)
    graph = _graph_for_item(item, graph, episode)
    kwargs.setdefault('reference_actions', _reference_actions(item))
    return score_survey_action_sequence(
        graph,
        source_node,
        goal_node,
        raw_actions,
        _edges_for_score(item, graph, episode, 'viewed_edges'),
        _edges_for_score(item, graph, episode, 'traversed_edges'),
        start_heading_deg=heading,
        start_pos=start_pos,
        goal_pos=goal_pos,
        **kwargs,
    )
