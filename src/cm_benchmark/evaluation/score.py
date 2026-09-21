"""[CODE] deterministic scorers. An LLM must never judge correctness."""

from __future__ import annotations

from typing import Any, Optional

from cm_benchmark.generation.nav_graph import (
    score_route_action_sequence,
    score_survey_action_sequence,
)


def _item_extra(item: Optional[dict[str, Any]]) -> dict:
    rec = item or {}
    extra = rec.get('context') or rec.get('extra') or {}
    return extra if isinstance(extra, dict) else {}


def _nav_endpoints(item: Optional[dict[str, Any]]) -> tuple[Any, Any, float]:
    extra = _item_extra(item)
    rec = item or {}
    source_node = extra.get('source_node') or rec.get('source_node')
    goal_node = extra.get('goal_node') or rec.get('goal_node')
    heading = extra.get('start_heading_deg', rec.get('start_heading_deg', 0.0))
    return source_node, goal_node, float(heading or 0.0)


def _item_edges(item: Optional[dict[str, Any]], key: str) -> list:
    extra = _item_extra(item)
    rec = item or {}
    edges = extra.get(key)
    if edges is None:
        edges = rec.get(key) or []
    return edges


def score_route_knowledge(
    item: Optional[dict[str, Any]],
    graph,
    raw_actions,
    **kwargs,
) -> dict:
    """Score a route_knowledge reply: success, validity, SPL, route SPL."""
    source_node, goal_node, heading = _nav_endpoints(item)
    return score_route_action_sequence(
        graph,
        source_node,
        goal_node,
        raw_actions,
        _item_edges(item, 'traversed_edges'),
        start_heading_deg=heading,
        **kwargs,
    )


def score_survey_based_route_planning(
    item: Optional[dict[str, Any]],
    graph,
    raw_actions,
    **kwargs,
) -> dict:
    """Score a survey reply on viewed_edges with a required untraversed hop."""
    source_node, goal_node, heading = _nav_endpoints(item)
    return score_survey_action_sequence(
        graph,
        source_node,
        goal_node,
        raw_actions,
        _item_edges(item, 'viewed_edges'),
        _item_edges(item, 'traversed_edges'),
        start_heading_deg=heading,
        **kwargs,
    )
