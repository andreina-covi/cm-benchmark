"""Deterministic ego-direction helpers for draft generation (no invented facts)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from cm_benchmark.generation.constructs import angle_relation_to_ego_label
from cm_benchmark.utils.spatial_relations import get_direction_angle
from cm_benchmark.utils.spatial_transformer import world_to_local


def _relation_thresholds(episode: dict) -> tuple[float, float]:
    thr = (episode.get('thresholds') or {}).get('relation') or {}
    angle = float(thr.get('lateral_deg', 15))
    vertical = float(thr.get('vertical_m', 0.1))
    return angle, vertical


def ego_label_from_world_pose(
    agent_pos,
    agent_rot,
    world_pos,
    episode: dict,
) -> Optional[str]:
    """Map agent pose + world object pose → horizontal MC ego label."""
    if agent_pos is None or agent_rot is None or world_pos is None:
        return None
    try:
        local = world_to_local(agent_pos, agent_rot, world_pos)
    except Exception:
        return None
    angle_thr, vertical_thr = _relation_thresholds(episode)
    angle_relation = get_direction_angle(
        np.asarray(local, dtype=float), angle_thr, vertical_thr
    )
    return angle_relation_to_ego_label(list(angle_relation))


def agent_pose_at_step(episode: dict, step_idx: int) -> tuple[Optional[Any], Optional[Any]]:
    for step in episode.get('steps') or []:
        if int(step.get('step')) == int(step_idx):
            agent = step.get('agent') or {}
            return agent.get('position'), agent.get('rotation')
    for entry in episode.get('agent_trajectory') or []:
        if int(entry.get('step', -1)) == int(step_idx):
            return entry.get('position'), entry.get('rotation')
    return None, None
