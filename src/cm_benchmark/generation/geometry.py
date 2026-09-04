"""Deterministic ego-direction helpers for draft generation (no invented facts)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from cm_benchmark.generation.constructs import (
    AHEAD_HALF_WIDTH_FULL,
    local_offset_to_ego_label,
)
from cm_benchmark.utils.spatial_transformer import world_to_local


def ego_label_from_world_pose(
    agent_pos,
    agent_rot,
    world_pos,
    episode: dict = None,
    *,
    ahead_half_width: float = AHEAD_HALF_WIDTH_FULL,
) -> Optional[str]:
    """Map agent pose + world object pose → horizontal MC ego label.

    ``episode`` is accepted for call-site compatibility. Pass
    ``AHEAD_HALF_WIDTH_FOV`` when the object is constrained to the camera FOV.
    """
    del episode  # unused — wedge width is explicit via ahead_half_width
    if agent_pos is None or agent_rot is None or world_pos is None:
        return None
    try:
        local = world_to_local(agent_pos, agent_rot, world_pos)
    except Exception:
        return None
    return local_offset_to_ego_label(
        np.asarray(local, dtype=float), ahead_half_width=ahead_half_width
    )


def agent_pose_at_step(episode: dict, step_idx: int) -> tuple[Optional[Any], Optional[Any]]:
    for step in episode.get('steps') or []:
        if int(step.get('step')) == int(step_idx):
            agent = step.get('agent') or {}
            return agent.get('position'), agent.get('rotation')
    for entry in episode.get('agent_trajectory') or []:
        if int(entry.get('step', -1)) == int(step_idx):
            return entry.get('position'), entry.get('rotation')
    return None, None
