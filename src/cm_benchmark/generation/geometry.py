"""Deterministic ego-direction helpers for item generation (no invented facts)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from cm_benchmark.generation.config import get_generation_config
from cm_benchmark.generation.constructs import (
    angular_span_crosses_sector,
    bearing_deg_xz,
    local_offset_to_ego_label,
)
from cm_benchmark.utils.spatial_transformer import world_to_local


def _yaw_only_rotation(agent_rot):
    """Drop camera pitch. Horizontal labels use agent yaw, not the 28° nav mount.

    SPOC stores ``camera-horizon`` (look-down) in the pose passed as rotation[0]
    and the agent yaw in rotation[1]. Pitch changes ``atan2(x, z)`` in the tilted
    camera frame, so the same ground bearing gets different labels for objects
    above vs below the camera. Yaw-only keeps the label in the horizontal plane.
    Camera *position* is unchanged (the nav camera, not the body origin).
    """
    try:
        rot = [float(v) for v in agent_rot]
    except (TypeError, ValueError):
        return agent_rot
    if len(rot) < 2:
        return agent_rot
    rot[0] = 0.0
    return rot


def ego_label_from_world_pose(
    agent_pos,
    agent_rot,
    world_pos,
    episode: dict = None,
    *,
    ahead_half_width: Optional[float] = None,
    ang_width_deg: float | None = None,
) -> Optional[str]:
    """Map agent pose + world object pose → horizontal MC ego label.

    Bearing is yaw-only (pitch zeroed). If ``ang_width_deg`` straddles a sector
    boundary the label is refused: VSI-Bench and ViewSpatial-Bench both drop
    ambiguous directions, and neither publishes a ±10° cutoff to copy.
    ``episode`` is accepted for call-site compatibility. Pass
    ``AHEAD_HALF_WIDTH_FOV`` when the object is constrained to the camera FOV.
    """
    del episode  # unused — wedge width is explicit via ahead_half_width
    if ahead_half_width is None:
        ahead_half_width = get_generation_config().directions.ahead_half_width_full_deg
    if agent_pos is None or agent_rot is None or world_pos is None:
        return None
    try:
        local = world_to_local(agent_pos, _yaw_only_rotation(agent_rot), world_pos)
    except Exception:
        return None
    local = np.asarray(local, dtype=float)
    bearing = bearing_deg_xz(float(local[0]), float(local[2]))
    if bearing is None:
        return None
    if angular_span_crosses_sector(
        bearing, ang_width_deg, ahead_half_width=ahead_half_width
    ):
        return None
    return local_offset_to_ego_label(local, ahead_half_width=ahead_half_width)


def agent_pose_at_step(episode: dict, step_idx: int) -> tuple[Optional[Any], Optional[Any]]:
    """``(position, rotation)`` of the agent at a step (steps first, then trajectory)."""
    for step in episode.get('steps') or []:
        if int(step.get('step')) == int(step_idx):
            agent = step.get('agent') or {}
            return agent.get('position'), agent.get('rotation')
    for entry in episode.get('agent_trajectory') or []:
        if int(entry.get('step', -1)) == int(step_idx):
            return entry.get('position'), entry.get('rotation')
    return None, None
