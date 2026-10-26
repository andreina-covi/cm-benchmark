"""Deterministic ego-direction helpers for item generation (no invented facts)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import numpy as np

from cm_benchmark.generation.config import get_generation_config
from cm_benchmark.generation.constructs import (
    angular_span_crosses_sector,
    bearings_cross_sector,
    bearing_deg_xz,
    bbox_edges_cross_sector,
    camera_horizontal_fov_deg,
    camera_size_wh,
    local_offset_to_ego_label,
)
from cm_benchmark.utils.spatial_transformer import world_to_local

# Answer-locking drops when refuse_straddling_objects rejects a direction.
_STRADDLE_DROPS: Counter = Counter()


def reset_straddle_drop_counts() -> None:
    """Clear per-construct straddle drop counters (call at generation start)."""
    _STRADDLE_DROPS.clear()


def straddle_drop_counts() -> dict[str, int]:
    """``{construct: n}`` answer labels refused for sector straddling."""
    return dict(_STRADDLE_DROPS)


def _record_straddle_drop(construct: Optional[str]) -> None:
    if construct:
        _STRADDLE_DROPS[construct] += 1


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


def aabb_corner_bearings(
    agent_pos,
    agent_rot,
    aabb_min,
    aabb_max,
) -> list[float]:
    """Yaw-only horizontal bearings of the eight AABB corners from the agent."""
    try:
        amin = [float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2])]
        amax = [float(aabb_max[0]), float(aabb_max[1]), float(aabb_max[2])]
    except (TypeError, ValueError, IndexError):
        return []
    corners = (
        (amin[0], amin[1], amin[2]),
        (amin[0], amin[1], amax[2]),
        (amin[0], amax[1], amin[2]),
        (amin[0], amax[1], amax[2]),
        (amax[0], amin[1], amin[2]),
        (amax[0], amin[1], amax[2]),
        (amax[0], amax[1], amin[2]),
        (amax[0], amax[1], amax[2]),
    )
    yaw_rot = _yaw_only_rotation(agent_rot)
    out: list[float] = []
    for corner in corners:
        try:
            local = world_to_local(agent_pos, yaw_rot, corner)
        except Exception:
            continue
        local = np.asarray(local, dtype=float)
        if local.shape[0] < 3 or not np.isfinite(local).all():
            continue
        bearing = bearing_deg_xz(float(local[0]), float(local[2]))
        if bearing is not None:
            out.append(bearing)
    return out


def aabb_corners_cross_sector(
    agent_pos,
    agent_rot,
    aabb_min,
    aabb_max,
    *,
    ahead_half_width: Optional[float] = None,
) -> bool:
    """True when AABB corners do not all share one ego sector (hidden objects)."""
    bearings = aabb_corner_bearings(agent_pos, agent_rot, aabb_min, aabb_max)
    if len(bearings) < 2:
        return False
    return bearings_cross_sector(bearings, ahead_half_width=ahead_half_width)


def direction_extent_crosses_sector(
    agent_pos,
    agent_rot,
    *,
    ahead_half_width: Optional[float] = None,
    episode: Optional[dict] = None,
    visible: bool = False,
    bbox=None,
    aabb_min=None,
    aabb_max=None,
    ang_width_deg: float | None = None,
    bearing_deg: float | None = None,
) -> bool:
    """True when the object's extent straddles a direction sector.

    Visible + bbox → both 2D left/right edges must share a sector (camera FOV).
    Otherwise → all eight AABB corners must share a sector.
    Fallback → center ± exported ``ang_width_deg`` (legacy).
    """
    if not get_generation_config().directions.refuse_straddling_objects:
        return False
    if ahead_half_width is None:
        ahead_half_width = get_generation_config().directions.ahead_half_width_full_deg

    if visible and bbox is not None and episode is not None:
        size = camera_size_wh(episode)
        fov_h = camera_horizontal_fov_deg(episode)
        if size is not None and fov_h is not None:
            return bbox_edges_cross_sector(
                bbox,
                size[0],
                fov_h,
                ahead_half_width=ahead_half_width,
            )

    if aabb_min is not None and aabb_max is not None and agent_pos is not None and agent_rot is not None:
        return aabb_corners_cross_sector(
            agent_pos,
            agent_rot,
            aabb_min,
            aabb_max,
            ahead_half_width=ahead_half_width,
        )

    if bearing_deg is not None:
        return angular_span_crosses_sector(
            bearing_deg, ang_width_deg, ahead_half_width=ahead_half_width
        )
    return False


def ego_label_from_world_pose(
    agent_pos,
    agent_rot,
    world_pos,
    episode: dict = None,
    *,
    ahead_half_width: Optional[float] = None,
    ang_width_deg: float | None = None,
    bbox=None,
    aabb_min=None,
    aabb_max=None,
    visible: bool = False,
    count_straddle_drop: Optional[str] = None,
) -> Optional[str]:
    """Map agent pose + world object pose → horizontal MC ego label.

    Bearing is yaw-only (pitch zeroed). Direction answers that straddle a sector
    are refused: visible objects use 2D bbox edge angles (camera FOV); hidden
    objects use the eight AABB corners; otherwise the legacy ``ang_width_deg``
    check. Pass ``AHEAD_HALF_WIDTH_FOV`` when the object is constrained to the
    camera FOV. ``count_straddle_drop`` is a construct id for the drop report.
    """
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
    if direction_extent_crosses_sector(
        agent_pos,
        agent_rot,
        ahead_half_width=ahead_half_width,
        episode=episode,
        visible=visible,
        bbox=bbox,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        ang_width_deg=ang_width_deg,
        bearing_deg=bearing,
    ):
        _record_straddle_drop(count_straddle_drop)
        return None
    return local_offset_to_ego_label(local, ahead_half_width=ahead_half_width)


def point_in_camera_frustum(
    agent_pos,
    agent_rot,
    world_pos,
    episode: Optional[dict] = None,
) -> Optional[bool]:
    """True when ``world_pos`` projects inside the camera frame (z>0, u/v in bounds).

    None when camera size / FOV or poses are missing — caller must not invent a gate.
    Uses the real camera pitch (not yaw-only) so the check matches the recorded view.
    """
    if agent_pos is None or agent_rot is None or world_pos is None:
        return None
    cam = {}
    if isinstance(episode, dict):
        cam = episode.get('camera') or (episode.get('episode_meta') or {}).get('camera') or {}
    try:
        width = int(cam.get('width'))
        height = int(cam.get('height'))
        fov_v = float(cam.get('fov_vertical_deg') or cam.get('fov_v'))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or fov_v <= 0:
        return None
    from cm_benchmark.utils.spatial_transformer import (
        get_focal_length,
        projection_with_local_vector,
        world_to_local,
    )

    try:
        local = world_to_local(agent_pos, agent_rot, world_pos)
    except Exception:
        return None
    local = np.asarray(local, dtype=float)
    if local.shape[0] < 3 or not np.isfinite(local).all():
        return None
    if float(local[2]) <= 0.0:
        return False
    fx, fy, _fov_h = get_focal_length(width, height, fov_v)
    try:
        u, v = projection_with_local_vector(
            (float(local[0]), float(local[1]), float(local[2])),
            (width / 2.0, height / 2.0),
            (fx, fy),
            {'ez': 0.0},
        )
    except (TypeError, ValueError):
        return False
    return 0.0 <= float(u) < float(width) and 0.0 <= float(v) < float(height)


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
