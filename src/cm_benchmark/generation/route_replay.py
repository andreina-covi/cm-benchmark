"""Replay recorded navigation actions through the continuous class-4 simulator."""

from __future__ import annotations

from typing import Optional

from cm_benchmark.generation.nav_graph import (
    episode_motion,
    simulate_nav_action_sequence,
    snap_position_to_graph,
    trajectory_snap_tolerance,
)


def _pose_delta_action(prev, cur, *, move_m: float, rotation_deg: float) -> Optional[str]:
    """Infer one nav token from consecutive recorded poses, or None if stationary."""
    from cm_benchmark.generation.nav_graph import NAV_ACTION_MOVE_AHEAD, NAV_ACTION_MOVE_BACK
    from cm_benchmark.generation.nav_graph import NAV_ACTION_ROTATE_LEFT, NAV_ACTION_ROTATE_RIGHT

    pp, pr = prev.get('position'), prev.get('rotation')
    cp, cr = cur.get('position'), cur.get('rotation')
    if not pp or not cp:
        return None
    dx = float(cp[0]) - float(pp[0])
    dz = float(cp[2]) - float(pp[2]) if len(cp) > 2 else 0.0
    dist = (dx * dx + dz * dz) ** 0.5
    yaw0 = float(pr[1] if pr is not None and len(pr) > 1 else 0.0)
    yaw1 = float(cr[1] if cr is not None and len(cr) > 1 else 0.0)
    dyaw = (yaw1 - yaw0 + 180.0) % 360.0 - 180.0
    if dist >= 0.5 * float(move_m):
        return NAV_ACTION_MOVE_AHEAD if dist > 0 else None
    if abs(dyaw) >= 0.5 * float(rotation_deg):
        return NAV_ACTION_ROTATE_RIGHT if dyaw > 0 else NAV_ACTION_ROTATE_LEFT
    return None


def action_log_vs_pose_deltas(episode: dict) -> dict:
    """Compare logged actions to pose-to-pose deltas (mismatch diagnosis)."""
    traj = episode.get('agent_trajectory') or []
    acts = episode.get('agent_actions') or []
    move_m, rot = episode_motion(episode)
    mismatches = []
    inferred = []
    for i in range(len(traj) - 1):
        inf = _pose_delta_action(traj[i], traj[i + 1], move_m=move_m, rotation_deg=rot)
        inferred.append(inf)
        logged = None
        if i < len(acts) and acts[i].get('action_success') is not False:
            logged = acts[i].get('action')
        if inf and logged and str(logged).lower().replace(' ', '_') != str(inf):
            mismatches.append(
                {'step': traj[i].get('step'), 'logged': logged, 'inferred_from_pose': inf}
            )
    return {
        'n_transitions': max(0, len(traj) - 1),
        'n_mismatched_transitions': len(mismatches),
        'first_mismatches': mismatches[:4],
        'inferred': inferred,
    }


def successful_recorded_actions(episode: dict) -> list[str]:
    """Actions that produced later poses: skip the start-row action, keep successes.

    Collector rows store the action that *arrived* at that step. Replaying from
    the first pose therefore applies ``actions[1:]`` (failed actions dropped).
    """
    rows = list(episode.get('agent_actions') or [])
    out: list[str] = []
    for rec in rows[1:]:
        if rec.get('action_success') is False:
            continue
        act = rec.get('action')
        if act:
            out.append(str(act))
    return out


_GLANCE_TURN = {'rotate_left': -1, 'rotate_right': 1}


def recorded_actions_between(
    episode: dict, t0: int, t1: int, *, skip_glances: bool = False
) -> list[str]:
    """Successful actions with ``t0 < step <= t1`` (arrival convention).

    ``skip_glances`` drops each SPOC glance (``scripted_glance``: consecutive
    turns toward an object and back) that lies wholly inside the window and
    turns back to its start heading; the walked route is the same without it.
    A glance the window cuts (start or end pose mid-glance) keeps its turns,
    which the agent needs from that pose.
    """
    return [act for _step, act in recorded_steps_between(episode, t0, t1, skip_glances=skip_glances)]


def recorded_steps_between(
    episode: dict, t0: int, t1: int, *, skip_glances: bool = False
) -> list[tuple[int, str]]:
    """``(step, action)`` pairs of ``recorded_actions_between``."""
    lo, hi = int(t0), int(t1)
    if hi < lo:
        lo, hi = hi, lo
    dropped = _complete_glance_steps(episode, lo, hi) if skip_glances else set()
    out: list[tuple[int, str]] = []
    for rec in episode.get('agent_actions') or []:
        try:
            step = int(rec.get('step'))
        except (TypeError, ValueError):
            continue
        if step <= lo or step > hi or step in dropped:
            continue
        if rec.get('action_success') is False:
            continue
        act = rec.get('action')
        if act:
            out.append((step, str(act)))
    return out


def collapse_rotations(actions: list[str], rotation_deg: float) -> list[str]:
    """Replace each run of consecutive turns by its net turn.

    Rotations in place do not move the agent, so the end pose is unchanged:
    ``rotate_right`` x8 (a full spin, e.g. SPOC done-override turns) vanishes,
    ``rotate_right, rotate_right, rotate_left`` becomes one ``rotate_right``.
    A net turn over half a revolution goes the short way; exactly half keeps
    the run's own direction.
    """
    per_rev = max(1, int(round(360.0 / float(rotation_deg))))
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        net = sum(_GLANCE_TURN[a] for a in run) % per_rev
        if net * 2 > per_rev:
            out.extend(['rotate_left'] * (per_rev - net))
        elif net * 2 == per_rev:
            first = 'rotate_left' if run[0] == 'rotate_left' else 'rotate_right'
            out.extend([first] * net)
        else:
            out.extend(['rotate_right'] * net)
        run.clear()

    for act in actions:
        if act in _GLANCE_TURN:
            run.append(act)
        else:
            flush()
            out.append(act)
    flush()
    return out


def _complete_glance_steps(episode: dict, lo: int, hi: int) -> set[int]:
    """Steps of glances inside ``(lo, hi]`` whose turns cancel out."""
    runs: list[list[dict]] = []
    prev_step = None
    for rec in episode.get('agent_actions') or []:
        try:
            step = int(rec.get('step'))
        except (TypeError, ValueError):
            continue
        if rec.get('scripted_glance'):
            if runs and prev_step is not None and step == prev_step + 1 and runs[-1]:
                runs[-1].append(rec)
            else:
                runs.append([rec])
        elif runs and runs[-1]:
            runs.append([])
        prev_step = step
    out: set[int] = set()
    for run in runs:
        if not run:
            continue
        steps = [int(r['step']) for r in run]
        if steps[0] <= lo or steps[-1] > hi:
            continue
        acts = [str(r.get('action')) for r in run if r.get('action_success') is not False]
        if any(a not in _GLANCE_TURN for a in acts):
            continue
        if sum(_GLANCE_TURN[a] for a in acts) == 0:
            out.update(steps)
    return out


def _yaw_of(pose) -> float:
    rot = (pose or {}).get('rotation') or [0.0, 0.0, 0.0]
    try:
        return float(rot[1] if len(rot) > 1 else rot[0])
    except (TypeError, ValueError, IndexError):
        return 0.0


def replay_recorded_actions(episode: dict) -> dict:
    """Simulate successful recorded actions; compare the end pose to the last recorded pose."""
    traj = episode.get('agent_trajectory') or []
    empty = {
        'match': False,
        'reason': 'missing_trajectory_or_graph',
        'scene': episode.get('scene') or episode.get('scene_id'),
        'n_actions': 0,
        'start_node': None,
        'goal_node': None,
        'end_node': None,
        'end_d_m': None,
        'end_dh_deg': None,
        'success': False,
    }
    if len(traj) < 2 or not (episode.get('nav_graph') or episode.get('nav_graph_path')):
        return empty
    from cm_benchmark.generation.nav_graph import load_episode_nav_graph

    graph = load_episode_nav_graph(episode)
    if graph is None:
        return {**empty, 'reason': 'nav_graph_unreadable'}
    start_pos = traj[0].get('position')
    goal_pos = traj[-1].get('position')
    heading = _yaw_of(traj[0])
    goal_yaw = _yaw_of(traj[-1])
    tol = trajectory_snap_tolerance(graph)
    start_node = snap_position_to_graph(graph, start_pos, tolerance=tol)
    goal_node = snap_position_to_graph(graph, goal_pos, tolerance=tol)
    actions = successful_recorded_actions(episode)
    sim = simulate_nav_action_sequence(
        graph,
        start_node or '',
        actions,
        start_heading_deg=heading,
        start_pos=start_pos,
    )
    end_xz = sim.get('end_xz')
    end_d = None
    if end_xz is not None and goal_pos is not None:
        gx = float(goal_pos[0])
        gz = float(goal_pos[2] if len(goal_pos) > 2 else goal_pos[1])
        end_d = (end_xz[0] - gx) ** 2
        end_d = (end_d + (end_xz[1] - gz) ** 2) ** 0.5
    end_h = sim.get('end_heading_deg')
    end_dh = None
    if end_h is not None:
        end_dh = abs((float(end_h) - goal_yaw + 180.0) % 360.0 - 180.0)
    goal_tol = float(sim.get('goal_tolerance') or 1.0)
    heading_ok = end_dh is None or end_dh <= 0.5 * float(graph.graph.get('agent_rotation_deg') or 1.0)
    match = bool(sim.get('ok')) and end_d is not None and end_d <= goal_tol and heading_ok
    delta = action_log_vs_pose_deltas(episode)
    return {
        'match': match,
        'reason': None if match else 'end_pose_mismatch',
        'action_vs_pose': delta,
        'scene': episode.get('scene') or episode.get('scene_id'),
        'n_actions': len(actions),
        'start_node': start_node,
        'goal_node': goal_node,
        'end_node': sim.get('end_node'),
        'end_d_m': end_d,
        'end_dh_deg': end_dh,
        'blocked_moves': sim.get('blocked_moves'),
        'block_tolerance': sim.get('block_tolerance'),
        'success': bool(sim.get('ok') and end_d is not None and end_d <= goal_tol),
        'parse_ok': bool(sim.get('parse_ok')),
        'sim': sim,
    }


def replay_collection(episodes: list[dict]) -> list[dict]:
    """Replay each episode; keep the full record (matches and mismatches)."""
    return [replay_recorded_actions(ep) for ep in episodes]
