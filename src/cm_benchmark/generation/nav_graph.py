"""Offline navigability graph helpers for class-4 constructs.

Built only from exported SPOC artifacts (`nav_graph-*.json`, agent trajectory).
No live AI2-THOR controller access.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Optional, Sequence, Union

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# 8-neighbor max step ≈ grid_size * sqrt(2)
_SQRT2 = math.sqrt(2.0)

TURN_LABELS = (
    'straight',
    'turn left',
    'sharp turn left',
    'turn right',
    'sharp turn right',
    'turn around',
)

OPPOSITE_TURN = {
    'straight': 'turn around',
    'turn around': 'straight',
    'turn left': 'turn right',
    'turn right': 'turn left',
    'sharp turn left': 'sharp turn right',
    'sharp turn right': 'sharp turn left',
}


def _as_xyz(pos) -> Optional[tuple[float, float, float]]:
    if pos is None:
        return None
    if isinstance(pos, dict):
        try:
            return (float(pos['x']), float(pos.get('y', 0.0)), float(pos['z']))
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        try:
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        except (TypeError, ValueError):
            return None
    return None


def _xz_dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[2] - b[2])


def select_nav_graph_snapshot(
    nav_graph_json: dict, *, prefer: str = 'episode_start'
) -> dict:
    """Return one snapshot dict from the multi-snapshot export."""
    if not isinstance(nav_graph_json, dict):
        raise TypeError('nav_graph_json must be a dict')
    snapshots = nav_graph_json.get('snapshots') or {}
    if prefer in snapshots:
        return snapshots[prefer]
    # Flat single-snapshot shape (nodes at top level)
    if 'nodes' in nav_graph_json:
        return nav_graph_json
    if snapshots:
        return next(iter(snapshots.values()))
    raise ValueError('nav_graph_json has no snapshots/nodes')


def build_nav_graph(
    nav_graph_json: dict,
    *,
    snapshot: str = 'episode_start',
    use_exported_edges: bool = True,
) -> nx.Graph:
    """Build an undirected NetworkX graph from exported reachable positions.

    Prefers edges written by SPOC (`snapshots.*.edges`). If absent, connects
    nodes within one 8-neighbor step (``grid_size * sqrt(2)``).
    """
    snap = select_nav_graph_snapshot(nav_graph_json, prefer=snapshot)
    params = snap.get('params') or {}
    grid_size = float(params.get('grid_size') or 0.25)
    nodes = snap.get('nodes') or []
    if not nodes:
        raise ValueError('nav graph snapshot has no nodes')

    G = nx.Graph()
    G.graph['grid_size'] = grid_size
    G.graph['agent_rotation_deg'] = float(
        params.get('agent_rotation_deg') or 45.0
    )
    G.graph['snapshot'] = snap.get('snapshot') or snapshot
    G.graph['scene_id'] = snap.get('scene_id') or nav_graph_json.get('scene_id')

    for n in nodes:
        nid = n.get('node_id')
        if nid is None:
            continue
        xyz = (float(n['x']), float(n.get('y', 0.0)), float(n['z']))
        G.add_node(nid, pos=xyz, x=xyz[0], y=xyz[1], z=xyz[2])

    edges = snap.get('edges') if use_exported_edges else None
    if edges:
        for e in edges:
            a, b = e.get('from_node'), e.get('to_node')
            if a not in G or b not in G:
                continue
            w = float(e.get('cost') or e.get('distance_xz') or 0.0)
            if w <= 0:
                w = _xz_dist(G.nodes[a]['pos'], G.nodes[b]['pos'])
            G.add_edge(a, b, weight=w)
    else:
        # Build 8-connected adjacency from positions.
        max_step = grid_size * _SQRT2 + 1e-6
        ids = list(G.nodes)
        positions = [G.nodes[i]['pos'] for i in ids]
        for i, ni in enumerate(ids):
            pi = positions[i]
            for j in range(i + 1, len(ids)):
                d = _xz_dist(pi, positions[j])
                if 0 < d <= max_step:
                    G.add_edge(ni, ids[j], weight=d)

    return G


def snap_position_to_graph(
    graph: nx.Graph,
    world_pos,
    *,
    tolerance: Optional[float] = None,
) -> Optional[str]:
    """Nearest graph node within tolerance (default: half grid_size), else None."""
    xyz = _as_xyz(world_pos)
    if xyz is None or graph.number_of_nodes() == 0:
        return None
    grid = float(graph.graph.get('grid_size') or 0.25)
    tol = float(tolerance) if tolerance is not None else grid / 2.0
    best_id = None
    best_d = float('inf')
    for nid, data in graph.nodes(data=True):
        pos = data.get('pos')
        if pos is None:
            continue
        d = _xz_dist(xyz, pos)
        if d < best_d:
            best_d = d
            best_id = nid
    if best_id is None or best_d > tol:
        return None
    return best_id


def snap_landmark_to_graph(
    graph: nx.Graph,
    world_pos,
    *,
    max_distance_m: Optional[float] = None,
) -> Optional[str]:
    """Nearest navigable node to a landmark pose (may sit on a receptacle).

    Landmarks are often non-navigable (countertops, beds). We still map them to
    the closest floor cell for pathfinding, with a larger radius than agent
    trajectory snapping (default: max(1.5 m, 10 × grid_size)).
    """
    grid = float(graph.graph.get('grid_size') or 0.25)
    tol = (
        float(max_distance_m)
        if max_distance_m is not None
        else max(1.5, 10.0 * grid)
    )
    return snap_position_to_graph(graph, world_pos, tolerance=tol)


# Taxonomy / docs alias (survey endpoints: nearest navigable node).
snap_to_nearest_node = snap_landmark_to_graph


def snap_to_nearest_of(
    graph: nx.Graph,
    world_pos,
    candidate_nodes: Sequence[str],
    *,
    max_distance_m: Optional[float] = None,
) -> Optional[str]:
    """Nearest node among ``candidate_nodes`` within landmark-scale radius.

    Used for route_knowledge: name a landmark, but the endpoint must be a
    node the agent actually visited (subset of the snapped trajectory).
    """
    xyz = _as_xyz(world_pos)
    if xyz is None or not candidate_nodes:
        return None
    grid = float(graph.graph.get('grid_size') or 0.25)
    tol = (
        float(max_distance_m)
        if max_distance_m is not None
        else max(1.5, 10.0 * grid)
    )
    best_id = None
    best_d = float('inf')
    for nid in candidate_nodes:
        if nid not in graph:
            continue
        pos = graph.nodes[nid].get('pos')
        if pos is None:
            continue
        d = _xz_dist(xyz, pos)
        if d < best_d:
            best_d = d
            best_id = nid
    if best_id is None or best_d > tol:
        return None
    return best_id


def snap_trajectory_to_graph(
    graph: nx.Graph,
    agent_trajectory: Sequence[dict],
    *,
    tolerance: Optional[float] = None,
) -> list[dict]:
    """Map each agent pose to nearest node; drop steps beyond tolerance.

    Returns ordered list of ``{step, node_id, position, snapped}``.
    Consecutive duplicate node_ids are collapsed (agent standing still).
    """
    out: list[dict] = []
    last_node = None
    for entry in agent_trajectory or []:
        if not isinstance(entry, dict):
            continue
        step = entry.get('step', entry.get('timestep'))
        pos = entry.get('position') or entry.get('pos')
        nid = snap_position_to_graph(graph, pos, tolerance=tolerance)
        if nid is None:
            logger.debug(
                'trajectory step %s has no graph node within snap tolerance', step
            )
            continue
        if nid == last_node:
            continue
        out.append(
            {
                'step': step,
                'node_id': nid,
                'position': _as_xyz(pos),
                'snapped': True,
            }
        )
        last_node = nid
    return out


def traversed_node_ids(snapped_trajectory: Sequence[dict]) -> list[str]:
    return [row['node_id'] for row in snapped_trajectory if row.get('node_id')]


def shortest_path(
    graph: nx.Graph, source: str, target: str, *, weight: str = 'weight'
) -> list[str]:
    """Dijkstra shortest path on the navigability graph (node ids)."""
    if source not in graph or target not in graph:
        raise nx.NodeNotFound(f'{source!r} or {target!r} not in graph')
    return list(nx.shortest_path(graph, source, target, weight=weight))


def was_traversed(candidate_path: Sequence[str], traversed_nodes: Sequence[str]) -> bool:
    """True if candidate_path is a contiguous, order-preserving subsequence."""
    cand = list(candidate_path)
    trav = list(traversed_nodes)
    if not cand:
        return False
    if len(cand) > len(trav):
        return False
    n, m = len(trav), len(cand)
    for i in range(n - m + 1):
        if trav[i : i + m] == cand:
            return True
    return False


def is_valid_untraversed_shortcut(
    candidate_path: Sequence[str],
    traversed_nodes: Sequence[str],
    graph: nx.Graph,
    *,
    min_unique_fraction: float = 0.5,
) -> bool:
    """True if path is valid in the graph and not an experienced (near-)route.

    Rejects:
    - paths that appear as a contiguous subsequence of the traversed walk
    - paths that only trivially detour an experienced segment (share too many
      consecutive nodes / insufficient novel nodes relative to length)
    """
    cand = list(candidate_path)
    if len(cand) < 2:
        return False
    for i in range(len(cand) - 1):
        if not graph.has_edge(cand[i], cand[i + 1]):
            return False
    if was_traversed(cand, traversed_nodes):
        return False
    # Near-duplicate: a long contiguous overlap with the walked sequence.
    trav = list(traversed_nodes)
    if not trav:
        return True
    m = len(cand)
    max_overlap = 0
    for i in range(len(trav)):
        k = 0
        while i + k < len(trav) and k < m and trav[i + k] == cand[k]:
            k += 1
        # also check suffix/prefix windows of candidate against trav
        max_overlap = max(max_overlap, k)
        for start in range(1, m):
            k2 = 0
            while (
                i + k2 < len(trav)
                and start + k2 < m
                and trav[i + k2] == cand[start + k2]
            ):
                k2 += 1
            max_overlap = max(max_overlap, k2)
    # Require enough novel length beyond the longest experienced overlap.
    novel = m - max_overlap
    if novel / m < min_unique_fraction and max_overlap >= 2:
        return False
    # Same endpoints with nearly identical node set → trivial detour.
    if len(trav) >= 2 and cand[0] in trav and cand[-1] in trav:
        try:
            i0 = trav.index(cand[0])
            # last occurrence of end after i0
            i1 = None
            for j in range(len(trav) - 1, i0, -1):
                if trav[j] == cand[-1]:
                    i1 = j
                    break
            if i1 is not None and i1 > i0:
                walked_seg = set(trav[i0 : i1 + 1])
                cand_set = set(cand)
                if walked_seg and len(cand_set & walked_seg) / len(cand_set) > 0.8:
                    return False
        except ValueError:
            pass
    return True


def _heading_xz_deg(
    p0: tuple[float, float, float], p1: tuple[float, float, float]
) -> Optional[float]:
    dx, dz = p1[0] - p0[0], p1[2] - p0[2]
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return None
    # AI2-THOR: yaw 0 faces +Z; positive yaw typically CW from +Z in exports.
    return math.degrees(math.atan2(dx, dz)) % 360.0


def _signed_delta_deg(h0: float, h1: float) -> float:
    d = (h1 - h0 + 180.0) % 360.0 - 180.0
    return d


def _bin_turn_label(delta_deg: float, rotation_deg: float = 45.0) -> str:
    """Bin heading change to agent rotation granularity."""
    step = float(rotation_deg) or 45.0
    # Round to nearest rotation increment
    n = int(round(delta_deg / step))
    # Clamp to ±180 / step
    max_n = int(round(180.0 / step))
    n = max(-max_n, min(max_n, n))
    abs_n = abs(n)
    if n == 0:
        return 'straight'
    if abs_n * step >= 180.0 - step / 2:
        return 'turn around'
    # 1 step (~45°) → turn; ≥2 steps (~90°+) → sharp turn
    if n > 0:
        return 'sharp turn right' if abs_n >= 2 else 'turn right'
    return 'sharp turn left' if abs_n >= 2 else 'turn left'


def derive_turns(
    path_nodes: Sequence[str],
    graph: nx.Graph,
    *,
    rotation_deg: Optional[float] = None,
    landmark_at_node: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Classify heading change at each interior node.

    Returns list of ``{node_id, label, landmark}`` (landmark optional).
    """
    rot = float(
        rotation_deg
        if rotation_deg is not None
        else graph.graph.get('agent_rotation_deg') or 45.0
    )
    lm = landmark_at_node or {}
    nodes = list(path_nodes)
    if len(nodes) < 3:
        # Degenerate: single segment → one "straight"
        if len(nodes) == 2:
            return [
                {
                    'node_id': nodes[0],
                    'label': 'straight',
                    'landmark': lm.get(nodes[0]),
                }
            ]
        return []

    out: list[dict] = []
    for i in range(1, len(nodes) - 1):
        p_prev = graph.nodes[nodes[i - 1]].get('pos')
        p_cur = graph.nodes[nodes[i]].get('pos')
        p_next = graph.nodes[nodes[i + 1]].get('pos')
        if not p_prev or not p_cur or not p_next:
            continue
        h_in = _heading_xz_deg(p_prev, p_cur)
        h_out = _heading_xz_deg(p_cur, p_next)
        if h_in is None or h_out is None:
            label = 'straight'
        else:
            label = _bin_turn_label(_signed_delta_deg(h_in, h_out), rot)
        entry = {
            'node_id': nodes[i],
            'label': label,
            'landmark': lm.get(nodes[i]),
        }
        out.append(entry)
    return out


def format_turn_sequence(turns: Sequence[dict], *, compress_straight: bool = True) -> str:
    """Render derive_turns output as ``straight → turn left @ Doorway``.

    When ``compress_straight`` is true, consecutive unlabeled straights collapse
    to one ``straight`` so MC options stay decision-point sized.
    """
    parts: list[str] = []
    pending_straight = False
    for t in turns:
        lab = t.get('label') or 'straight'
        landmark = t.get('landmark')
        if compress_straight and lab == 'straight' and not landmark:
            pending_straight = True
            continue
        if pending_straight:
            parts.append('straight')
            pending_straight = False
        if landmark:
            parts.append(f'{lab} @ {landmark}')
        else:
            parts.append(lab)
    if pending_straight:
        parts.append('straight')
    return ' → '.join(parts) if parts else ''


def perturb_turn_sequence(
    turns: Sequence[dict], mode: str
) -> Optional[list[dict]]:
    """Mechanical distractors for route_knowledge MC options.

    Taxonomy names: reversed_sequence, swapped_two_turns,
    plausible_but_unwalked_route. Older aliases are accepted.
    """
    if not turns:
        return None
    seq = [dict(t) for t in turns]
    # Aliases → canonical
    if mode in ('opposite_direction',):
        mode = 'reversed_sequence'
    if mode in ('wrong_decision_point',):
        mode = 'swapped_two_turns'
    if mode in ('extra_turn', 'no_turn'):
        mode = 'plausible_but_unwalked_route'

    if mode == 'reversed_sequence':
        rev = list(reversed(seq))
        return rev if rev != seq else None

    if mode == 'swapped_two_turns':
        idxs = [i for i, t in enumerate(seq) if (t.get('label') or 'straight') != 'straight']
        if len(idxs) < 2:
            if len(seq) < 2:
                return None
            i, j = 0, 1
        else:
            i, j = idxs[0], idxs[1]
        seq[i]['label'], seq[j]['label'] = seq[j].get('label'), seq[i].get('label')
        seq[i]['landmark'], seq[j]['landmark'] = (
            seq[j].get('landmark'),
            seq[i].get('landmark'),
        )
        return seq

    if mode == 'plausible_but_unwalked_route':
        # Change turn count: insert a spurious turn (preferred) or drop one.
        if len(seq) >= 1:
            mid = dict(seq[len(seq) // 2])
            mid['label'] = 'turn left' if mid.get('label') == 'straight' else 'straight'
            seq.insert(len(seq) // 2, mid)
            return seq
        return None

    return None


def filter_valid_connectivity(
    connectivity: Iterable[dict], *, log: bool = True
) -> list[dict]:
    """Drop world_layout.connectivity rows where from_region == to_region."""
    valid: list[dict] = []
    for row in connectivity or []:
        a, b = row.get('from_region'), row.get('to_region')
        if a is not None and b is not None and a == b:
            if log:
                logger.warning(
                    'invalid world_layout.connectivity (from_region == to_region): %s',
                    row.get('passage_id') or row,
                )
            continue
        valid.append(row)
    return valid


def sanitize_world_layout(layout: Optional[dict]) -> Optional[dict]:
    """Return a copy of world_layout with invalid connectivity filtered."""
    if not layout:
        return layout
    out = dict(layout)
    out['connectivity'] = filter_valid_connectivity(layout.get('connectivity') or [])
    return out


def distance_label_from_xz(
    dist_m: float, episode: Optional[dict] = None
) -> str:
    thr = ((episode or {}).get('thresholds') or {}).get('distance_label') or {
        'within_reach': [0.0, 0.5],
        'nearby': [0.5, 1.0],
        'far': [1.0, 1.5],
        'beyond': [1.5, 1e9],
    }
    for name, bounds in thr.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo <= dist_m < hi:
            return name
    return 'beyond'


def direction_distance_between_landmarks(
    pos_a, pos_b, *, episode: Optional[dict] = None
) -> Optional[tuple[str, str]]:
    """Allocentric-ish relation of B relative to A in world xz (+Z = front).

    Returns ``(direction_label, distance_label)`` or None.
    """
    a = _as_xyz(pos_a)
    b = _as_xyz(pos_b)
    if a is None or b is None:
        return None
    dx, dz = b[0] - a[0], b[2] - a[2]
    dist = math.hypot(dx, dz)
    if dist < 1e-6:
        return None
    # Treat world +Z as "front", +X as "right" (AI2-THOR yaw-0 frame).
    angle_thr = float(
        ((episode or {}).get('thresholds') or {}).get('relation', {}).get(
            'lateral_deg', 15
        )
    )
    # Primary axis
    abs_x, abs_z = abs(dx), abs(dz)
    # Angle from +Z toward +X
    deg = abs(math.degrees(math.atan2(dx, dz)))
    if abs_z >= abs_x and deg <= (90 - angle_thr):
        direction = 'ahead of' if dz > 0 else 'behind'
    elif abs_x >= abs_z:
        direction = 'to the right of' if dx > 0 else 'to the left of'
    else:
        direction = 'ahead of' if dz > 0 else 'behind'
    return direction, distance_label_from_xz(dist, episode)


def format_survey_relation(
    direction: str, distance: str, *, source_name: str
) -> str:
    if direction in ('ahead of', 'behind'):
        return f'{direction} the {source_name} and {distance}'
    return f'{direction} the {source_name} and {distance}'


def remove_edges_near_position(
    graph: nx.Graph,
    world_pos,
    *,
    radius_m: float = 0.6,
) -> nx.Graph:
    """Copy of ``graph`` with edges whose midpoint is within ``radius_m`` of pos removed.

    Used for conditional_detour: block a recorded closed passage without inventing
    edges — only existing nearby edges are dropped.
    """
    xyz = _as_xyz(world_pos)
    if xyz is None:
        return graph.copy()
    G = graph.copy()
    to_drop = []
    for u, v, data in G.edges(data=True):
        pu = G.nodes[u].get('pos')
        pv = G.nodes[v].get('pos')
        if not pu or not pv:
            continue
        mid = ((pu[0] + pv[0]) / 2.0, (pu[1] + pv[1]) / 2.0, (pu[2] + pv[2]) / 2.0)
        if _xz_dist(xyz, mid) <= radius_m:
            to_drop.append((u, v))
    G.remove_edges_from(to_drop)
    return G


def first_hop_direction_label(
    graph: nx.Graph, path_nodes: Sequence[str], *, source_pos=None, goal_pos=None
) -> Optional[str]:
    """Initial travel direction of ``path_nodes`` from an A-facing-B frame.

    Standing at source facing goal, report where the first hop heads
    (ahead / left / right / behind). Falls back to world +Z frame if goal
    pose is missing.
    """
    nodes = list(path_nodes)
    if len(nodes) < 2:
        return None
    p0 = graph.nodes[nodes[0]].get('pos')
    p1 = graph.nodes[nodes[1]].get('pos')
    if not p0 or not p1:
        return None
    # Prefer relational A→B heading when both poses given
    if source_pos is not None and goal_pos is not None:
        from cm_benchmark.generation.constructs import (
            imagined_perspective_label,
            xyz_as_dict,
        )

        a = xyz_as_dict(source_pos)
        b = xyz_as_dict(goal_pos)
        c = {'x': p1[0], 'y': p1[1], 'z': p1[2]}
        if a and b:
            return imagined_perspective_label(a, b, c)
    # World-frame fallback
    dx, dz = p1[0] - p0[0], p1[2] - p0[2]
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return None
    if abs(dz) >= abs(dx):
        return 'ahead of you' if dz > 0 else 'behind you'
    return 'to your right' if dx > 0 else 'to your left'
