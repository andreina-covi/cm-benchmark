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
    G.graph['agent_move_m'] = float(params.get('agent_move_m') or grid_size)
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


def trajectory_snap_tolerance(graph: nx.Graph) -> float:
    """Default agent-pose snap radius: one 8-neighbor step of the coarser lattice.

    Exports set ``snap_to_grid=False`` and often ``agent_move_m != grid_size``
    (e.g. 0.20 m steps on a 0.15 m GetReachablePositions grid). Half
    ``grid_size`` (0.075 m at 0.15) then drops legitimate walked poses that
    sit between cells or beside a missing reachable node. One 8-neighbor
    step of ``max(grid_size, agent_move_m)`` still rejects poses that are
    actually far from the graph.
    """
    grid = float(graph.graph.get('grid_size') or 0.25)
    move = float(graph.graph.get('agent_move_m') or grid)
    return max(grid, move) * _SQRT2 + 1e-6


def nodes_within_radius(
    graph: nx.Graph,
    world_pos,
    radius: float,
) -> list[tuple[str, float]]:
    """Graph nodes whose xz position is within ``radius`` of ``world_pos``.

    Same distance used by ``snap_position_to_graph``; this returns every hit,
    not only the nearest.
    """
    xyz = _as_xyz(world_pos)
    if xyz is None or graph.number_of_nodes() == 0:
        return []
    r = float(radius)
    hits: list[tuple[str, float]] = []
    for nid, data in graph.nodes(data=True):
        pos = data.get('pos')
        if pos is None:
            continue
        d = _xz_dist(xyz, pos)
        if d <= r:
            hits.append((nid, d))
    return hits


def snap_position_to_graph(
    graph: nx.Graph,
    world_pos,
    *,
    tolerance: Optional[float] = None,
) -> Optional[str]:
    """Nearest graph node within tolerance, else None.

    Default tolerance is half ``grid_size`` (strict on-node). Trajectory
    snapping should pass ``trajectory_snap_tolerance(graph)`` instead.
    """
    grid = float(graph.graph.get('grid_size') or 0.25)
    tol = float(tolerance) if tolerance is not None else grid / 2.0
    hits = nodes_within_radius(graph, world_pos, tol)
    if not hits:
        return None
    return min(hits, key=lambda item: item[1])[0]


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

    Default tolerance is ``trajectory_snap_tolerance`` (one 8-neighbor step
    of ``max(grid_size, agent_move_m)``), not half ``grid_size``.

    Returns ordered list of ``{step, node_id, position, snapped}``.
    Consecutive duplicate node_ids are collapsed (agent standing still).
    """
    out: list[dict] = []
    last_node = None
    tol = (
        float(tolerance)
        if tolerance is not None
        else trajectory_snap_tolerance(graph)
    )
    for entry in agent_trajectory or []:
        if not isinstance(entry, dict):
            continue
        step = entry.get('step', entry.get('timestep'))
        pos = entry.get('position') or entry.get('pos')
        nid = snap_position_to_graph(graph, pos, tolerance=tol)
        if nid is None:
            logger.warning(
                'trajectory step %s has no graph node within snap tolerance; '
                'rejecting pose rather than snapping far',
                step,
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


def traversed_edges_from_snapped(
    snapped_trajectory: Sequence[dict],
) -> set[tuple[str, str]]:
    """Undirected, deduplicated edge set from consecutive snapped agent poses.

    Collapses backtracking / repeated visits into set membership — not a walk
    sequence. Edge keys are sorted ``(node_a, node_b)`` pairs.
    """
    edges: set[tuple[str, str]] = set()
    nodes = [row['node_id'] for row in snapped_trajectory if row.get('node_id')]
    for a, b in zip(nodes, nodes[1:]):
        if a == b:
            continue
        edges.add((a, b) if a <= b else (b, a))
    return edges


def calibrate_view_radius_m(
    graph: nx.Graph,
    positions: Sequence,
    *,
    percentile: float = 0.5,
) -> float:
    """Scene-calibrated vista radius from typical landmark/room xz spacing.

    Median pairwise xz distance of the supplied poses, floored at
    ``trajectory_snap_tolerance`` so viewed_nodes cover snapped trajectory
    nodes (viewed_edges ⊇ exported traversed_edges). Not a class-4 pair
    reject gate — pair length is geodesic metres on the full nav_graph.
    """
    floor = trajectory_snap_tolerance(graph)
    coords: list[tuple[float, float]] = []
    for pos in positions or []:
        xyz = _as_xyz(pos)
        if xyz is None:
            continue
        coords.append((xyz[0], xyz[2]))
    dists: list[float] = []
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            d = math.hypot(a[0] - b[0], a[1] - b[1])
            if d > 1e-9:
                dists.append(d)
    if not dists:
        return floor
    dists.sort()
    idx = max(0, int(float(percentile) * (len(dists) - 1)))
    return max(floor, float(dists[idx]))


def viewed_nodes_from_trajectory(
    graph: nx.Graph,
    agent_trajectory: Sequence[dict],
    radius: float,
) -> set[str]:
    """Nodes within ``radius`` of any recorded agent pose (not just snaps)."""
    viewed: set[str] = set()
    r = float(radius)
    for entry in agent_trajectory or []:
        if not isinstance(entry, dict):
            continue
        pos = entry.get('position') or entry.get('pos')
        for nid, _d in nodes_within_radius(graph, pos, r):
            viewed.add(nid)
    return viewed


def viewed_edges_from_nodes(
    graph: nx.Graph, viewed_nodes: Iterable[str]
) -> set[tuple[str, str]]:
    """Exported graph edges whose both endpoints are viewed."""
    viewed = set(viewed_nodes or [])
    out: set[tuple[str, str]] = set()
    for a, b in graph.edges():
        if a in viewed and b in viewed:
            out.add((a, b) if a <= b else (b, a))
    return out


def viewed_edges_from_trajectory(
    graph: nx.Graph,
    agent_trajectory: Sequence[dict],
    radius: float,
    *,
    traversed_edges: Optional[Iterable[tuple[str, str]]] = None,
) -> tuple[set[tuple[str, str]], set[str]]:
    """Radius-proxy viewed_edges (superset of exported traversed_edges).

    Depth / raycasting is deferred. Union with exported traversed edges so the
    inclusion holds even if a pose snaps but sits just outside ``radius``.
    """
    nodes = viewed_nodes_from_trajectory(graph, agent_trajectory, radius)
    edges = viewed_edges_from_nodes(graph, nodes)
    if traversed_edges:
        edges |= filter_traversed_to_exported(graph, traversed_edges)
    return edges, nodes


def subgraph_from_traversed_edges(
    graph: nx.Graph, traversed_edges: Iterable[tuple[str, str]]
) -> nx.Graph:
    """Copy of ``graph`` restricted to ``traversed_edges`` only.

    Does not re-infer adjacency from distance. Edges present in the exported
    graph keep their weights; a consecutive snap pair missing from the export
    is still added (agent walked it) with an xz weight, and logged.
    """
    G = nx.Graph()
    G.graph.update(dict(graph.graph))
    for edge in traversed_edges:
        if not edge or len(edge) != 2:
            continue
        a, b = edge[0], edge[1]
        if a not in graph or b not in graph:
            continue
        if a not in G:
            G.add_node(a, **dict(graph.nodes[a]))
        if b not in G:
            G.add_node(b, **dict(graph.nodes[b]))
        if graph.has_edge(a, b):
            G.add_edge(a, b, **dict(graph.edges[a, b]))
        else:
            pa = graph.nodes[a].get('pos')
            pb = graph.nodes[b].get('pos')
            w = _xz_dist(pa, pb) if pa and pb else 1.0
            G.add_edge(a, b, weight=w)
            logger.debug(
                'traversed snap edge %s–%s absent from exported nav_graph; '
                'kept from trajectory',
                a,
                b,
            )
    return G


def path_exists(graph: nx.Graph, source: str, target: str) -> bool:
    """True if ``source`` and ``target`` are connected in ``graph``."""
    if source not in graph or target not in graph:
        return False
    return nx.has_path(graph, source, target)


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


# Collected-export action names (SPOC / AI2-THOR navigation CSV).
NAV_ACTION_MOVE_AHEAD = 'move_ahead'
NAV_ACTION_MOVE_BACK = 'move_back'
NAV_ACTION_ROTATE_LEFT = 'rotate_left'
NAV_ACTION_ROTATE_RIGHT = 'rotate_right'

NAV_ACTION_ALIASES = {
    'move_ahead': NAV_ACTION_MOVE_AHEAD,
    'moveahead': NAV_ACTION_MOVE_AHEAD,
    'move ahead': NAV_ACTION_MOVE_AHEAD,
    'forward': NAV_ACTION_MOVE_AHEAD,
    'straight': NAV_ACTION_MOVE_AHEAD,
    'move_back': NAV_ACTION_MOVE_BACK,
    'moveback': NAV_ACTION_MOVE_BACK,
    'move back': NAV_ACTION_MOVE_BACK,
    'backward': NAV_ACTION_MOVE_BACK,
    'back': NAV_ACTION_MOVE_BACK,
    'rotate_left': NAV_ACTION_ROTATE_LEFT,
    'rotateleft': NAV_ACTION_ROTATE_LEFT,
    'rotate left': NAV_ACTION_ROTATE_LEFT,
    'turn left': NAV_ACTION_ROTATE_LEFT,
    'turn_left': NAV_ACTION_ROTATE_LEFT,
    'rotate_right': NAV_ACTION_ROTATE_RIGHT,
    'rotateright': NAV_ACTION_ROTATE_RIGHT,
    'rotate right': NAV_ACTION_ROTATE_RIGHT,
    'turn right': NAV_ACTION_ROTATE_RIGHT,
    'turn_right': NAV_ACTION_ROTATE_RIGHT,
}

# Look / idle tokens from the stream — ignore for xz graph walking.
_NAV_ACTION_IGNORE = frozenset(
    {
        'look_up',
        'lookup',
        'look up',
        'look_down',
        'lookdown',
        'look down',
        'pass',
        'none',
        'stop',
        'done',
    }
)


def canonicalize_nav_action(token: str) -> Optional[str]:
    """Map a free-text token onto the collected action vocabulary, or None."""
    if token is None:
        return None
    raw = str(token).strip()
    if not raw:
        return None
    key = raw.lower().replace('-', '_').strip()
    key = ' '.join(key.split())
    compact = key.replace('_', '').replace(' ', '')
    if key in _NAV_ACTION_IGNORE or compact in {s.replace('_', '').replace(' ', '') for s in _NAV_ACTION_IGNORE}:
        return None
    if key in NAV_ACTION_ALIASES:
        return NAV_ACTION_ALIASES[key]
    if compact in NAV_ACTION_ALIASES:
        return NAV_ACTION_ALIASES[compact]
    # "rotate_right 90" / "MoveAhead," leftovers
    for alias, canon in NAV_ACTION_ALIASES.items():
        if key.startswith(alias + ' ') or key.startswith(alias + '_') or compact.startswith(
            alias.replace('_', '').replace(' ', '')
        ):
            return canon
    return None


def parse_nav_actions(raw) -> Optional[list[str]]:
    """Parse a model reply or collected list into canonical action names.

    Accepts a string (comma / arrow / newline separated), a list of strings,
    or collected ``[{action, degrees}, …]`` rows. Returns None if *no* token
    could be parsed (empty after ignore-only is ``[]``).
    """
    if raw is None:
        return None
    tokens: list[str] = []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        for sep in ('→', '->', ';', ',', '\n', '|'):
            text = text.replace(sep, ' ')
        tokens = [t for t in text.split() if t and t not in {'and', 'then', 'to'}]
        # Re-join split multiword aliases: "move ahead", "rotate left"
        merged: list[str] = []
        i = 0
        while i < len(tokens):
            pair = f'{tokens[i]} {tokens[i + 1]}' if i + 1 < len(tokens) else ''
            if pair.lower() in NAV_ACTION_ALIASES or pair.lower() in _NAV_ACTION_IGNORE:
                merged.append(pair)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        tokens = merged
    elif isinstance(raw, (list, tuple)):
        for entry in raw:
            if isinstance(entry, dict):
                act = entry.get('action')
                if act is not None:
                    tokens.append(str(act))
            elif entry is not None:
                tokens.append(str(entry))
    else:
        return None

    out: list[str] = []
    saw_unknown = False
    for tok in tokens:
        if canonicalize_nav_action(tok) is None and str(tok).strip().lower() in _NAV_ACTION_IGNORE:
            continue
        canon = canonicalize_nav_action(tok)
        if canon is None:
            # skip punctuation-only leftovers
            if str(tok).strip() in {'.', '?', '!', ':'}:
                continue
            saw_unknown = True
            continue
        out.append(canon)
    if saw_unknown and not out:
        return None
    return out


def format_nav_actions(actions: Sequence[str]) -> str:
    """Render canonical actions as ``move_ahead → rotate_left → move_ahead``."""
    return ' → '.join(a for a in actions if a)


def path_start_heading_deg(
    graph: nx.Graph, path_nodes: Sequence[str]
) -> Optional[float]:
    """Heading of the first hop (AI2-THOR yaw, +Z = 0)."""
    nodes = list(path_nodes)
    if len(nodes) < 2:
        return None
    p0 = graph.nodes[nodes[0]].get('pos')
    p1 = graph.nodes[nodes[1]].get('pos')
    if not p0 or not p1:
        return None
    return _heading_xz_deg(p0, p1)


def path_to_nav_actions(
    path_nodes: Sequence[str],
    graph: nx.Graph,
    *,
    start_heading_deg: Optional[float] = None,
    rotation_deg: Optional[float] = None,
) -> list[str]:
    """Convert a node path to collected-format actions (rotate_* + move_ahead).

    ``start_heading_deg`` defaults to the first-hop heading so a matching
    walk begins with ``move_ahead``.
    """
    rot = float(
        rotation_deg
        if rotation_deg is not None
        else graph.graph.get('agent_rotation_deg') or 45.0
    ) or 45.0
    nodes = list(path_nodes)
    if len(nodes) < 2:
        return []
    heading = start_heading_deg
    if heading is None:
        heading = path_start_heading_deg(graph, nodes)
    if heading is None:
        return []
    heading = float(heading) % 360.0
    out: list[str] = []
    for a, b in zip(nodes, nodes[1:]):
        pa = graph.nodes[a].get('pos')
        pb = graph.nodes[b].get('pos')
        if not pa or not pb:
            continue
        needed = _heading_xz_deg(pa, pb)
        if needed is None:
            continue
        delta = _signed_delta_deg(heading, needed)
        n = int(round(delta / rot))
        max_n = int(round(180.0 / rot))
        n = max(-max_n, min(max_n, n))
        if n > 0:
            out.extend([NAV_ACTION_ROTATE_RIGHT] * n)
        elif n < 0:
            out.extend([NAV_ACTION_ROTATE_LEFT] * abs(n))
        out.append(NAV_ACTION_MOVE_AHEAD)
        heading = needed
    return out


def count_direction_changes(actions: Sequence[str]) -> int:
    """Real turns in an action sequence: contiguous rotations count once.

    ``rotate_right, rotate_right, rotate_right`` is one 135° turn, not three.
    Used for class-4 difficulty gates, where raw rotation tokens overcount.
    """
    turns = 0
    prev_was_rotation = False
    for act in actions or []:
        is_rotation = act in (NAV_ACTION_ROTATE_LEFT, NAV_ACTION_ROTATE_RIGHT)
        if is_rotation and not prev_was_rotation:
            turns += 1
        prev_was_rotation = is_rotation
    return turns


def _perp_distance_xz(p, a, b) -> float:
    """Distance from ``p`` to segment ``a``–``b`` in the xz plane."""
    ax, az, bx, bz, px, pz = a[0], a[2], b[0], b[2], p[0], p[2]
    dx, dz = bx - ax, bz - az
    den = dx * dx + dz * dz
    if den < 1e-12:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / den))
    return math.hypot(px - (ax + t * dx), pz - (az + t * dz))


def simplify_polyline_xz(points: Sequence[Any], tolerance_m: float) -> list:
    """Ramer–Douglas–Peucker simplification in the xz plane.

    Collapses lattice staircases into the straight corridor they approximate.
    An axis-aligned grid cannot represent a walk at, say, yaw 285°, so
    ``shortest_path`` zig-zags; the underlying route is one straight run.

    Iterative: a near-straight path recurses once per point, and class-4 paths
    run to hundreds of nodes.
    """
    pts = [p for p in points if p is not None]
    if len(pts) < 3:
        return list(pts)
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        worst_i, worst_d = lo, -1.0
        for i in range(lo + 1, hi):
            d = _perp_distance_xz(pts[i], pts[lo], pts[hi])
            if d > worst_d:
                worst_i, worst_d = i, d
        if worst_d > tolerance_m:
            keep[worst_i] = True
            stack.append((lo, worst_i))
            stack.append((worst_i, hi))
    return [p for p, k in zip(pts, keep) if k]


def follow_path_actions(
    graph: nx.Graph,
    path_nodes: Sequence[str],
    *,
    start_heading_deg: Optional[float] = None,
    move_m: Optional[float] = None,
    rotation_deg: Optional[float] = None,
    simplify_tolerance_m: Optional[float] = None,
    max_actions: int = 512,
) -> Optional[dict]:
    """Greedy follower turning a node path into collected-format actions.

    Mirrors Habitat's ``GreedyGeodesicFollower``: the pose is continuous and
    only the heading is quantized to ``rotation_deg``, so a straight corridor
    becomes ``rotate × k`` then ``move_ahead × n``. Deriving one ``move_ahead``
    per lattice hop instead (``path_to_nav_actions``) emits a rotation at every
    staircase step and inflates the turn count by an order of magnitude.

    Returns ``{actions, start_heading_deg, turn_count, end_xz}`` or None.
    """
    nodes = [n for n in (path_nodes or []) if n in graph]
    if len(nodes) < 2:
        return None
    pts = [graph.nodes[n].get('pos') for n in nodes]
    if any(p is None for p in pts):
        return None
    rot = float(
        rotation_deg
        if rotation_deg is not None
        else graph.graph.get('agent_rotation_deg') or 45.0
    ) or 45.0
    step = float(
        move_m
        if move_m is not None
        else graph.graph.get('agent_move_m') or graph.graph.get('grid_size') or 0.25
    )
    if step <= 1e-9:
        return None
    tol = (
        float(simplify_tolerance_m)
        if simplify_tolerance_m is not None
        else max(step, float(graph.graph.get('grid_size') or 0.25)) * _SQRT2
    )
    waypoints = simplify_polyline_xz(pts, tol)
    if len(waypoints) < 2:
        return None

    x, z = float(pts[0][0]), float(pts[0][2])
    heading = start_heading_deg
    if heading is None:
        heading = _heading_xz_deg(pts[0], waypoints[1])
    if heading is None:
        return None
    heading = float(heading) % 360.0

    actions: list[str] = []
    arrive = step * 0.75
    for wp in waypoints[1:]:
        wx, wz = float(wp[0]), float(wp[2])
        stalled = 0
        while len(actions) < max_actions:
            dist = math.hypot(wx - x, wz - z)
            if dist <= arrive:
                break
            desired = math.degrees(math.atan2(wx - x, wz - z)) % 360.0
            delta = _signed_delta_deg(heading, desired)
            if abs(delta) > rot / 2.0:
                if delta > 0:
                    actions.append(NAV_ACTION_ROTATE_RIGHT)
                    heading = (heading + rot) % 360.0
                else:
                    actions.append(NAV_ACTION_ROTATE_LEFT)
                    heading = (heading - rot) % 360.0
                continue
            dx, dz = _heading_step_xz(heading, step)
            nx_, nz_ = x + dx, z + dz
            # Heading is quantized, so a move can overshoot; stop before it does.
            if math.hypot(wx - nx_, wz - nz_) >= dist:
                stalled += 1
                if stalled >= 2:
                    break
                continue
            x, z = nx_, nz_
            actions.append(NAV_ACTION_MOVE_AHEAD)
            stalled = 0
        if len(actions) >= max_actions:
            return None
    if not actions:
        return None
    return {
        'actions': actions,
        'start_heading_deg': float(start_heading_deg)
        if start_heading_deg is not None
        else _heading_xz_deg(pts[0], waypoints[1]),
        'turn_count': count_direction_changes(actions),
        'end_xz': (x, z),
    }


def neighbor_in_heading(
    graph: nx.Graph,
    node_id: str,
    heading_deg: float,
    *,
    half_width_deg: Optional[float] = None,
) -> Optional[str]:
    """Neighbor whose xz bearing is closest to ``heading_deg``, within a wedge."""
    if node_id not in graph:
        return None
    rot = float(graph.graph.get('agent_rotation_deg') or 45.0) or 45.0
    width = float(half_width_deg) if half_width_deg is not None else rot / 2.0 + 1e-6
    src = graph.nodes[node_id].get('pos')
    if not src:
        return None
    best = None
    best_err = float('inf')
    for nbr in graph.neighbors(node_id):
        dst = graph.nodes[nbr].get('pos')
        if not dst:
            continue
        bear = _heading_xz_deg(src, dst)
        if bear is None:
            continue
        err = abs(_signed_delta_deg(heading_deg, bear))
        if err <= width and err < best_err:
            best_err = err
            best = nbr
    return best


def execute_nav_actions(
    graph: nx.Graph,
    start_node: str,
    actions: Sequence[str],
    *,
    start_heading_deg: float = 0.0,
    rotation_deg: Optional[float] = None,
) -> tuple[Optional[str], float, bool]:
    """Walk ``actions`` on ``graph`` from ``start_node``.

    Discrete neighbor-hop helper (not the route scorer). A blocked
    ``move_ahead`` / ``move_back`` stops the walk and sets the flag False.
    """
    if start_node not in graph:
        return None, float(start_heading_deg) % 360.0, False
    rot = float(
        rotation_deg
        if rotation_deg is not None
        else graph.graph.get('agent_rotation_deg') or 45.0
    ) or 45.0
    node = start_node
    heading = float(start_heading_deg) % 360.0
    parsed = parse_nav_actions(list(actions))
    if parsed is None:
        return node, heading, False
    for act in parsed:
        if act == NAV_ACTION_ROTATE_RIGHT:
            heading = (heading + rot) % 360.0
        elif act == NAV_ACTION_ROTATE_LEFT:
            heading = (heading - rot) % 360.0
        elif act == NAV_ACTION_MOVE_AHEAD:
            nxt = neighbor_in_heading(graph, node, heading)
            if nxt is None:
                return node, heading, False
            node = nxt
        elif act == NAV_ACTION_MOVE_BACK:
            nxt = neighbor_in_heading(graph, node, (heading + 180.0) % 360.0)
            if nxt is None:
                return node, heading, False
            node = nxt
    return node, heading, True


def filter_traversed_to_exported(
    graph: nx.Graph, traversed_edges: Iterable[tuple[str, str]]
) -> set[tuple[str, str]]:
    """Keep only traversed edges that also exist on the exported graph."""
    out: set[tuple[str, str]] = set()
    for edge in traversed_edges or []:
        if not edge or len(edge) != 2:
            continue
        a, b = edge[0], edge[1]
        if a not in graph or b not in graph:
            continue
        if not graph.has_edge(a, b):
            continue
        out.add((a, b) if a <= b else (b, a))
    return out


def _heading_step_xz(heading_deg: float, step_m: float) -> tuple[float, float]:
    """AI2-THOR: yaw 0 faces +Z; positive yaw toward +X."""
    rad = math.radians(float(heading_deg) % 360.0)
    return step_m * math.sin(rad), step_m * math.cos(rad)


def _spl_ratio(opt_m: Optional[float], path_m: float, *, success: bool) -> float:
    """Anderson SPL: ``S * ℓ / max(p, ℓ)``. Failures are 0; never exceeds 1."""
    if not success:
        return 0.0
    if opt_m is None:
        return 0.0
    opt = float(opt_m)
    path = float(path_m)
    if opt <= 1e-9 and path <= 1e-9:
        return 1.0
    if opt <= 1e-9:
        return 0.0
    return opt / max(path, opt)


def _route_outcome(success: bool, validity: bool) -> str:
    if success and validity:
        return 'valid_success'
    if success:
        return 'invalid_success'
    if validity:
        return 'valid_fail'
    return 'invalid_fail'


def _route_score_record(
    *,
    success: bool,
    validity: bool,
    parse_ok: bool,
    end_node,
    actions,
    snapped_nodes=None,
    simulated_length_m: float = 0.0,
    crossed_edges=None,
    illegal_edges=None,
    novel_edges=None,
    shortest_path_m: Optional[float] = None,
    shortest_path_full_m: Optional[float] = None,
    efficiency: float = 0.0,
    route_efficiency: float = 0.0,
) -> dict:
    crossed = [tuple(e) for e in (crossed_edges or [])]
    illegal = [tuple(e) for e in (illegal_edges or [])]
    novel = [tuple(e) for e in (novel_edges or [])]
    return {
        'success': bool(success),
        'success_raw': bool(success),
        'validity': bool(validity),
        'route_success': bool(success and validity),
        'efficiency': float(efficiency),
        'route_efficiency': float(route_efficiency),
        'outcome': _route_outcome(success, validity),
        'parse_ok': bool(parse_ok),
        'end_node': end_node,
        'actions': list(actions or []),
        'snapped_nodes': list(snapped_nodes or []),
        'simulated_length_m': float(simulated_length_m),
        'shortest_path_m': shortest_path_m,
        'shortest_path_full_m': shortest_path_full_m,
        'crossed_edges': crossed,
        'illegal_edges': illegal,
        'novel_edges': novel,
    }


def _crossed_undirected(snapped: Sequence[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for a, b in zip(snapped, snapped[1:]):
        if a == b:
            continue
        out.append((a, b) if a <= b else (b, a))
    return out


def node_within_goal_tolerance(
    graph: nx.Graph, end_node, goal_node, goal_tol: float
) -> bool:
    """Public alias of the scorer's goal test, for generation-side verification."""
    return _goal_reached(graph, end_node, goal_node, goal_tol)


def _goal_reached(graph: nx.Graph, end_node, goal_node, goal_tol: float) -> bool:
    if end_node not in graph or goal_node not in graph:
        return False
    goal_pos = graph.nodes[goal_node].get('pos')
    end_pos = graph.nodes[end_node].get('pos')
    return bool(goal_pos and end_pos and _xz_dist(end_pos, goal_pos) <= goal_tol)


def _shortest_on_edges(
    graph: nx.Graph,
    edges: Iterable[tuple[str, str]],
    source_node: str,
    goal_node: str,
) -> Optional[float]:
    exported = filter_traversed_to_exported(graph, edges)
    if not exported:
        return None
    walked = subgraph_from_traversed_edges(graph, exported)
    try:
        return float(
            nx.shortest_path_length(walked, source_node, goal_node, weight='weight')
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def simulate_nav_action_sequence(
    graph: nx.Graph,
    source_node: str,
    raw_actions,
    *,
    start_heading_deg: float = 0.0,
    start_pos=None,
    move_m: Optional[float] = None,
    rotation_deg: Optional[float] = None,
    snap_tolerance: Optional[float] = None,
    goal_tolerance: Optional[float] = None,
) -> dict:
    """Metric walk shared by route and survey scorers.

    ``ok`` is False on unparseable input or the first pose that does not snap.
    """
    parsed = parse_nav_actions(raw_actions)
    snap_tol = (
        float(snap_tolerance)
        if snap_tolerance is not None
        else trajectory_snap_tolerance(graph)
    )
    goal_tol = (
        float(goal_tolerance) if goal_tolerance is not None else snap_tol
    )
    empty = {
        'ok': False,
        'parse_ok': parsed is not None,
        'actions': list(parsed or []),
        'snapped': [],
        'simulated_length_m': 0.0,
        'end_node': None,
        'snap_tolerance': snap_tol,
        'goal_tolerance': goal_tol,
    }
    if parsed is None:
        empty['parse_ok'] = False
        return empty
    if source_node not in graph:
        return empty
    rot = float(
        rotation_deg
        if rotation_deg is not None
        else graph.graph.get('agent_rotation_deg') or 45.0
    ) or 45.0
    step = float(
        move_m
        if move_m is not None
        else graph.graph.get('agent_move_m') or graph.graph.get('grid_size') or 0.25
    )
    heading = float(start_heading_deg) % 360.0
    origin = _as_xyz(start_pos) or graph.nodes[source_node].get('pos')
    if origin is None:
        return empty
    x, y, z = float(origin[0]), float(origin[1]), float(origin[2])
    start_nid = snap_position_to_graph(graph, (x, y, z), tolerance=snap_tol)
    if start_nid is None:
        return empty

    snapped = [start_nid]
    sim_len = 0.0
    for act in parsed:
        if act == NAV_ACTION_ROTATE_RIGHT:
            heading = (heading + rot) % 360.0
        elif act == NAV_ACTION_ROTATE_LEFT:
            heading = (heading - rot) % 360.0
        elif act == NAV_ACTION_MOVE_AHEAD:
            dx, dz = _heading_step_xz(heading, step)
            x += dx
            z += dz
            sim_len += step
        elif act == NAV_ACTION_MOVE_BACK:
            dx, dz = _heading_step_xz(heading, step)
            x -= dx
            z -= dz
            sim_len += step
        else:
            continue
        nid = snap_position_to_graph(graph, (x, y, z), tolerance=snap_tol)
        if nid is None:
            return {
                'ok': False,
                'parse_ok': True,
                'actions': parsed,
                'snapped': snapped,
                'simulated_length_m': sim_len,
                'end_node': snapped[-1],
                'snap_tolerance': snap_tol,
                'goal_tolerance': goal_tol,
            }
        if nid != snapped[-1]:
            snapped.append(nid)
    return {
        'ok': True,
        'parse_ok': True,
        'actions': parsed,
        'snapped': snapped,
        'simulated_length_m': sim_len,
        'end_node': snapped[-1],
        'snap_tolerance': snap_tol,
        'goal_tolerance': goal_tol,
    }


def _fail_from_sim(sim: dict) -> dict:
    return _route_score_record(
        success=False,
        validity=False,
        parse_ok=sim.get('parse_ok', False),
        end_node=sim.get('end_node'),
        actions=sim.get('actions') or [],
        snapped_nodes=sim.get('snapped') or [],
        simulated_length_m=float(sim.get('simulated_length_m') or 0.0),
    )


def score_route_action_sequence(
    graph: nx.Graph,
    source_node: str,
    goal_node: str,
    raw_actions,
    traversed_edges: Iterable[tuple[str, str]],
    *,
    start_heading_deg: float = 0.0,
    start_pos=None,
    move_m: Optional[float] = None,
    rotation_deg: Optional[float] = None,
    snap_tolerance: Optional[float] = None,
    goal_tolerance: Optional[float] = None,
) -> dict:
    """[CODE] route_knowledge scorer: metric simulation, not node hopping.

    Returns ``success``, ``validity``, and two SPL fields as separate values.

    * Simulate with ``agent_move_m`` / ``agent_rotation_deg`` (continuous xz).
    * Snap each pose; reject at the first pose that does not snap.
    * ``success`` / ``success_raw``: final snap within ``goal_tolerance``.
    * ``validity``: every snapped edge is in the exported-only subset of
      ``traversed_edges``. Always logged (``illegal_edges``, ``outcome``).
    * ``efficiency``: SPL given success: ``S * ℓ / max(p, ℓ)`` on the filtered
      traversed subgraph (0 on failure, capped at 1).
    * ``route_efficiency``: same SPL given success **and** validity (construct
      score). ``route_success`` is the valid-success bit.
    """
    sim = simulate_nav_action_sequence(
        graph,
        source_node,
        raw_actions,
        start_heading_deg=start_heading_deg,
        start_pos=start_pos,
        move_m=move_m,
        rotation_deg=rotation_deg,
        snap_tolerance=snap_tolerance,
        goal_tolerance=goal_tolerance,
    )
    if not sim['ok']:
        return _fail_from_sim(sim)
    if goal_node not in graph:
        return _fail_from_sim({**sim, 'ok': False})

    snapped = sim['snapped']
    sim_len = float(sim['simulated_length_m'])
    exported = filter_traversed_to_exported(graph, traversed_edges)
    crossed = _crossed_undirected(snapped)
    illegal = [edge for edge in crossed if edge not in exported]
    validity = not illegal
    success = _goal_reached(graph, snapped[-1], goal_node, sim['goal_tolerance'])

    opt_traversed = _shortest_on_edges(graph, exported, source_node, goal_node)
    try:
        opt_full = float(
            nx.shortest_path_length(graph, source_node, goal_node, weight='weight')
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        opt_full = None
    opt_nav = opt_traversed if opt_traversed is not None else opt_full
    efficiency = _spl_ratio(opt_nav, sim_len, success=success)
    route_efficiency = _spl_ratio(
        opt_traversed, sim_len, success=bool(success and validity)
    )
    return _route_score_record(
        success=success,
        validity=validity,
        parse_ok=True,
        end_node=snapped[-1],
        actions=sim['actions'],
        snapped_nodes=snapped,
        simulated_length_m=sim_len,
        crossed_edges=crossed,
        illegal_edges=illegal,
        shortest_path_m=opt_traversed,
        shortest_path_full_m=opt_full,
        efficiency=efficiency,
        route_efficiency=route_efficiency,
    )


def score_survey_action_sequence(
    graph: nx.Graph,
    source_node: str,
    goal_node: str,
    raw_actions,
    viewed_edges: Iterable[tuple[str, str]],
    traversed_edges: Iterable[tuple[str, str]],
    *,
    start_heading_deg: float = 0.0,
    start_pos=None,
    move_m: Optional[float] = None,
    rotation_deg: Optional[float] = None,
    snap_tolerance: Optional[float] = None,
    goal_tolerance: Optional[float] = None,
) -> dict:
    """[CODE] survey scorer: same metric simulator as route_knowledge.

    * ``success``: final snap within goal tolerance.
    * ``validity``: every crossed edge is in exported ``viewed_edges``, and at
      least one crossed edge is **not** in exported ``traversed_edges``.
    * ``efficiency``: SPL given success against the shortest path in the
      viewed_edges subgraph (this construct's optimal, not traversed-only).
    """
    sim = simulate_nav_action_sequence(
        graph,
        source_node,
        raw_actions,
        start_heading_deg=start_heading_deg,
        start_pos=start_pos,
        move_m=move_m,
        rotation_deg=rotation_deg,
        snap_tolerance=snap_tolerance,
        goal_tolerance=goal_tolerance,
    )
    if not sim['ok']:
        return _fail_from_sim(sim)
    if goal_node not in graph:
        return _fail_from_sim({**sim, 'ok': False})

    snapped = sim['snapped']
    sim_len = float(sim['simulated_length_m'])
    viewed = filter_traversed_to_exported(graph, viewed_edges)
    traversed = filter_traversed_to_exported(graph, traversed_edges)
    crossed = _crossed_undirected(snapped)
    illegal = [edge for edge in crossed if edge not in viewed]
    novel = [edge for edge in crossed if edge not in traversed]
    validity = (not illegal) and bool(novel)
    success = _goal_reached(graph, snapped[-1], goal_node, sim['goal_tolerance'])
    opt_viewed = _shortest_on_edges(graph, viewed, source_node, goal_node)
    efficiency = _spl_ratio(opt_viewed, sim_len, success=success)
    return _route_score_record(
        success=success,
        validity=validity,
        parse_ok=True,
        end_node=snapped[-1],
        actions=sim['actions'],
        snapped_nodes=snapped,
        simulated_length_m=sim_len,
        crossed_edges=crossed,
        illegal_edges=illegal,
        novel_edges=novel,
        shortest_path_m=opt_viewed,
        efficiency=efficiency,
        route_efficiency=0.0,
    )


def action_sequence_reaches_goal(
    graph: nx.Graph,
    source_node: str,
    goal_node: str,
    raw_actions,
    *,
    start_heading_deg: float = 0.0,
    rotation_deg: Optional[float] = None,
    traversed_edges: Optional[Iterable[tuple[str, str]]] = None,
    **kwargs,
) -> dict:
    """Compatibility wrapper around ``score_route_action_sequence``."""
    edges = traversed_edges
    if edges is None:
        edges = list(graph.edges())
        edges = [(a, b) if a <= b else (b, a) for a, b in edges]
    scored = score_route_action_sequence(
        graph,
        source_node,
        goal_node,
        raw_actions,
        edges,
        start_heading_deg=start_heading_deg,
        rotation_deg=rotation_deg,
        **kwargs,
    )
    scored['reached_goal'] = scored['success']
    return scored


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
    """Mechanical turn-sequence perturbations.

    Taxonomy names: reversed_sequence, swapped_two_turns,
    plausible_but_unwalked_route. Older aliases are accepted.

    Not used by the class-4 pipeline: route_knowledge is a free action sequence,
    not MCQ, and its ``distractor_pattern`` is empty. Kept for turn-level
    analysis only.
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

    Kept for offline graph edits / experiments. Survey generation no longer uses
    conditional_detour (doors are static in current data).
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
    (ahead / left / right / behind) via equal-wedge ego labels. Falls back
    to world +Z frame if goal pose is missing.
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
            AHEAD_HALF_WIDTH_FULL,
            angle_to_ego_label,
            bearing_deg_xz,
            xyz_as_dict,
        )

        a = xyz_as_dict(source_pos)
        b = xyz_as_dict(goal_pos)
        if a and b:
            heading = bearing_deg_xz(
                float(b['x']) - float(a['x']), float(b['z']) - float(a['z'])
            )
            hop = bearing_deg_xz(p1[0] - float(a['x']), p1[2] - float(a['z']))
            if heading is not None and hop is not None:
                rel = (hop - heading) % 360.0
                return angle_to_ego_label(rel, ahead_half_width=AHEAD_HALF_WIDTH_FULL)
    # World-frame fallback
    dx, dz = p1[0] - p0[0], p1[2] - p0[2]
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return None
    if abs(dz) >= abs(dx):
        return 'ahead of you' if dz > 0 else 'behind you'
    return 'to your right' if dx > 0 else 'to your left'
