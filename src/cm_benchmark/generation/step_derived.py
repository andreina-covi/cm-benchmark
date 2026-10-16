"""On-demand memory, edges, and hydrated visible objects for slim episode GT.

Stored steps keep pose, action, action_success, and per-frame visibility.
Category, pose, and size come from ``object_catalog`` (or the sparse
``object_state_track`` when the object moved). ``non_visible_objects`` and
inferred / egocentric edges are rebuilt with the same rules as
``NavSequenceGenerator`` and cached on the step view.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np

from cm_benchmark.utils.spatial_relations import get_direction_angle, get_distance_text
from cm_benchmark.utils.spatial_transformer import transform_3d_to_2d_with_fov

FRAME_KEYS = (
    'bbox',
    'bbox_area',
    'min_side',
    'occupancy_ratio',
    'visible_pixels',
    'obj_distance',
    'ang_width_deg',
)

_STEP_KEEP = (
    'step',
    'image_path',
    'action',
    'degrees',
    'action_success',
    'agent',
    'current_room',
    'current_room_type',
)

_DERIVED_KEYS = (
    'non_visible_objects',
    'edges_egocentric',
    'edges_allocentric',
    'edges_object_frame',
    'edges_inferred',
)


class _Derived:
    """Per-episode cache shared by that episode's step views."""

    def __init__(self) -> None:
        self.non_visible: Optional[dict[int, dict]] = None
        self.ego: dict[int, list] = {}
        self.allo: dict[int, list] = {}
        self.inferred: dict[int, list] = {}
        self.visible: dict[int, dict] = {}


def _is_slim_object(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    return (
        'category' not in obj
        and 'position' not in obj
        and 'local_position' not in obj
    )


def _is_slim_visible(objs: Any) -> bool:
    if not isinstance(objs, dict) or not objs:
        return False
    sample = next(iter(objs.values()))
    return _is_slim_object(sample)


def episode_is_slim(episode: Optional[dict]) -> bool:
    """True when steps omit stored memory and edges."""
    if not isinstance(episode, dict):
        return False
    if episode.get('gt_slim'):
        return True
    steps = episode.get('steps') or []
    if not steps or not isinstance(steps[0], dict):
        return False
    step = steps[0]
    return 'non_visible_objects' not in step and 'edges_egocentric' not in step


def _xyz_list(value) -> Optional[list[float]]:
    if value is None:
        return None
    try:
        xyz = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError, IndexError):
        return None
    if any(v != v or v in (float('inf'), float('-inf')) for v in xyz):
        return None
    return xyz


def edge_hyperparams(episode: dict) -> dict[str, Any]:
    """Camera and distance bins stored on the episode, else generation.yaml."""
    from cm_benchmark.generation.config import get_generation_config

    egt = get_generation_config().episode_gt
    thr = episode.get('thresholds') or {}
    dist = thr.get('distance_label') or {}
    rel = thr.get('relation') or {}
    cam = episode.get('camera') or {}
    meta_cam = ((episode.get('episode_meta') or {}).get('camera') or {})
    within = dist.get('within_reach') or [0.0, egt.min_distance_m]
    nearby = dist.get('nearby') or [within[1], egt.med_distance_m]
    far = dist.get('far') or [nearby[1], egt.max_distance_m]
    width = cam.get('width', meta_cam.get('width', egt.frame_w))
    height = cam.get('height', meta_cam.get('height', egt.frame_h))
    fov = cam.get('fov_vertical_deg', meta_cam.get('fov_vertical_deg', egt.fov_v_deg))
    return {
        'w': int(width),
        'h': int(height),
        'fov_v': float(fov),
        'k_neighbors': int(egt.k_neighbors),
        'radius': float(egt.radius_m),
        'ey': float(rel.get('vertical_m', egt.ey_m)),
        'ez': float(egt.ez_m),
        'angle_threshold_xz': rel.get('lateral_deg', egt.angle_threshold_xz_deg),
        'min_distance': float(within[1]),
        'med_distance': float(nearby[1]),
        'max_distance': float(far[1]),
    }


def _catalog(episode: dict, obj_id: str) -> dict:
    cat = (episode.get('object_catalog') or {}).get(obj_id) or {}
    return cat if isinstance(cat, dict) else {}


def _track_state(episode: dict, obj_id: str, step_idx: int):
    from cm_benchmark.generator.ai2thor_nav_generator import state_at_step

    track = (episode.get('object_state_track') or {}).get(obj_id) or {}
    entries = track.get('entries') if isinstance(track, dict) else None
    if not entries:
        return None
    return state_at_step(entries, int(step_idx))


def _displaced_ids(episode: dict) -> set[str]:
    out: set[str] = set()
    for event in episode.get('displacement_events') or []:
        if isinstance(event, dict) and event.get('obj_id'):
            out.add(event['obj_id'])
    return out


def _project(agent_pos, agent_rot, obj_pos, hyper: dict):
    w_to_l, p_l, alpha, betha = transform_3d_to_2d_with_fov(
        agent_pos, agent_rot, obj_pos, hyper
    )
    local = tuple(np.round(np.asarray(w_to_l, dtype=float), 3))
    if p_l[0] is None or p_l[1] is None:
        point = (None, None)
    else:
        point = tuple(np.round(np.asarray(p_l, dtype=float), 3))
    angles = (float(np.round(alpha, 3)), float(np.round(betha, 3)))
    return local, point, angles


def _edges_between(source, neighbors, data, hyper: dict, last_seen: int = -1) -> list[dict]:
    edges = []
    origin = np.array(data[source]['local_position'], dtype=float)
    angle_thr = hyper['angle_threshold_xz']
    vertical_thr = hyper['ey']
    for neighbor in neighbors:
        other = np.array(data[neighbor]['local_position'], dtype=float)
        dist = float(np.linalg.norm(other - origin))
        diff = other - origin
        angle_direction = get_direction_angle(diff, angle_thr, vertical_thr)
        edge = {
            'source': source,
            'target': neighbor,
            'distance_metric': round(dist, 2),
            'distance_label': get_distance_text(
                dist, hyper['min_distance'], hyper['med_distance'], hyper['max_distance']
            ),
            'visible': last_seen < 0,
            'angle_relation': list(angle_direction),
            'inferred': last_seen >= 0,
        }
        if last_seen >= 0:
            edge['last_seen'] = last_seen
        edges.append(edge)
    return edges


def _knn(obj_id: str, data: dict, hyper: dict) -> list[str]:
    origin = np.array(data[obj_id]['local_position'], dtype=float)
    neighbors = []
    radius = hyper['radius']
    for other_id, other_data in data.items():
        if other_id == obj_id:
            continue
        other = np.array(other_data['local_position'], dtype=float)
        alpha = other_data.get('angles', (None, None))[0]
        dist = float(np.linalg.norm(origin - other))
        if dist <= radius and alpha is not None:
            neighbors.append((other_id, dist))
    neighbors.sort(key=lambda item: item[1])
    k = hyper['k_neighbors']
    return [name for name, _dist in neighbors[:k]]


def raw_visible_frames(step: dict) -> dict:
    """Per-frame visibility records, ignoring a question-visibility filter."""
    frames = step.get('visible_frames')
    if isinstance(frames, dict):
        return frames
    objs = step.get('visible_objects') if isinstance(step, dict) else None
    if _is_slim_visible(objs):
        return objs
    return {}


def hydrate_visible_object(
    episode: dict,
    step: dict,
    obj_id: str,
    frame: dict,
    *,
    hyper: Optional[dict] = None,
) -> dict:
    """Merge one per-frame record with catalog / object-state static fields."""
    hyper = hyper or edge_hyperparams(episode)
    step_idx = int(step.get('step') or 0)
    catalog = _catalog(episode, obj_id)
    track = (episode.get('object_state_track') or {}).get(obj_id) or {}
    state = _track_state(episode, obj_id, step_idx)
    if state is not None and _xyz_list(state.get('position')) is not None:
        position = _xyz_list(state.get('position'))
        rotation = _xyz_list(state.get('rotation'))
        category = None
        if isinstance(track, dict):
            category = track.get('category')
        category = category or catalog.get('category')
        use_center = False
    else:
        position = _xyz_list(catalog.get('position'))
        rotation = _xyz_list(catalog.get('rotation'))
        category = catalog.get('category')
        use_center = True
    obj: dict[str, Any] = {}
    if category is not None:
        obj['category'] = category
    if position is not None:
        obj['position'] = position
    if rotation is not None:
        obj['rotation'] = rotation
    obj['facing'] = None
    for key in FRAME_KEYS:
        if isinstance(frame, dict) and key in frame:
            obj[key] = frame.get(key)
    agent = step.get('agent') or {}
    if position is not None and agent.get('position') is not None and agent.get('rotation') is not None:
        local, point, angles = _project(
            agent['position'], agent['rotation'], position, hyper
        )
        obj['local_position'] = local
        obj['local_point'] = point
        obj['angles'] = angles
    if catalog.get('synset'):
        obj['synset'] = catalog['synset']
    size = _xyz_list(catalog.get('size'))
    if size is not None:
        obj['size'] = size
    amin = _xyz_list(catalog.get('aabb_min'))
    amax = _xyz_list(catalog.get('aabb_max'))
    if amin is not None:
        obj['aabb_min'] = amin
    if amax is not None:
        obj['aabb_max'] = amax
    if catalog.get('room_id'):
        obj['room_id'] = catalog['room_id']
    if use_center:
        center = _xyz_list(catalog.get('bbox_center'))
        if center is not None:
            obj['bbox_center'] = center
    if state is not None:
        if 'in_camera_fov' in state:
            obj['in_camera_fov'] = state.get('in_camera_fov')
        if state.get('parent_receptacle') is not None:
            obj['parent_receptacle'] = state.get('parent_receptacle')
    return obj


def hydrated_visible(episode: dict, step: dict) -> dict[str, dict]:
    """Full visible-object map for one step (frame fields + catalog)."""
    hyper = edge_hyperparams(episode)
    frames = raw_visible_frames(step)
    return {
        oid: hydrate_visible_object(episode, step, oid, frame, hyper=hyper)
        for oid, frame in frames.items()
    }


def _memory_tables(episode: dict) -> dict[int, dict]:
    """step index → non-visible memory, same update order as collection."""
    memory: dict[str, dict] = {}
    out: dict[int, dict] = {}
    displaced = _displaced_ids(episode)
    for step in episode.get('steps') or []:
        if not isinstance(step, dict):
            continue
        step_idx = int(step['step']) if step.get('step') is not None else 0
        if raw_visible_frames(step) or _is_slim_visible(dict.get(step, 'visible_objects')):
            visible = hydrated_visible(episode, step)
        elif not episode_is_slim(episode):
            stored = dict.get(step, 'visible_objects') or {}
            visible = stored if isinstance(stored, dict) else {}
        else:
            visible = {}
        for obj_id, obj_data in visible.items():
            remembered = {
                'category': obj_data.get('category'),
                'position': obj_data.get('position'),
                'last_seen_step': step_idx,
            }
            memory[obj_id] = remembered
        for obj_id in displaced:
            if obj_id not in memory:
                continue
            state = _track_state(episode, obj_id, step_idx)
            if state is None:
                continue
            pos = _xyz_list(state.get('position'))
            if pos is not None:
                memory[obj_id]['position'] = pos
            memory[obj_id]['parent_receptacle'] = state.get('parent_receptacle')
            memory[obj_id]['in_camera_fov'] = state.get('in_camera_fov')
        visible_ids = set(visible)
        out[step_idx] = {
            obj_id: dict(memory_data)
            for obj_id, memory_data in memory.items()
            if obj_id not in visible_ids
        }
    return out


def _ego_and_allo(episode: dict, step: dict, visible: dict) -> tuple[list, list]:
    hyper = edge_hyperparams(episode)
    agent = {'agent': {'local_position': (0.0, 0.0, 0.0)}}
    ids = [oid for oid in visible if 'local_position' in (visible.get(oid) or {})]
    ego = _edges_between('agent', ids, {**visible, **agent}, hyper)
    allo: list[dict] = []
    for obj_id in visible:
        if 'local_position' not in visible[obj_id] or 'angles' not in visible[obj_id]:
            continue
        neighbors = _knn(obj_id, visible, hyper)
        allo.extend(_edges_between(obj_id, neighbors, visible, hyper))
    return ego, allo


def _inferred_edges(episode: dict, step: dict, non_visible: dict) -> list[dict]:
    hyper = edge_hyperparams(episode)
    agent = step.get('agent') or {}
    ag_pos = agent.get('position')
    ag_rot = agent.get('rotation')
    if ag_pos is None or ag_rot is None:
        return []
    edges: list[dict] = []
    for obj_id, remembered in non_visible.items():
        pos = remembered.get('position')
        if pos is None:
            continue
        local, _point, _angles = _project(ag_pos, ag_rot, pos, hyper)
        aux = {
            obj_id: {'local_position': local},
            'agent': {'local_position': (0.0, 0.0, 0.0)},
        }
        last_seen = remembered.get('last_seen_step', -1)
        edges.extend(
            _edges_between('agent', [obj_id], aux, hyper, last_seen=int(last_seen))
        )
    return edges


def slim_visible_map(visible: Optional[dict]) -> dict:
    """Drop static fields from a visible-object map."""
    out = {}
    for obj_id, obj in (visible or {}).items():
        if not isinstance(obj, dict):
            continue
        if _is_slim_object(obj):
            out[obj_id] = {k: obj[k] for k in FRAME_KEYS if k in obj}
            continue
        frame = {k: obj[k] for k in FRAME_KEYS if k in obj}
        out[obj_id] = frame
    return out


def slim_step(step: dict) -> dict:
    """Pose, action, action_success, and per-frame visible objects only."""
    if isinstance(step, StepView):
        return step.slim_dict()
    out = {}
    for key in _STEP_KEEP:
        if key in step:
            out[key] = step[key]
    frames = step.get('visible_frames')
    if not isinstance(frames, dict):
        frames = slim_visible_map(step.get('visible_objects'))
    out['visible_objects'] = frames
    return out


def persistable_episode(episode: dict) -> dict:
    """Plain episode dict for JSON/SQLite: no embedded nav graph, no memory/edges."""
    out = {}
    for key, value in episode.items():
        if key in ('nav_graph', 'steps'):
            continue
        if key.startswith('_'):
            continue
        out[key] = value
    out.pop('nav_graph', None)
    out['gt_slim'] = True
    out['steps'] = [slim_step(step) for step in episode.get('steps') or []]
    return out


class StepView(dict):
    """Dict step that rebuilds memory and edges on read and caches them."""

    def __init__(self, data=None, episode=None, derived: Optional[_Derived] = None):
        super().__init__()
        if data:
            self.update(data)
        self._episode = episode if episode is not None else {}
        self._derived = derived or _Derived()
        self._cache: dict[str, Any] = {}
        stored = dict.get(self, 'visible_objects')
        if _is_slim_visible(stored):
            self['visible_frames'] = stored
            del self['visible_objects']

    def slim_dict(self) -> dict:
        raw = {k: v for k, v in dict.items(self) if k not in _DERIVED_KEYS and k != 'visible_objects'}
        frames = raw.get('visible_frames')
        out = {k: raw[k] for k in _STEP_KEEP if k in raw}
        out['visible_objects'] = frames if isinstance(frames, dict) else slim_visible_map(
            dict.get(self, 'visible_objects')
        )
        return out

    def __deepcopy__(self, memo):
        new = StepView.__new__(StepView)
        memo[id(self)] = new
        dict.__init__(new)
        for key, value in dict.items(self):
            dict.__setitem__(new, copy.deepcopy(key, memo), copy.deepcopy(value, memo))
        new._episode = memo.get(id(self._episode), self._episode)
        side_key = ('step-derived', id(self._episode))
        side = memo.get(side_key)
        if side is None:
            side = _Derived()
            memo[side_key] = side
        new._derived = side
        new._cache = {}
        return new

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        if key == 'visible_objects':
            return self._visible()
        if key in _DERIVED_KEYS and key not in self:
            return self._derived_value(key)
        return dict.__getitem__(self, key)

    def _visible(self) -> dict:
        stored = dict.get(self, 'visible_objects')
        if isinstance(stored, dict) and not _is_slim_visible(stored):
            return stored
        step_idx = int(dict.get(self, 'step') or 0)
        cached = self._derived.visible.get(step_idx)
        if cached is None:
            cached = hydrated_visible(self._episode, self)
            self._derived.visible[step_idx] = cached
        return cached

    def _derived_value(self, key: str):
        step_idx = int(dict.get(self, 'step') or 0)
        if key == 'edges_object_frame':
            return []
        if key == 'non_visible_objects':
            if self._derived.non_visible is None:
                self._derived.non_visible = _memory_tables(self._episode)
            return self._derived.non_visible.get(step_idx, {})
        if key == 'edges_egocentric':
            if step_idx not in self._derived.ego:
                ego, allo = _ego_and_allo(self._episode, self, self._visible_for_edges())
                self._derived.ego[step_idx] = ego
                self._derived.allo[step_idx] = allo
            return self._derived.ego[step_idx]
        if key == 'edges_allocentric':
            if step_idx not in self._derived.allo:
                ego, allo = _ego_and_allo(self._episode, self, self._visible_for_edges())
                self._derived.ego[step_idx] = ego
                self._derived.allo[step_idx] = allo
            return self._derived.allo[step_idx]
        if key == 'edges_inferred':
            if step_idx not in self._derived.inferred:
                self._derived.inferred[step_idx] = _inferred_edges(
                    self._episode, self, self._derived_value('non_visible_objects')
                )
            return self._derived.inferred[step_idx]
        raise KeyError(key)

    def _visible_for_edges(self) -> dict:
        """Raw-FOV objects, not the question-visibility subset."""
        if 'visible_frames' in self or _is_slim_visible(dict.get(self, 'visible_objects')):
            step_idx = int(dict.get(self, 'step') or 0)
            cached = self._derived.visible.get(step_idx)
            if cached is None:
                cached = hydrated_visible(self._episode, self)
                self._derived.visible[step_idx] = cached
            return cached
        stored = dict.get(self, 'visible_objects')
        return stored if isinstance(stored, dict) else {}


def prepare_episode(episode: dict) -> dict:
    """Wrap slim steps so callers can read memory and edges as before."""
    if not isinstance(episode, dict) or not episode_is_slim(episode):
        return episode
    steps = episode.get('steps') or []
    if steps and isinstance(steps[0], StepView):
        return episode
    derived = _Derived()
    episode['steps'] = [
        step if isinstance(step, StepView) else StepView(step, episode, derived)
        for step in steps
    ]
    return episode
