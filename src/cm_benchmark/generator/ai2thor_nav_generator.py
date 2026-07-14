import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from src.cm_benchmark.generator.episode_paths import resolve_episode_paths
from src.cm_benchmark.generator.nav_sequence_generator import NavSequenceGenerator, parse_args
from src.cm_benchmark.utils.spatial_transformer import transform_text2list, transform_3d_to_2d_with_fov


def _is_valid_obj_id(obj):
    if obj is None:
        return False
    if isinstance(obj, float) and math.isnan(obj):
        return False
    if isinstance(obj, str) and obj.strip().lower() in ('', 'nan'):
        return False
    return True


def _as_bool(val):
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return False
    if isinstance(val, (int, np.integer)):
        return bool(val)
    s = str(val).strip().lower()
    return s in ('true', '1', 'yes')


def _pose_changed(a, b, eps=1e-4):
    if a is None or b is None:
        return a != b
    return any(abs(float(x) - float(y)) > eps for x, y in zip(a, b))


def sparsify_state_entries(entries, eps=1e-4):
    """Keep first observation + steps where pose/visibility/receptacle change."""
    if not entries:
        return []
    sparse = [entries[0]]
    prev = entries[0]
    for entry in entries[1:]:
        changed = (
            _pose_changed(prev['position'], entry['position'], eps)
            or _pose_changed(prev['rotation'], entry['rotation'], eps)
            or prev.get('visible') != entry.get('visible')
            or prev.get('in_camera_fov') != entry.get('in_camera_fov')
            or prev.get('parent_receptacle') != entry.get('parent_receptacle')
            or prev.get('is_inside_receptacle') != entry.get('is_inside_receptacle')
            or prev.get('receptacle_is_open') != entry.get('receptacle_is_open')
        )
        if changed:
            sparse.append(entry)
            prev = entry
    return sparse


def state_at_step(entries, step):
    """
    Resolve object state at query step t from a sparse track.

    Uses the latest entry with entry['step'] <= t (carry-forward).
    Returns None if step is before the first logged observation.
    """
    if not entries:
        return None
    chosen = None
    for entry in entries:
        if entry['step'] <= step:
            chosen = entry
        else:
            break
    return chosen


def sparsify_timestep_series(rows, key_fields, time_field='timestep'):
    """Keep first row + rows where any of key_fields changes (carry-forward semantics)."""
    if not rows:
        return []
    sparse = [rows[0]]
    prev = rows[0]
    for row in rows[1:]:
        if any(prev.get(k) != row.get(k) for k in key_fields):
            sparse.append(row)
            prev = row
    return sparse


def series_at_step(rows, step, time_field='timestep'):
    """Latest row with time_field <= step."""
    if not rows:
        return None
    chosen = None
    for row in rows:
        if row[time_field] <= step:
            chosen = row
        else:
            break
    return chosen


def _file_overrides_from_args(args):
    mapping = {
        'navigation': getattr(args, 'file_navigation', None),
        'objects': getattr(args, 'file_objects', None),
        'object_state': getattr(args, 'file_object_state', None),
        'displacement_events': getattr(args, 'file_displacement_events', None),
    }
    return {k: v for k, v in mapping.items() if v}


class Ai2ThorNavGenerator(NavSequenceGenerator):
    def __init__(
        self,
        path_navigation=None,
        path_objects=None,
        output_path='.',
        output_filename='nav_data.json',
        hyperparams=None,
        csv_path_folder=None,
        scene_id=None,
        file_overrides=None,
    ):
        if hyperparams is None:
            hyperparams = {
                'w': 396,
                'h': 224,
                'fov_v': 59,
                'epsilon': 1 / 3,
                'k_neighbors': 3,
                'radius': 1.5,
                'fraction_threshold': 0.15,
                'ex': 0.1,
                'ey': 0.1,
                'ez': 0.15,
                'angle_threshold_xz': 15,
                'min_distance': 0.5,
                'med_distance': 1.0,
                'max_distance': 1.5,
                'mov_constant': 0.2,
            }

        self.episode_meta = {}
        self.displacement_events = []
        self.world_layout = None
        self.passage_state = []
        self.region_trajectory = []
        # (timestep, obj_id) -> state dict for displaced objects only
        self.state_lookup = {}
        self.displaced_obj_ids = set()
        self.scene_id = scene_id

        if csv_path_folder is not None:
            paths = resolve_episode_paths(
                csv_path_folder, scene_id=scene_id, file_overrides=file_overrides
            )
            path_navigation = str(paths['navigation'])
            path_objects = str(paths['objects'])
            self.scene_id = paths['scene_id']
            self.episode_meta = paths.get('episode_meta_data') or {}
            self._paths = paths
            self._load_optional_logs(paths)
        else:
            if path_navigation is None or path_objects is None:
                raise ValueError('Provide csv_path_folder or both path_navigation and path_objects')
            self._paths = {
                'navigation': Path(path_navigation),
                'objects': Path(path_objects),
                'scene_id': scene_id,
            }

        super().__init__(path_navigation, path_objects, output_path, output_filename, hyperparams)

    def _load_optional_logs(self, paths):
        if 'displacement_events' in paths:
            self.displacement_events = self._read_displacement_events(paths['displacement_events'])
            self.displaced_obj_ids = {
                e['obj_id'] for e in self.displacement_events if e.get('obj_id')
            }

        if 'object_state' in paths and self.displaced_obj_ids:
            self.state_lookup = self._read_object_state(
                paths['object_state'], self.displaced_obj_ids
            )

        if 'world_layout' in paths:
            with open(paths['world_layout']) as f:
                self.world_layout = json.load(f)

        if 'passage_state' in paths:
            self.passage_state = self._read_passage_state(paths['passage_state'])

        if 'region_trajectory' in paths:
            self.region_trajectory = self._read_region_trajectory(paths['region_trajectory'])

    def _read_displacement_events(self, path):
        df = pd.read_csv(path)
        events = []
        for _, row in df.iterrows():
            events.append({
                'event_id': row.get('event_id'),
                'obj_id': row.get('obj-id'),
                'at_timestep': int(row.get('at_timestep')) if pd.notna(row.get('at_timestep')) else None,
                'action': row.get('action'),
                'from_receptacle': row.get('from_receptacle') if pd.notna(row.get('from_receptacle')) else None,
                'to_receptacle': row.get('to_receptacle') if pd.notna(row.get('to_receptacle')) else None,
                'from_position': (
                    float(row.get('from_pos-x')),
                    float(row.get('from_pos-y')),
                    float(row.get('from_pos-z')),
                ),
                'to_position': (
                    float(row.get('to_pos-x')),
                    float(row.get('to_pos-y')),
                    float(row.get('to_pos-z')),
                ),
                'hidden_during': _as_bool(row.get('hidden_during')),
                'visible_just_before': _as_bool(row.get('visible_just_before')),
                'visible_just_after': _as_bool(row.get('visible_just_after')),
                'in_fov_just_before': _as_bool(row.get('in_fov_just_before')),
                'in_fov_just_after': _as_bool(row.get('in_fov_just_after')),
                'moved_via': row.get('moved_via') if pd.notna(row.get('moved_via')) else None,
                'notes': row.get('notes') if pd.notna(row.get('notes')) else None,
            })
        return events

    def _read_object_state(self, path, obj_ids):
        df = pd.read_csv(path)
        df = df[df['obj-id'].isin(obj_ids)]
        lookup = {}
        for _, row in df.iterrows():
            tid = int(row['timestep'])
            oid = row['obj-id']
            lookup[(tid, oid)] = {
                'timestep': tid,
                'obj_id': oid,
                'category': row.get('obj-type'),
                'position': (
                    float(row.get('obj-pos-x')),
                    float(row.get('obj-pos-y')),
                    float(row.get('obj-pos-z')),
                ),
                'rotation': (
                    float(row.get('obj-rot-x')),
                    float(row.get('obj-rot-y')),
                    float(row.get('obj-rot-z')),
                ),
                'visible': _as_bool(row.get('visible')),
                'in_camera_fov': _as_bool(row.get('in_camera_fov')),
                'parent_receptacle': (
                    row.get('parent_receptacle')
                    if pd.notna(row.get('parent_receptacle'))
                    else None
                ),
                'is_inside_receptacle': _as_bool(row.get('is_inside_receptacle')),
                'receptacle_is_open': (
                    _as_bool(row.get('receptacle_is_open'))
                    if pd.notna(row.get('receptacle_is_open'))
                    else None
                ),
                'distance_from_agent': (
                    float(row.get('distance_from_agent'))
                    if pd.notna(row.get('distance_from_agent'))
                    else None
                ),
            }
        return lookup

    def _read_passage_state(self, path):
        df = pd.read_csv(path)
        rows = []
        for _, row in df.iterrows():
            rows.append({
                'timestep': int(row['timestep']),
                'passage_id': row.get('passage_id'),
                'obj_id': row.get('obj-id'),
                'is_open': _as_bool(row.get('is_open')),
                'is_locked': (
                    _as_bool(row.get('is_locked')) if pd.notna(row.get('is_locked')) else None
                ),
                'from_region': row.get('from_region') if pd.notna(row.get('from_region')) else None,
                'to_region': row.get('to_region') if pd.notna(row.get('to_region')) else None,
            })
        # Sparse per passage_id: only when open/locked/regions change
        by_passage = {}
        for row in rows:
            by_passage.setdefault(row['passage_id'], []).append(row)
        sparse = []
        for _pid, group in by_passage.items():
            sparse.extend(
                sparsify_timestep_series(
                    group, key_fields=('is_open', 'is_locked', 'from_region', 'to_region')
                )
            )
        sparse.sort(key=lambda r: (r['timestep'], str(r.get('passage_id'))))
        return sparse

    def _read_region_trajectory(self, path):
        df = pd.read_csv(path)
        rows = []
        for _, row in df.iterrows():
            rows.append({
                'timestep': int(row['timestep']),
                'region_id': row.get('region_id'),
                'region_type': row.get('region_type') if pd.notna(row.get('region_type')) else None,
            })
        return sparsify_timestep_series(rows, key_fields=('region_id', 'region_type'))

    def get_records_navigation(self):
        df = pd.read_csv(self.path_navigation)
        dict_navigation = {}
        for _, row in df.iterrows():
            timestep = row.get('timestep')
            obj = row.get('obj-id')
            if timestep not in dict_navigation:
                dict_navigation[timestep] = {
                    'action': row.get('ag-action'),
                    'degrees': row.get('degrees'),
                    'action_success': (
                        _as_bool(row.get('action_success'))
                        if 'action_success' in row.index and pd.notna(row.get('action_success'))
                        else None
                    ),
                    'held_obj_id': (
                        row.get('held_obj-id')
                        if 'held_obj-id' in row.index and pd.notna(row.get('held_obj-id'))
                        else None
                    ),
                    'ag_pos': (
                        row.get('camera-pos-x'),
                        row.get('camera-pos-y'),
                        row.get('camera-pos-z'),
                    ),
                    'ag_rot': (
                        row.get('camera-horizon'),
                        row.get('ag-rot-y'),
                        row.get('ag-rot-z'),
                    ),
                    'path': row.get('path'),
                    'current_room': (
                        row.get('current-room')
                        if 'current-room' in row.index and pd.notna(row.get('current-room'))
                        else None
                    ),
                    'current_room_type': (
                        row.get('current-room-type')
                        if 'current-room-type' in row.index
                        and pd.notna(row.get('current-room-type'))
                        else None
                    ),
                    'objects': [],
                    'bboxes': [],
                }
            if _is_valid_obj_id(obj):
                dict_navigation[timestep]['objects'].append(obj)
                dict_navigation[timestep]['bboxes'].append(
                    (row.get('cmin'), row.get('rmin'), row.get('cmax'), row.get('rmax'))
                )
        return dict_navigation

    def get_records_objects(self):
        df = pd.read_csv(self.path_objects)
        dict_objects = {}
        for _, row in df.iterrows():
            receptacle = row.get('receptacleObjectIds')
            try:
                receptacle_ids = transform_text2list(receptacle) if pd.notna(receptacle) else []
            except (ValueError, SyntaxError):
                receptacle_ids = []

            dict_objects[row.get('obj-id')] = {
                'obj-type': row.get('obj-type'),
                'obj-pos': (
                    row.get('obj-pos-x'),
                    row.get('obj-pos-y'),
                    row.get('obj-pos-z'),
                ),
                'obj-rot': (
                    row.get('obj-rot-x'),
                    row.get('obj-rot-y'),
                    row.get('obj-rot-z'),
                ),
                'receptacleObjectIds': receptacle_ids,
                'bbox-c': (
                    row.get('bBox-center-x'),
                    row.get('bBox-center-y'),
                    row.get('bBox-center-z'),
                ),
                'size': (
                    row.get('size-x'),
                    row.get('size-y'),
                    row.get('size-z'),
                ),
            }
        return dict_objects

    def _pose_for_object(self, obj_id, timestep):
        """Prefer object_state pose for displaced objects; else catalog pose."""
        state = self.state_lookup.get((timestep, obj_id))
        if state is not None:
            return state['position'], state['rotation'], state
        data_object = self.dict_objects.get(obj_id)
        if data_object is None:
            return None, None, None
        return data_object['obj-pos'], data_object['obj-rot'], None

    def get_visible_objects(self, data):
        visible_objs = {}
        ag_pos = data['ag_pos']
        ag_rot = data['ag_rot']
        timestep = data.get('timestep')
        for obj in data['objects']:
            obj_pos, obj_rot, state = self._pose_for_object(obj, timestep)
            if obj_pos is None:
                continue
            category = (
                state['category']
                if state is not None
                else self.dict_objects[obj]['obj-type']
            )
            w_to_l, p_l, alpha, betha = transform_3d_to_2d_with_fov(
                ag_pos, ag_rot, obj_pos, self.hyperparams
            )
            visible_objs[obj] = {
                'category': category,
                'position': obj_pos,
                'rotation': obj_rot,
                'facing': None,
                'local_position': tuple(np.round(w_to_l, 3)),
                'local_point': (
                    tuple(np.round(p_l, 3))
                    if p_l[0] is not None and p_l[1] is not None
                    else (None, None)
                ),
                'angles': (float(np.round(alpha, 3)), float(np.round(betha, 3))),
            }
            if state is not None:
                visible_objs[obj]['in_camera_fov'] = state['in_camera_fov']
                visible_objs[obj]['parent_receptacle'] = state['parent_receptacle']
        return visible_objs

    def update_memory(self, memory, visible_objs, timestep):
        for obj_id, obj_data in visible_objs.items():
            memory[obj_id] = {
                'category': obj_data['category'],
                'position': obj_data['position'],
                'last_seen_step': timestep,
            }
        # Refresh remembered poses for displaced objects using object_state
        for obj_id in self.displaced_obj_ids:
            if obj_id not in memory:
                continue
            state = self.state_lookup.get((timestep, obj_id))
            if state is not None:
                memory[obj_id]['position'] = state['position']
                memory[obj_id]['parent_receptacle'] = state['parent_receptacle']
                memory[obj_id]['in_camera_fov'] = state['in_camera_fov']

    def resolve_remembered_position(self, obj_id, timestep, memory_position):
        state = self.state_lookup.get((timestep, obj_id))
        if state is not None:
            return state['position']
        return memory_position

    def build_object_state_track(self):
        """
        Sparse tracks for displaced objects only.

        Each object keeps the first observation plus later steps where pose,
        visibility, in_camera_fov, or receptacle fields change. At query step t,
        use state_at_step(entries, t) to carry forward the latest prior state.
        """
        if not self.displaced_obj_ids:
            return None
        by_obj = {oid: [] for oid in self.displaced_obj_ids}
        for (tid, oid), state in sorted(self.state_lookup.items()):
            by_obj[oid].append({
                'step': tid,
                'position': list(state['position']),
                'rotation': list(state['rotation']),
                'visible': state['visible'],
                'in_camera_fov': state['in_camera_fov'],
                'parent_receptacle': state['parent_receptacle'],
                'is_inside_receptacle': state['is_inside_receptacle'],
                'receptacle_is_open': state['receptacle_is_open'],
            })
        cleaned = {}
        for oid, entries in by_obj.items():
            sparse = sparsify_state_entries(entries)
            if not sparse:
                continue
            cleaned[oid] = {
                'category': self._category_for(oid),
                'entries': sparse,
            }
        return cleaned or None

    def _category_for(self, obj_id):
        for (_t, oid), state in self.state_lookup.items():
            if oid == obj_id:
                return state['category']
        if obj_id in self.dict_objects:
            return self.dict_objects[obj_id]['obj-type']
        return None

    def enrich_episode_data(self, episode_dict):
        episode_dict['object_state_track'] = self.build_object_state_track()
        episode_dict['displacement_events'] = self.displacement_events
        episode_dict['world_layout'] = self.world_layout
        episode_dict['passage_state'] = self.passage_state
        episode_dict['region_trajectory'] = self.region_trajectory
        episode_dict['episode_meta'] = self.episode_meta
        if self.scene_id:
            episode_dict['scene'] = self.scene_id
        return episode_dict


def main(args):
    overrides = _file_overrides_from_args(args)
    generator = Ai2ThorNavGenerator(
        csv_path_folder=args.csv_path_folder,
        scene_id=args.scene_id,
        file_overrides=overrides or None,
        output_path=args.output_path,
        output_filename=args.output_filename,
        hyperparams=None,
    )
    scene_name = generator.scene_id or 'unknown'
    extra_data = {
        'palette': sns.color_palette('Set2', n_colors=10),
        'scene': scene_name,
        'all_distances': [],
    }
    episode_dict = generator.collect_episode_data(extra_data=extra_data)

    meta = generator.episode_meta or {}
    episode_id = (
        args.episode_id
        or meta.get('episode_id')
        or f"{args.environment}_{scene_name}"
    )
    environment = meta.get('environment') or args.environment
    generator.export_to_db(
        episode_dict,
        db_path=args.db_path,
        episode_id=episode_id,
        environment=environment,
    )
    if args.export_json:
        out_name = args.output_filename
        if out_name == 'nav_data.json':
            out_name = f'nav_data_{scene_name}.json'
        generator.export_to_json(episode_dict, out_name)


if __name__ == '__main__':
    args = parse_args()
    main(args)
