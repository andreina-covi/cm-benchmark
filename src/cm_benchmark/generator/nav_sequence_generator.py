import os
import json
import argparse
import numpy as np

from abc import ABC, abstractmethod
from src.cm_benchmark.utils.spatial_transformer import transform_3d_to_2d_with_fov
from src.cm_benchmark.utils.spatial_relations import get_distance_text, get_direction_angle
from src.cm_benchmark.storage.episode_store import EpisodeStore


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class NavSequenceGenerator(ABC):
    def __init__(self, path_navigation, path_objects, output_path, output_filename, hyperparams):
        self.path_navigation = path_navigation
        self.path_objects = path_objects
        self.output_path = output_path
        self.output_filename = output_filename
        self.hyperparams = hyperparams
        self.dict_navigation = self.get_records_navigation()
        self.dict_objects = self.get_records_objects()

    @abstractmethod
    def get_records_navigation(self):
        pass

    @abstractmethod
    def get_records_objects(self):
        pass

    @abstractmethod
    def get_visible_objects(self, data):
        pass

    @abstractmethod
    def update_memory(self, memory, visible_objs, timestep):
        pass

    def collect_episode_data(self, extra_data=None):
        if extra_data is None:
            extra_data = {'scene': 'unknown', 'all_distances': []}
        if 'all_distances' not in extra_data:
            extra_data['all_distances'] = []

        episode_dict = {
            'scene': extra_data['scene'],
            'thresholds': {
                'distance_label': {
                    'within_reach': [0.0, self.hyperparams['min_distance']],
                    'nearby': [self.hyperparams['min_distance'], self.hyperparams['med_distance']],
                    'far': [self.hyperparams['med_distance'], self.hyperparams['max_distance']],
                },
                'relation': {
                    'lateral_deg': self.hyperparams['angle_threshold_xz'],
                    'vertical_m': self.hyperparams['ey'],
                    'depth_deg': self.hyperparams['angle_threshold_xz'],
                },
            },
            'movement_constant': self.hyperparams['mov_constant'],
            # Episode-level fields for Class 2–4 constructs
            'agent_trajectory': [],
            'agent_actions': [],
            'route': {'landmarks': [], 'turns': []},
            # Placeholders until manipulation CSV supports invisible displacement
            'object_state_track': None,
            'displacement_events': [],
            'steps': [],
        }

        seen_objects_memory = {}
        for timestep, data in self.dict_navigation.items():
            step_dict = {
                'step': timestep,
                'image_path': data['path'],
                'action': data['action'],
                'degrees': data['degrees'],
                'agent': {
                    'position': data['ag_pos'],
                    'rotation': data['ag_rot'],
                },
            }

            visible_objs = self.get_visible_objects(data)
            self.update_memory(seen_objects_memory, visible_objs, timestep)
            visible_ids = set(visible_objs)

            # Cumulative memory: all previously seen objects not in the current view
            non_visible_objs = {
                obj_id: memory_data
                for obj_id, memory_data in seen_objects_memory.items()
                if obj_id not in visible_ids
            }

            edges_egocentric, edges_allocentric = self.create_edges_for_visible_objects(
                visible_objs, extra_data=extra_data
            )
            edges_inferred = self.create_edges_for_inferred_objects(
                non_visible_objs, seen_objects_memory, data
            )

            step_dict['visible_objects'] = visible_objs
            step_dict['non_visible_objects'] = non_visible_objs
            step_dict['edges_egocentric'] = edges_egocentric
            step_dict['edges_allocentric'] = edges_allocentric
            # Empty until object facing/heading is available (perspective taking)
            step_dict['edges_object_frame'] = []
            step_dict['edges_inferred'] = edges_inferred

            episode_dict['steps'].append(step_dict)
            episode_dict['agent_trajectory'].append({
                'step': timestep,
                'position': data['ag_pos'],
                'rotation': data['ag_rot'],
                'image_path': data['path'],
            })
            episode_dict['agent_actions'].append({
                'step': timestep,
                'action': data['action'],
                'degrees': data['degrees'],
            })

        episode_dict['route'] = self.build_route(episode_dict['steps'])
        return episode_dict

    def create_edges_for_visible_objects(self, visible_objs, extra_data=None):
        """Split agent→object (egocentric) and object→object (allocentric) edges."""
        edges_egocentric = []
        edges_allocentric = []

        agent_data = {'agent': {'local_position': (0.0, 0.0, 0.0)}}
        visible_with_agent = {**visible_objs, **agent_data}
        edges_egocentric.extend(
            self.edges_btw_neighbors(
                'agent', visible_objs.keys(), visible_with_agent, extra_data=extra_data
            )
        )

        for obj_id in visible_objs:
            neighbors = self.get_knn(obj_id, visible_objs)
            edges_allocentric.extend(
                self.edges_btw_neighbors(
                    obj_id, neighbors, visible_objs, extra_data=extra_data
                )
            )
        return edges_egocentric, edges_allocentric

    def create_edges_for_inferred_objects(self, non_visible_objs, memory, data):
        edges = []
        ag_pos = data['ag_pos']
        ag_rot = data['ag_rot']
        for obj_id, obj_memory in non_visible_objs.items():
            obj_pos = obj_memory['position']
            w_to_l, _, _, _ = transform_3d_to_2d_with_fov(
                ag_pos, ag_rot, obj_pos, self.hyperparams
            )
            aux_dict = {
                obj_id: {'local_position': tuple(np.round(w_to_l, 3))},
                'agent': {'local_position': (0.0, 0.0, 0.0)},
            }
            edges.extend(
                self.edges_btw_neighbors(
                    'agent',
                    [obj_id],
                    aux_dict,
                    last_seen=memory[obj_id]['last_seen_step'],
                )
            )
        return edges

    def edges_btw_neighbors(self, obj_id, neighbors, data, last_seen=-1, extra_data=None):
        edges = []
        local_position_object = np.array(data[obj_id]['local_position'], dtype=float)
        angle_thr = self.hyperparams['angle_threshold_xz']
        vertical_thr = self.hyperparams['ey']

        for neighbor in neighbors:
            local_position_neighbor = np.array(data[neighbor]['local_position'], dtype=float)
            dist = float(np.linalg.norm(local_position_neighbor - local_position_object))
            dist_text = get_distance_text(
                dist,
                self.hyperparams['min_distance'],
                self.hyperparams['med_distance'],
                self.hyperparams['max_distance'],
            )
            diff = local_position_neighbor - local_position_object
            angle_direction = get_direction_angle(diff, angle_thr, vertical_thr)
            edge = {
                'source': obj_id,
                'target': neighbor,
                'distance_metric': round(dist, 2),
                'distance_label': dist_text,
                'visible': last_seen < 0,
                'angle_relation': list(angle_direction),
                'inferred': last_seen >= 0,
            }
            if last_seen >= 0:
                edge['last_seen'] = last_seen
            if extra_data is not None:
                extra_data['all_distances'].append(dist)
            edges.append(edge)
        return edges

    def get_knn(self, obj_id, data):
        obj_pos = np.array(data[obj_id]['local_position'], dtype=float)
        neighbors = []
        k = self.hyperparams['k_neighbors']
        radius = self.hyperparams['radius']

        for other_id, other_data in data.items():
            if other_id == obj_id:
                continue
            other_pos = np.array(other_data['local_position'], dtype=float)
            alpha, _betha = other_data['angles']
            dist = float(np.linalg.norm(obj_pos - other_pos))
            if dist <= radius and alpha is not None:
                neighbors.append((other_id, dist))

        neighbors.sort(key=lambda x: x[1])
        return [n[0] for n in neighbors[:k]]

    def build_route(self, steps):
        """Landmark-ordered route from nearby visible objects + executed turns."""
        landmarks = []
        seen_landmarks = set()
        turns = []

        for step in steps:
            step_id = step['step']
            action = step.get('action')
            if action is not None and str(action).lower() not in ('', 'nan', 'none'):
                turns.append({
                    'step': step_id,
                    'action': action,
                    'degrees': step.get('degrees'),
                })

            for obj_id, obj_data in step.get('visible_objects', {}).items():
                if obj_id in seen_landmarks:
                    continue
                dist = float(np.linalg.norm(np.array(obj_data['local_position'], dtype=float)))
                label = get_distance_text(
                    dist,
                    self.hyperparams['min_distance'],
                    self.hyperparams['med_distance'],
                    self.hyperparams['max_distance'],
                )
                # Landmarks = objects first seen within nearby/reach along the path
                if label in ('within_reach', 'nearby'):
                    landmarks.append({
                        'step': step_id,
                        'object_id': obj_id,
                        'category': obj_data['category'],
                        'distance_label': label,
                    })
                    seen_landmarks.add(obj_id)

        return {'landmarks': landmarks, 'turns': turns}

    def export_to_json(self, data, json_filename="nav_data.json"):
        os.makedirs(self.output_path, exist_ok=True)
        filename = os.path.join(self.output_path, json_filename)
        print(f"Exporting data to {filename}")
        with open(filename, "w") as f:
            json.dump(data, f, indent=4, default=_json_default)

    def export_to_db(
        self,
        data,
        db_path,
        episode_id,
        environment='ai2thor',
        replace=True,
    ):
        """Persist episode GT in SQLite (scalable system of record)."""
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or '.', exist_ok=True)
        with EpisodeStore(db_path) as store:
            store.save_episode(
                data, episode_id=episode_id, environment=environment, replace=replace
            )
        print(f"Saved episode '{episode_id}' to {db_path}")
        return episode_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build navigation episode GT from CSV (DB + optional JSON)"
    )
    parser.add_argument("--csv_path_navigation", type=str, required=True)
    parser.add_argument("--csv_path_objects", type=str, required=True)
    parser.add_argument(
        "--output_path",
        type=str,
        default="src/cm_benchmark/storage/ai2thor/nav_data",
        help="Directory for optional JSON export",
    )
    parser.add_argument("--output_filename", type=str, default="nav_data.json")
    parser.add_argument(
        "--db_path",
        type=str,
        default="src/cm_benchmark/storage/ai2thor/episodes.db",
        help="SQLite DB path (system of record)",
    )
    parser.add_argument(
        "--episode_id",
        type=str,
        default=None,
        help="Episode id in the DB (default: derived from scene + filename)",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="ai2thor",
        help="Environment tag stored with the episode",
    )
    parser.add_argument(
        "--export_json",
        action="store_true",
        help="Also write a JSON artifact for inspection / LLM drafting",
    )
    return parser.parse_args()
