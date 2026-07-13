import math
import numpy as np
import pandas as pd
import seaborn as sns

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


class Ai2ThorNavGenerator(NavSequenceGenerator):
    def __init__(self, path_navigation, path_objects, output_path, output_filename, hyperparams=None):
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
        super().__init__(path_navigation, path_objects, output_path, output_filename, hyperparams)

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

    def get_visible_objects(self, data):
        visible_objs = {}
        ag_pos = data['ag_pos']
        ag_rot = data['ag_rot']
        for obj in data['objects']:
            if obj not in self.dict_objects:
                continue
            data_object = self.dict_objects[obj]
            obj_pos = data_object['obj-pos']
            w_to_l, p_l, alpha, betha = transform_3d_to_2d_with_fov(
                ag_pos, ag_rot, obj_pos, self.hyperparams
            )
            visible_objs[obj] = {
                'category': data_object['obj-type'],
                'position': obj_pos,
                'rotation': data_object['obj-rot'],
                # Placeholder until object facing/heading is available
                'facing': None,
                'local_position': tuple(np.round(w_to_l, 3)),
                'local_point': (
                    tuple(np.round(p_l, 3))
                    if p_l[0] is not None and p_l[1] is not None
                    else (None, None)
                ),
                'angles': (float(np.round(alpha, 3)), float(np.round(betha, 3))),
            }
        return visible_objs

    def update_memory(self, memory, visible_objs, timestep):
        for obj_id, obj_data in visible_objs.items():
            memory[obj_id] = {
                'category': obj_data['category'],
                'position': obj_data['position'],
                'last_seen_step': timestep,
            }


def main(args):
    path_navigation = args.csv_path_navigation
    path_objects = args.csv_path_objects
    output_path = args.output_path
    output_filename = args.output_filename
    generator = Ai2ThorNavGenerator(
        path_navigation, path_objects, output_path, output_filename, hyperparams=None
    )
    scene_name = path_navigation.split('/')[-1]
    if '-' in scene_name:
        scene_name = scene_name.split('-')[1].replace('.csv', '')
    extra_data = {
        'palette': sns.color_palette('Set2', n_colors=10),
        'scene': scene_name,
        'all_distances': [],
    }
    episode_dict = generator.collect_episode_data(extra_data=extra_data)

    episode_id = args.episode_id or f"{args.environment}_{scene_name}"
    generator.export_to_db(
        episode_dict,
        db_path=args.db_path,
        episode_id=episode_id,
        environment=args.environment,
    )
    if args.export_json:
        generator.export_to_json(episode_dict, output_filename)


if __name__ == '__main__':
    args = parse_args()
    main(args)
