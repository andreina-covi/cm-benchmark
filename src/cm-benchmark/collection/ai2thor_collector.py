import os
import datetime

import pandas as pd
import numpy as np
# import cv2
from .base_collector import BaseCollector

class Ai2ThorCollector(BaseCollector):

    def __init__(self, nav_path):
        super().__init__()
        self.dict_agent = {}
        self.data_objects = {'objects': set()}
        self.timestep = 0
        self.scene_name = None
        self.dt = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f")
        self.nav_path = nav_path
        self.image_path = os.path.join(self.nav_path, self.dt, 'images')
        self.dict_colors = {}

        os.makedirs(self.image_path, exist_ok=True)
    
    def get_object_data(self, arr_objects, detections):
        objects = set()
        cond_objs = []
        # colors = controller.last_event.instance_segmentation_colors
        # colors = controller.last_event.metadata['colors']
        # dict_colors = {d['name']: d['color'] for d in colors}
        # print("colors: ", colors)
        # print("objects: ", arr_objects)
        # print("objects", controller.last_event.metadata['objects'])
        # print("metadata", controller.last_event.metadata)
        pos_dict_default = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        for obj_dict in arr_objects:
            if not obj_dict['visible']:
                continue
            if not detections.__contains__(obj_dict['objectId']):
                continue
            bbox = detections.__getitem__(obj_dict['objectId'])
            cond_objs.append([
                # class label
                # obj_dict['objectType'],
                # id
                obj_dict['objectId'],
                # name
                # obj_dict['name'],
                # global position 
                # tuple(self.round_number(obj_dict['position'], 2)),
                # local rotation
                # tuple(self.round_number(obj_dict['rotation'], 2)),
                # euclidean distance from the center-point to the agent
                np.round(obj_dict['distance'], 4),
                # bounding box
                bbox
            ])
            # print("Object: ", obj_dict)
            # bbox_name = 'objectOrientedBoundingBox' if obj_dict['objectOrientedBoundingBox'] != None else 'axisAlignedBoundingBox'
            bbox_name = 'axisAlignedBoundingBox' if obj_dict['axisAlignedBoundingBox'] != None else 'objectOrientedBoundingBox'
            # print('dict_colors', dict_colors)
            # print('color', dict_colors[obj_dict['objectId']])
            objects.add((
                obj_dict['objectType'],
                # self.get_name_object(obj_dict['name']),
                obj_dict['objectId'],
                # color
                tuple(self.dict_colors[obj_dict['objectId']]),
                # obj_dict['name'],
                tuple(self.round_number(obj_dict['position'], 2)),
                tuple(self.round_number(obj_dict['rotation'], 2)),
                # obj_dict['parentReceptacles'],
                tuple(obj_dict['receptacleObjectIds']) if obj_dict['receptacleObjectIds'] != None else (),
                # self.get_min_by_axis(obj_dict[bbox_name]["cornerPoints"])
                tuple(self.round_number(obj_dict[bbox_name].get("center", pos_dict_default), 2)),
                tuple(self.round_number(obj_dict[bbox_name].get("size", pos_dict_default), 2))
            ))
            # print(bbox_name, objects)
            # break
        return cond_objs , objects

    def save_bbox(self, dict_data, bbox):
        assert len(bbox) == 4, "size error of bbox, it must be of size 4"
        dict_data['cmin'].append(bbox[0])
        dict_data['rmin'].append(bbox[1])
        dict_data['cmax'].append(bbox[2])
        dict_data['rmax'].append(bbox[3])

    def add_basic_navigation_data(self, dict_navigation, key):
        dict_navigation['timestep'].append(key[0])
        dict_navigation['ag-action'].append(key[1])
        self.save_data_by_axis(dict_navigation, 'ag-pos', self.dict_agent[key]['position'])
        self.save_data_by_axis(dict_navigation, 'ag-rot', self.dict_agent[key]['rotation'])
        self.save_data_by_axis(dict_navigation, 'camera-pos', self.dict_agent[key]['camera_position'])
        dict_navigation['degrees'].append(self.dict_agent[key]['degrees'])
        dict_navigation['camera-horizon'].append(self.dict_agent[key]['camera_horizon'])
        dict_navigation['path'].append(self.dict_agent[key]['image'])

    def save_data_navigation(self, dict_navigation, key):
        objects_data = self.dict_agent[key]['objects']
        if not objects_data:
            self.add_basic_navigation_data(dict_navigation, key)
            dict_navigation['obj-id'].append(None)
            dict_navigation['obj-distance'].append(None)
            self.save_bbox(dict_navigation, [None, None, None, None])
        else:
            for object_data in objects_data:
                self.add_basic_navigation_data(dict_navigation, key)
                dict_navigation['obj-id'].append(object_data[0])
                dict_navigation['obj-distance'].append(object_data[1])
                self.save_bbox(dict_navigation, object_data[2])
    
    def get_dict_navigation(self):
        dict_navigation = {
            'timestep': [], 'ag-action': [], 'degrees': [], 
            'ag-pos-x': [], 'ag-pos-y': [], 'ag-pos-z': [], 
            'ag-rot-x': [], 'ag-rot-y': [], 'ag-rot-z': [],
            'cmin': [], 'rmin': [], 'cmax': [], 'rmax': [], # for saving bbox data 
            'camera-horizon': [], 'camera-pos-x': [], 'camera-pos-y': [], 'camera-pos-z': [], # additional camera data
            'obj-id': [], 'obj-distance': [], 'path': []
        }
        for key in self.dict_agent:
            self.save_data_navigation(dict_navigation, key)
        return dict_navigation

    def get_dict_objects(self):
        dict_objects = {
            'obj-type': [], 'obj-id': [], 'obj-color': [], 
            'obj-pos-x': [], 'obj-pos-y': [], 'obj-pos-z': [],
            'obj-rot-x': [], 'obj-rot-y': [], 'obj-rot-z': [], #'parentReceptacles': [],
            'receptacleObjectIds': [], 'bBox-center-x': [], 'bBox-center-y': [], 'bBox-center-z': [],
            'size-x': [], 'size-y': [], 'size-z': []
        }
        for t in self.data_objects['objects']:
            dict_objects['obj-type'].append(t[0])
            dict_objects['obj-id'].append(t[1])
            dict_objects['obj-color'].append(t[2])
            self.save_data_by_axis(dict_objects, 'obj-pos', t[3])
            self.save_data_by_axis(dict_objects, 'obj-rot', t[4])
            # dict_objects['parentReceptacles'].append(t[4])
            dict_objects['receptacleObjectIds'].append(t[5])
            self.save_data_by_axis(dict_objects, 'bBox-center', t[6])
            self.save_data_by_axis(dict_objects, 'size', t[7])
        return dict_objects

    # method called by the room visit task after each action to save the data of the agent and the visible objects
    def collect_data(self, event, action_name, v_objects, controller, ag_rot_deg):
        # print("METADATA: ", event.metadata)
        self.scene_name = event.metadata['sceneName'] # ["seed"]
        # print("Scene name: ", self.scene_name)
        if not self.dict_colors:
            self.dict_colors = {d['name']: d['color'] for d in event.metadata['colors']}

        position = self.round_number(event.metadata['agent']['position'], 2)
        rotation = self.round_number(event.metadata['agent']['rotation'], 2)
        camera_position = self.round_number(event.metadata['cameraPosition'], 2)
        camera_horizon = np.round(event.metadata['agent']['cameraHorizon'], 2)
        key = (self.timestep, action_name)
        if key not in self.dict_agent:
            self.dict_agent[key] = {'objects': []}
            self.dict_agent[key]['position'] = position
            self.dict_agent[key]['rotation'] = rotation
            self.dict_agent[key]["degrees"] = ag_rot_deg
            self.dict_agent[key]['camera_horizon'] = camera_horizon
            self.dict_agent[key]['camera_position'] = camera_position
            cond_objs, objects = self.get_object_data(v_objects, controller)
            self.dict_agent[key]['objects'] = cond_objs
            if self.data_objects['objects']:
                self.data_objects['objects'].update(objects)
            else:
                self.data_objects['objects'] = objects
            # save image
            image_name = os.path.join(self.image_path, 'img_' + str(self.timestep) + '.png')
            self.dict_agent[key]['image'] = image_name
            self.save_image(image_name, event)
            # update timestep
            self.timestep += 1

    def save_data(self):
        navigation_path = os.path.join(self.nav_path, self.dt, 'navigation-' + self.scene_name + '.csv')
        objects_path = os.path.join(self.nav_path, self.dt, 'objects-' + self.scene_name + '.csv')
        dict_navigation = self.get_dict_navigation()
        df_navigation = pd.DataFrame(dict_navigation)
        df_navigation.to_csv(navigation_path)
        dict_objects = self.get_dict_objects()
        df_objects = pd.DataFrame(dict_objects)
        df_objects.to_csv(objects_path)
