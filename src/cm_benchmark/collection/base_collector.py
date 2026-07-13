# abstract interface all envs implement
# import os
# import datetime

# import pandas as pd
import numpy as np
import cv2

class BaseCollector:

    def __init__(self):
        pass

    def round_number(self, arr_numbers, n_round):
        rounded_arr = []
        if type(arr_numbers) is dict:
            rounded_arr = [arr_numbers['x'], arr_numbers['y'], arr_numbers['z']]
        else:
            rounded_arr = arr_numbers
        return tuple([np.round(number, n_round) for number in rounded_arr])
    
    def get_min_by_axis(self, bbox):
        array = np.array(bbox)
        assert array.shape == (8, 3)
        x_min = np.min(array[:, 0])
        y_min = np.min(array[:, 1])
        z_min = np.min(array[:, 2])
        return (x_min, y_min, z_min)
        
    def save_data_by_axis(self, dict_data, base_name, array):
        axis_names = ['-x', '-y', '-z']
        for (axis, item) in zip(axis_names, array):
            dict_data[base_name + axis].append(item)

    def save_image(self, image_name, event):
        cv2.imwrite(image_name, event.cv2img) 
    
    def collect_data(self):
        pass

    def save_data(self):
        pass
