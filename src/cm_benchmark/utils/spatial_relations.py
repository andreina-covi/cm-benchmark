import math
import numpy as np

def get_x_direction(x_pos, thr_x):
    x_dir = "" # center / ignore
    if x_pos > thr_x:
        x_dir = "right"
    elif x_pos < thr_x * -1:
        x_dir = "left"
    return x_dir

def get_y_direction(y_pos, thr_y):
    y_dir = "" # center / ignore
    if y_pos > thr_y:
        y_dir = "above"
    elif y_pos < thr_y * -1:
        y_dir = "below"
    return y_dir

def get_z_direction(z_pos, thr_z):
    z_dir = "" # same depth / ignore
    if z_pos > thr_z:
        z_dir = "front"
    elif z_pos < thr_z * -1:
        z_dir = "behind"
    return z_dir

def get_x_direction_angle(angle_xz, angle_threshold_xz):
    abs_angle = abs(angle_xz)
    if abs_angle < angle_threshold_xz or abs_angle > (180 - angle_threshold_xz):
        # nearly straight ahead or straight behind → no left/right
        x_relation = ""
    elif angle_xz > 0:
        x_relation = "right"
    else:
        x_relation = "left"
    return x_relation

def get_z_direction_angle(angle_xz, angle_threshold_xz):
    abs_angle = abs(angle_xz)
    if abs_angle < (90 - angle_threshold_xz):
        # nearly straight ahead or straight behind → no above/below
        z_relation = "front"
    elif abs_angle > (90 + angle_threshold_xz):
        z_relation = "behind"
    else:
        z_relation = ""
    return z_relation

def get_distance_text(number, min_distance, med_distance, max_distance):
    # this method can be improved by using the actual distribution of distances in the dataset 
    # to define the thresholds for near, medium, and far. For now, we are using fixed thresholds for simplicity.

    if number <= min_distance:
        text = "within reach"
    elif number <= med_distance:
        text = "nearby"
    elif number <= max_distance:
        text = "far"
    else:
        text = "beyond"
    return text

def get_direction_angle(diff, angle_threshold_xz, vertical_threshold):
    x, y, z = diff
    angle_xz = math.atan2(x, z) * 180 / math.pi
    # angle_yz = math.atan2(y, z) * 180 / math.pi
    x_dir = get_x_direction_angle(angle_xz, angle_threshold_xz)
    z_dir = get_z_direction_angle(angle_xz, angle_threshold_xz)
    y_dir = get_y_direction(y, vertical_threshold)
    return (x_dir, y_dir, z_dir)