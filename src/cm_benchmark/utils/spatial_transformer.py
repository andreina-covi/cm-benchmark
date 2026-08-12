import ast

import numpy as np


def calculate_focal_length(val, fov):
    return val / (2 * np.tan(np.deg2rad(fov / 2)))


def get_other_fov(val1, val2, fov):
    fov_other = 2 * np.arctan((val1 / val2) * np.tan(np.deg2rad(fov / 2)))
    return np.degrees(fov_other)


def get_focal_length(w, h, fov_v):
    fy = calculate_focal_length(h, fov_v)
    fov_h = get_other_fov(w, h, fov_v)
    fx = calculate_focal_length(w, fov_h)
    return fx, fy, fov_h


def transform_text2list(text):
    return list(ast.literal_eval(text))


def calculate_angle(coord1, coord2):
    return np.degrees(np.arctan2(coord1, coord2))


def world_to_local(camera_pos, agent_rot_deg, object_pos):
    # agent_rot_deg = (pitch_x, yaw_y, roll_z) in degrees
    pitch, yaw, _roll = np.deg2rad(agent_rot_deg)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ]
    )
    R_wc = (Ry @ Rx).T  # world -> camera
    Pw = np.asarray(object_pos).reshape(3, 1)
    C = np.asarray(camera_pos).reshape(3, 1)
    return (R_wc @ (Pw - C)).flatten()


def projection_with_local_vector(local_xyz, c_point, foc_l, hyperparams):
    xl, yl, zl = local_xyz
    if zl <= hyperparams['ez']:
        raise ValueError(f'Z is too small: {zl}')
    u = foc_l[0] * (xl / zl) + c_point[0]
    v = c_point[1] - foc_l[1] * (yl / zl)
    return float(u), float(v)


def transform_3d_to_2d(obj1_pos, obj1_rot, obj2_pos, c_point, foc_l, hyperparams):
    x_l, y_l, z_l = world_to_local(obj1_pos, obj1_rot, obj2_pos)
    alpha = calculate_angle(x_l, z_l)
    betha = calculate_angle(y_l, z_l)
    try:
        u_l, v_l = projection_with_local_vector((x_l, y_l, z_l), c_point, foc_l, hyperparams)
    except ValueError as e:
        print(f'Error in projection: {e}')
        u_l, v_l = None, None
    return (x_l, y_l, z_l), (u_l, v_l), alpha, betha


def transform_3d_to_2d_with_fov(obj1_pos, obj1_rot, obj2_pos, hyperparams):
    w, h = hyperparams['w'], hyperparams['h']
    fov_v = hyperparams['fov_v']
    fx, fy, fov_h = get_focal_length(w, h, fov_v)
    hyperparams['fov_h'] = fov_h
    hyperparams['fx'] = fx
    hyperparams['fy'] = fy
    c_point = (w // 2, h // 2)
    return transform_3d_to_2d(obj1_pos, obj1_rot, obj2_pos, c_point, (fx, fy), hyperparams)
