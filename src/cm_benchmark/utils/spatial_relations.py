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
    """Angle-plane front/behind (kept for tests / callers). Prefer get_z_direction for depth."""
    abs_angle = abs(angle_xz)
    if abs_angle < (90 - angle_threshold_xz):
        z_relation = "front"
    elif abs_angle > (90 + angle_threshold_xz):
        z_relation = "behind"
    else:
        z_relation = ""
    return z_relation

def get_distance_text(number, min_distance, med_distance, max_distance):
    # Distance buckets are independent of front/behind.
    # “beyond” means farther than max_distance, not “behind the agent”.

    if number <= min_distance:
        text = "within reach"
    elif number <= med_distance:
        text = "nearby"
    elif number <= max_distance:
        text = "far"
    else:
        text = "beyond"
    return text

def get_direction_angle(
    diff,
    angle_threshold_xz,
    vertical_threshold,
    depth_threshold=0.0,
    ahead_half_width: float = 20.0,
):
    """
    Egocentric relation from a local offset (dx, dy, dz).

    Horizontal labels use the same ahead/right/behind/left sectors as
    ``angle_to_ego_label``. Default ``ahead_half_width=20`` matches FOV-
    constrained collection (visible objects). Pass 45 for full-circle use.
    ``angle_threshold_xz`` / ``depth_threshold`` are ignored (API compat).
    """
    del angle_threshold_xz, depth_threshold
    x, y, z = diff
    angle_xz = math.atan2(x, z) * 180 / math.pi
    a = angle_xz % 360.0
    w = float(ahead_half_width)
    y_dir = get_y_direction(y, vertical_threshold)
    if a < w or a >= 360.0 - w:
        return ("", y_dir, "front")
    if a < 180.0 - w:
        return ("right", y_dir, "")
    if a < 180.0 + w:
        return ("", y_dir, "behind")
    return ("left", y_dir, "")
