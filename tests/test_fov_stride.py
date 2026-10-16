"""fov_stride: only stride-aligned steps are shown frames / visibility evidence."""

from __future__ import annotations

from cm_benchmark.generation.constructs import fov_stride_of, is_fov_step
from cm_benchmark.generation.disambiguation import passing_objects_in_frames
from cm_benchmark.generation.planner import _images_between, _object_hidden_through


def _stride_episode():
    vis = {
        'Cup|1': {
            'category': 'Cup',
            'position': [0.5, 0.9, 1.0],
            'bbox_area': 2000,
            'min_side': 40,
            'visible_pixels': 500,
        }
    }
    steps = []
    for i in range(5):
        steps.append(
            {
                'step': i,
                'image_path': f'/img_{i}.png',
                'agent': {'position': [float(i), 1.0, 0.0], 'rotation': [0.0, 0.0, 0.0]},
                'visible_objects': vis if i % 2 == 1 else {},
            }
        )
    return {
        'episode_meta': {'fov_stride': 2, 'config': {'fov_stride': 2}},
        'steps': steps,
        'object_state_track': {
            'Cup|1': {
                'category': 'Cup',
                'entries': [
                    {'step': 0, 'visible': False, 'in_camera_fov': False},
                    {'step': 1, 'visible': True, 'in_camera_fov': True},
                    {'step': 2, 'visible': False, 'in_camera_fov': False},
                    {'step': 3, 'visible': True, 'in_camera_fov': True},
                    {'step': 4, 'visible': False, 'in_camera_fov': False},
                ],
            }
        },
    }


def test_fov_stride_helpers():
    ep = _stride_episode()
    assert fov_stride_of(ep) == 2
    assert is_fov_step(ep, 0)
    assert not is_fov_step(ep, 1)
    assert is_fov_step(ep, 2)
    assert fov_stride_of({'episode_meta': {}}) == 1


def test_shown_frames_skip_non_stride_steps():
    ep = _stride_episode()
    paths = _images_between(ep, 0, 4)
    assert paths == ['/img_0.png', '/img_2.png', '/img_4.png']
    frames = passing_objects_in_frames(ep, [0, 1, 2, 3, 4])
    assert set(frames) == {0, 2, 4}
    assert all('Cup|1' not in vis for vis in frames.values())


def test_hidden_check_ignores_non_stride_visibility():
    """Odd steps have the cup in FOV, but they are not visibility evidence."""
    ep = _stride_episode()
    assert _object_hidden_through(ep, 'Cup|1', 0, 4) is False
    # Drop the track hits on odd steps so only visible_objects remain — those
    # odd rows must not count.
    ep['object_state_track']['Cup|1']['entries'] = [
        {'step': 0, 'visible': False, 'in_camera_fov': False},
        {'step': 2, 'visible': False, 'in_camera_fov': False},
        {'step': 4, 'visible': False, 'in_camera_fov': False},
    ]
    assert _object_hidden_through(ep, 'Cup|1', 0, 4) is True
