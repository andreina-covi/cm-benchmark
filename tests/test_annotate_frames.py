"""Smoke tests for frame annotation overlays."""

from pathlib import Path

from PIL import Image

from cm_benchmark.utils.annotate_frames import annotate_episode, annotate_step_image


def test_annotate_step_uses_point_and_legend(tmp_path: Path):
    img_path = tmp_path / 'frame.png'
    Image.new('RGB', (200, 120), color=(40, 40, 40)).save(img_path)

    step = {
        'step': 0,
        'image_path': str(img_path),
        'visible_objects': {
            'Cup|1': {
                'category': 'Cup',
                'bbox': (20, 30, 80, 90),
                'local_point': (50, 60),
                'local_position': (0.1, 0.0, 1.2),
            }
        },
        'edges_egocentric': [
            {
                'source': 'agent',
                'target': 'Cup|1',
                'distance_label': 'nearby',
                'angle_relation': ['left', 'level', 'front'],
            }
        ],
    }

    out = annotate_step_image(step, show_relations=True)
    # Canvas is wider than the frame because of the legend panel
    assert out.size[0] > 200
    assert out.size[1] >= 120
    # Point at bbox center (50, 60) should not be the solid background
    assert out.getpixel((50, 60)) != (40, 40, 40)
    # Legend lives to the right of the frame
    assert out.getpixel((210, 10)) == (20, 20, 20) or out.getpixel((210, 10)) != (40, 40, 40)


def test_annotate_episode_writes_selected_steps(tmp_path: Path):
    img_path = tmp_path / 'frame.png'
    Image.new('RGB', (64, 64), color=(10, 10, 10)).save(img_path)
    episode = {
        'steps': [
            {
                'step': 0,
                'image_path': str(img_path),
                'visible_objects': {
                    'A|1': {'category': 'A', 'bbox': (5, 5, 20, 20), 'local_point': (10, 10)}
                },
                'edges_egocentric': [],
            },
            {
                'step': 1,
                'image_path': str(img_path),
                'visible_objects': {
                    'B|1': {'category': 'B', 'bbox': (10, 10, 30, 30), 'local_point': (15, 15)}
                },
                'edges_egocentric': [],
            },
        ]
    }
    out_dir = tmp_path / 'out'
    written = annotate_episode(episode, out_dir, steps=[1])
    assert len(written) == 1
    assert written[0].name == 'annotated_step_0001.png'
    assert written[0].is_file()
