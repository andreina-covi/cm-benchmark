"""Slide overlay copy: in-image SOURCE/GOAL markers, side-panel action glyphs."""

from pathlib import Path

import pytest

from cm_benchmark.generation.planner import _object_mark_at
from cm_benchmark.generation.slide_copy import (
    _point_xy,
    action_direction,
    action_glyphs,
    class4_pair_length,
    class4_slide_panel,
    endpoint_glyph,
    overlay_class4_frames,
)

try:
    from PIL import Image
except ImportError:
    Image = None


def test_action_glyphs_use_egocentric_arrows():
    assert action_direction('move_ahead') == 'up'
    assert action_direction('move_back') == 'down'
    assert action_direction('rotate_left') == 'left'
    assert action_direction('rotate_right') == 'right'
    assert action_glyphs(['move_ahead', 'rotate_right', 'move_back']) == ['↑', '→', '↓']


def test_class4_panel_hides_node_ids():
    panel = class4_slide_panel(
        'route_knowledge',
        {
            'source': 'Chair',
            'goal': 'Apple',
            'path_nodes': ['n12', 'n99'],
            'action_sequence': ['rotate_left', 'move_ahead'],
        },
    )
    assert 'n12' not in panel['path']
    assert '↑' in panel['path'] and '←' in panel['path']
    assert 'bbox center' in panel['actions']


def test_endpoint_glyph_uses_one_to_three_letters():
    assert endpoint_glyph('Chair') == 'Cha'
    assert endpoint_glyph('Table') == 'Tab'
    assert endpoint_glyph('Dining Table') == 'DT'
    assert endpoint_glyph('TV') == 'TV'
    assert endpoint_glyph('Chair close to the Bread') == 'CB'
    assert endpoint_glyph('', fallback='S') == 'S'


def test_point_xy_is_bbox_center_scaled_to_this_image():
    mark = {'bbox': [20, 20, 80, 90], 'frame_wh': [240, 160]}
    assert _point_xy(mark, (240, 160)) == (50, 55)
    assert _point_xy(mark, (480, 320)) == (100, 110)
    # Unity-pivot projection is not the marker; no bbox → no point.
    lp_only = {'local_point': [25.0, 50.0], 'frame_wh': [240, 160]}
    assert _point_xy(lp_only, (240, 160)) is None
    named = {'bbox': {'cmin': 10, 'rmin': 20, 'cmax': 40, 'rmax': 80}}
    assert _point_xy(named, (100, 100)) == (25, 50)


def test_class4_pair_length_ranks_longer_walks_first():
    short = {
        'answer': 'move_ahead',
        'context': {
            'path_nodes': ['n0', 'n1', 'n2'],
            'hop_count': 2,
            'action_sequence': ['move_ahead'],
        },
    }
    long = {
        'answer': 'move_ahead → rotate_left → move_ahead → rotate_right → move_ahead',
        'context': {
            'path_nodes': ['a', 'b', 'c', 'd', 'e', 'f'],
            'hop_count': 5,
            'action_sequence': [
                'move_ahead',
                'rotate_left',
                'move_ahead',
                'rotate_right',
                'move_ahead',
            ],
        },
    }
    assert class4_pair_length(long) > class4_pair_length(short)
    ranked = sorted([short, long], key=class4_pair_length, reverse=True)
    assert ranked[0] is long


def test_object_mark_at_reads_bbox_from_episode_gt():
    episode = {
        'episode_meta': {'camera': {'width': 396, 'height': 224}},
        'steps': [
            {
                'step': 4,
                'image_path': '/ep/images/img_4.png',
                'visible_objects': {
                    'Chair|1': {
                        'bbox': [10, 20, 40, 80],
                        'local_point': [25, 50],
                    },
                    'Chair|2': {
                        'bbox': [200, 20, 240, 80],
                    },
                },
            }
        ],
    }
    mark = _object_mark_at(episode, 4, 'Chair|1')
    assert mark['bbox'] == [10.0, 20.0, 40.0, 80.0]
    assert mark['local_point'] == [25.0, 50.0]
    assert mark['frame_wh'] == [396, 224]
    assert mark['image_path'] == '/ep/images/img_4.png'
    assert _object_mark_at(episode, 4, 'Table|1') is None
    # Duplicate category: do not pick a sibling Chair.
    assert _object_mark_at(episode, 4, 'Chair') is None


@pytest.mark.skipif(Image is None, reason='Pillow not installed in this venv')
def test_overlay_draws_only_the_bbox_center_markers(tmp_path: Path):
    src = tmp_path / 'source.png'
    goal = tmp_path / 'goal.png'
    Image.new('RGB', (240, 160), (90, 90, 90)).save(src)
    Image.new('RGB', (240, 160), (70, 70, 70)).save(goal)
    out_dir = tmp_path / 'over'
    paths = overlay_class4_frames(
        [str(src), str(goal)],
        roles=['source · Chair', 'goal · Table'],
        source_name='Chair',
        goal_name='Table',
        source_mark={'bbox': [20, 20, 80, 90]},
        goal_mark={'bbox': [140, 30, 210, 110]},
        out_dir=out_dir,
    )
    source = Image.open(paths[0])
    goal_im = Image.open(paths[1])
    assert source.size == (240, 160)
    assert goal_im.size == (240, 160)
    # Letter marker sits on the bbox center of this photo.
    assert source.getpixel((50, 55)) != (90, 90, 90)
    assert goal_im.getpixel((175, 70)) != (70, 70, 70)
    # Everything away from that marker is untouched: no floor arrows.
    far = [
        source.getpixel((x, y))
        for x in range(10, 240, 10)
        for y in range(10, 160, 10)
        if (x - 50) ** 2 + (y - 55) ** 2 > 40 ** 2
    ]
    assert all(px == (90, 90, 90) for px in far)


@pytest.mark.skipif(Image is None, reason='Pillow not installed in this venv')
def test_overlay_skips_bbox_from_a_different_still(tmp_path: Path):
    src = tmp_path / 'img_4.png'
    other = tmp_path / 'img_12.png'
    Image.new('RGB', (240, 160), (90, 90, 90)).save(src)
    out_dir = tmp_path / 'over'
    paths = overlay_class4_frames(
        [str(src)],
        roles=['source · Chair'],
        source_name='Chair',
        source_mark={
            'bbox': [20, 20, 80, 90],
            'image_path': str(other),
            'frame_wh': [240, 160],
        },
        out_dir=out_dir,
    )
    painted = Image.open(paths[0])
    assert painted.size == (240, 160)
    assert painted.getpixel((50, 55)) == (90, 90, 90)
