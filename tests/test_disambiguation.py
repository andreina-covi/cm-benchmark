"""Whole-item duplicate detection, proximity, and colour-dot marks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from cm_benchmark.generation.config import get_generation_config, use_generation_config
from cm_benchmark.generation.disambiguation import (
    aabb_gap_xz,
    aabb_minmax_gap_xz,
    active_margin_m,
    draw_marked_frames,
    episode_mark_colours,
    pixel_bbox_gap,
    resolve_named_objects,
)
from cm_benchmark.generation.pipeline import generate_items_for_episode


def _vis(cat, pos, *, bbox=None, area=2000, side=40, pix=500, size=None):
    row = {
        'category': cat,
        'position': pos,
        'bbox_area': area,
        'min_side': side,
        'visible_pixels': pix,
    }
    if bbox is not None:
        row['bbox'] = bbox
    if size is not None:
        row['size'] = size
    return row


def _ego(target):
    return {
        'source': 'agent',
        'target': target,
        'distance_metric': 1.0,
        'distance_label': 'near',
        'visible': True,
        'angle_relation': ['', '', 'front'],
        'inferred': False,
    }


def _step(idx, visible, *, image=None, x=0.0, z=0.0):
    return {
        'step': idx,
        'image_path': image or f'/img_{idx}.png',
        'action': 'MoveAhead',
        'agent': {'position': [x, 1.0, z], 'rotation': [0.0, 0.0, 0.0]},
        'visible_objects': visible,
        'non_visible_objects': {},
        'edges_egocentric': [_ego(oid) for oid in visible],
    }


def test_aabb_gap_overlap_and_separation():
    assert aabb_gap_xz((0, 0), (2, 0, 2), (0.5, 0), (2, 0, 2)) == 0.0
    gap = aabb_gap_xz((0, 0), (2, 0, 2), (4, 0), (2, 0, 2))
    assert gap == pytest.approx(2.0)
    # Missing size → centre distance
    assert aabb_gap_xz((0, 0, 0), None, (3, 0, 4), None) == pytest.approx(5.0)
    assert aabb_minmax_gap_xz((-1, 0, -1), (1, 0, 1), (3, 0, -1), (5, 0, 1)) == pytest.approx(2.0)


def test_vsi_margin_uses_world_layout_area():
    cfg = get_generation_config()
    vsi = replace(cfg.disambiguation, proximity_setting='vsi')
    with use_generation_config(replace(cfg, disambiguation=vsi)):
        small = {
            'world_layout': {'regions': [{'region_id': 'kitchen', 'area': 20.0}]},
            'steps': [{'current_room': 'kitchen'}],
        }
        large = {
            'world_layout': {'regions': [{'region_id': 'kitchen', 'area': 80.0}]},
            'steps': [{'current_room': 'kitchen'}],
        }
        assert active_margin_m(small)[0] == pytest.approx(0.15)
        assert active_margin_m(large)[0] == pytest.approx(0.30)


def test_pixel_bbox_gap():
    assert pixel_bbox_gap((0, 0, 10, 10), (5, 5, 15, 15)) == 0.0
    assert pixel_bbox_gap((0, 0, 10, 10), (20, 0, 30, 10)) == pytest.approx(10.0)


def test_sibling_only_in_non_encode_frame_is_detected():
    """Encode-only unique Chair; sibling appears in a later shown frame."""
    encode = _step(
        0,
        {
            'Chair|1': _vis('Chair', [0.0, 0.0, 1.0]),
            'Bread|1': _vis('Bread', [0.1, 0.0, 1.0]),
        },
    )
    later = _step(
        1,
        {
            'Chair|1': _vis('Chair', [0.0, 0.0, 1.0]),
            'Chair|2': _vis('Chair', [3.0, 0.0, 1.0]),
        },
        x=0.5,
    )
    episode = {'steps': [encode, later]}

    from cm_benchmark.generation.constructs import resolve_referring_disambiguator

    assert resolve_referring_disambiguator(encode, 'Chair|1') == ''

    names = resolve_named_objects(episode, [0, 1], ['Chair|1'])
    assert names is not None
    assert names['Chair|1'].mode in ('proximity', 'mark')
    assert names['Chair|1'].phrase


def test_proximity_requires_unique_co_visible_landmark():
    frames_ok = {
        0: {
            'Chair|1': _vis('Chair', [0.0, 0.0, 1.0]),
            'Chair|2': _vis('Chair', [3.0, 0.0, 1.0]),
            'Bread|1': _vis('Bread', [0.1, 0.0, 1.0]),
        }
    }
    episode = {'steps': [_step(0, frames_ok[0])]}
    names = resolve_named_objects(episode, [0], ['Chair|1'])
    assert names is not None
    assert names['Chair|1'].mode == 'proximity'
    assert names['Chair|1'].phrase == ' close to the Bread'
    assert names['Chair|1'].landmark_id == 'Bread|1'

    # Landmark type duplicated across shown frames → cannot use it
    two_bread = {
        'Chair|1': _vis('Chair', [0.0, 0.0, 1.0]),
        'Chair|2': _vis('Chair', [3.0, 0.0, 1.0]),
        'Bread|1': _vis('Bread', [0.1, 0.0, 1.0]),
        'Bread|2': _vis('Bread', [4.0, 0.0, 1.0]),
    }
    episode_dup_lm = {'steps': [_step(0, two_bread)]}
    names = resolve_named_objects(episode_dup_lm, [0], ['Chair|1'])
    assert names is not None
    assert names['Chair|1'].mode == 'mark'

    # Landmark never co-visible with the target → mark
    split = [
        _step(0, {'Chair|1': _vis('Chair', [0.0, 0.0, 1.0])}),
        _step(
            1,
            {
                'Chair|2': _vis('Chair', [3.0, 0.0, 1.0]),
                'Bread|1': _vis('Bread', [0.1, 0.0, 1.0]),
            },
            x=0.5,
        ),
    ]
    names = resolve_named_objects({'steps': split}, [0, 1], ['Chair|1'])
    assert names is not None
    assert names['Chair|1'].mode == 'mark'


def test_questions_have_no_landmark_relative_direction():
    episode = {
        'scene': 'DupScene',
        'episode_meta': {'camera': {'width': 396, 'height': 224}},
        'agent_trajectory': [
            {'step': 0, 'position': [0.0, 1.0, 0.0], 'rotation': [0.0, 0.0, 0.0]},
        ],
        'steps': [
            _step(
                0,
                {
                    'Chair|1': _vis('Chair', [0.4, 0.0, 1.2]),
                    'Chair|2': _vis('Chair', [0.6, 0.0, 1.2]),
                    'Bread|1': _vis('Bread', [0.5, 0.0, 1.0]),
                },
            )
        ],
    }
    items = generate_items_for_episode(
        episode,
        constructs=['egocentric_encoding'],
        max_per_construct=5,
        styles=('concise', 'verbose'),
    )
    import re

    banned = re.compile(r'\b(left|right|ahead|behind) of the\b', re.I)
    ok = [i for i in items if i.get('status') == 'ok']
    assert ok
    for item in ok:
        assert banned.search(item['question']) is None
        assert 'left of the' not in (item.get('question') or '')


def test_mark_colours_follow_shown_frames_only():
    """Colours are assigned inside one item's frames and may be reused later."""
    episode = {
        'steps': [
            _step(
                0,
                {
                    'Chair|1': _vis('Chair', [0.0, 0.0, 1.0]),
                    'Chair|2': _vis('Chair', [2.0, 0.0, 1.0]),
                },
            ),
            _step(
                1,
                {
                    'Table|1': _vis('Table', [0.0, 0.0, 1.0]),
                    'Table|2': _vis('Table', [2.0, 0.0, 1.0]),
                },
                x=1.0,
            ),
        ]
    }
    chairs = episode_mark_colours(episode, [0])
    tables = episode_mark_colours(episode, [1])
    assert set(chairs) == {'Chair|1', 'Chair|2'}
    assert set(tables) == {'Table|1', 'Table|2'}
    assert chairs['Chair|1'] == tables['Table|1']
    names = resolve_named_objects(episode, [0], ['Chair|1'])
    assert names is not None
    assert 'Table|1' not in names['Chair|1'].marks


def test_same_object_id_same_colour_across_items():
    episode = {
        'steps': [
            _step(
                0,
                {
                    'Chair|1': _vis('Chair', [0.0, 0.0, 1.0]),
                    'Chair|2': _vis('Chair', [2.0, 0.0, 1.0]),
                },
            )
        ]
    }
    colours = episode_mark_colours(episode)
    assert colours['Chair|1'] != colours['Chair|2']
    a = resolve_named_objects(episode, [0], ['Chair|1'])
    b = resolve_named_objects(episode, [0], ['Chair|2'])
    assert a is not None and b is not None
    assert a['Chair|1'].marks['Chair|1'] == b['Chair|2'].marks['Chair|1']
    assert a['Chair|1'].marks['Chair|1'] == colours['Chair|1']


def test_no_mark_on_failed_visibility_frame(tmp_path):
    from PIL import Image

    bg = (10, 10, 10)
    w, h = 200, 100
    p0 = tmp_path / 'f0.png'
    p1 = tmp_path / 'f1.png'
    Image.new('RGB', (w, h), bg).save(p0)
    Image.new('RGB', (w, h), bg).save(p1)
    bbox = [40, 20, 80, 70]
    episode = {
        'episode_meta': {'camera': {'width': w, 'height': h}},
        'steps': [
            _step(
                0,
                {'Chair|1': _vis('Chair', [0.0, 0.0, 1.0], bbox=bbox)},
                image=str(p0),
            ),
            {
                **_step(1, {}, image=str(p1), x=0.5),
                'visible_objects': {},
            },
        ],
    }
    dest = tmp_path / 'marked'
    copies = draw_marked_frames(
        episode, [str(p0), str(p1)], [0, 1], {'Chair|1': 'magenta'}, dest
    )
    marked = Image.open(copies[0]).convert('RGB')
    hidden = Image.open(copies[1]).convert('RGB')
    # Top-centre of the bbox should be a magenta-ish filled dot.
    px = marked.getpixel((60, 29))
    assert px[0] > 200 and px[1] < 80
    # Hidden frame: visibility filter failed, so the copy stays the background.
    assert hidden.getpixel((60, 29)) == bg


def test_all_marks_skips_proximity():
    episode = {
        'steps': [
            _step(
                0,
                {
                    'Chair|1': _vis('Chair', [0.0, 0.0, 1.0]),
                    'Chair|2': _vis('Chair', [3.0, 0.0, 1.0]),
                    'Bread|1': _vis('Bread', [0.1, 0.0, 1.0]),
                },
            )
        ]
    }
    cfg = replace(
        get_generation_config(),
        disambiguation=replace(get_generation_config().disambiguation, all_marks=True),
    )
    with use_generation_config(cfg):
        names = resolve_named_objects(episode, [0], ['Chair|1'])
    assert names is not None
    assert names['Chair|1'].mode == 'mark'
    assert 'magenta' in names['Chair|1'].phrase


def test_item_id_stable_across_proximity_settings():
    episode = {
        'scene': 'DupScene',
        'episode_meta': {'camera': {'width': 396, 'height': 224}},
        'agent_trajectory': [
            {'step': 0, 'position': [0.0, 1.0, 0.0], 'rotation': [0.0, 0.0, 0.0]},
        ],
        'steps': [
            _step(
                0,
                {
                    'Cup|1': _vis('Cup', [0.2, 0.0, 1.0]),
                    'Bread|1': _vis('Bread', [0.5, 0.0, 1.0]),
                },
            )
        ],
    }

    def _key(item):
        return (
            item['construct'],
            item.get('query_step'),
            item.get('queried_object_id'),
            item.get('question_style'),
        )

    by_setting = {}
    for setting in ('current', 'strict', 'sr3d'):
        cfg = replace(
            get_generation_config(),
            disambiguation=replace(
                get_generation_config().disambiguation, proximity_setting=setting
            ),
        )
        with use_generation_config(cfg):
            items = generate_items_for_episode(
                deepcopy(episode),
                constructs=['egocentric_encoding'],
                max_per_construct=3,
                styles=('concise',),
            )
        by_setting[setting] = {
            _key(i): i['item_id'] for i in items if i.get('status') == 'ok'
        }
    shared = set(by_setting['current']) & set(by_setting['strict']) & set(by_setting['sr3d'])
    assert shared
    for k in shared:
        assert by_setting['current'][k] == by_setting['strict'][k] == by_setting['sr3d'][k]


def test_unique_objects_have_none_disambiguation():
    episode = {
        'scene': 'UniqueScene',
        'episode_meta': {'camera': {'width': 396, 'height': 224}},
        'agent_trajectory': [
            {'step': 0, 'position': [0.0, 1.0, 0.0], 'rotation': [0.0, 0.0, 0.0]},
        ],
        'steps': [
            _step(
                0,
                {
                    'Cup|1': _vis('Cup', [0.2, 0.0, 1.0]),
                    'Bread|1': _vis('Bread', [0.5, 0.0, 1.0]),
                },
            )
        ],
    }
    items = generate_items_for_episode(
        episode,
        constructs=['egocentric_encoding'],
        max_per_construct=3,
        styles=('concise',),
    )
    ok = [i for i in items if i.get('status') == 'ok']
    assert ok
    for item in ok:
        if item.get('disambiguation') == 'none':
            assert not item.get('marks')
            assert 'marked with a' not in item['question']
            assert 'close to the' not in item['question']
