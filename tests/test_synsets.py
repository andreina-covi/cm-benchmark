"""Confusable grouping, word-split types, and mark colours by group."""

from __future__ import annotations

import pytest

from cm_benchmark.generation.constructs import object_type_from_id, words_from_obj_type
from cm_benchmark.generation.disambiguation import episode_mark_colours, resolve_named_objects
from cm_benchmark.generation.synsets import are_confusable, confusable_group_of, objects_are_confusable


def test_words_from_obj_type():
    assert words_from_obj_type('AlarmClock') == 'Alarm Clock'
    assert words_from_obj_type('FloorLamp') == 'Floor Lamp'
    assert words_from_obj_type('chair-diningtable') == 'chair diningtable'
    assert words_from_obj_type('Chair') == 'Chair'
    assert object_type_from_id(
        'FloorLamp|4|2', {'FloorLamp|4|2': {'category': 'FloorLamp'}}
    ) == 'Floor Lamp'


def test_same_synset_is_confusable():
    assert are_confusable('straight_chair.n.01', 'straight_chair.n.01')
    episode = {
        'object_catalog': {
            'Chair|1': {'category': 'Chair', 'synset': 'straight_chair.n.01'},
            'Chair|2': {'category': 'Armchair', 'synset': 'straight_chair.n.01'},
        }
    }
    assert objects_are_confusable(episode, 'Chair|1', 'Chair|2')
    assert confusable_group_of(episode, 'Chair|1', ['Chair|1', 'Chair|2', 'Cup|1']) == {
        'Chair|1',
        'Chair|2',
    }


def test_type_fallback_when_synset_missing():
    assert are_confusable(None, None, type_a='Chair', type_b='Chair')
    assert not are_confusable(None, None, type_a='Chair', type_b='Table')


def test_wordnet_hypernym_or_skip():
    try:
        from nltk.corpus import wordnet2022 as wn

        wn.synset('chair.n.01')
    except (LookupError, ImportError):
        pytest.skip('nltk wordnet2022 not installed')
    # furniture is a hypernym of chair
    assert are_confusable('chair.n.01', 'furniture.n.01')


def test_mark_colours_group_by_confusable_synset():
    episode = {
        'object_catalog': {
            'Chair|1': {'category': 'Chair', 'synset': 'straight_chair.n.01'},
            'Armchair|1': {'category': 'Armchair', 'synset': 'straight_chair.n.01'},
            'Cup|1': {'category': 'Cup', 'synset': 'cup.n.01'},
        },
        'steps': [
            {
                'step': 0,
                'visible_objects': {
                    'Chair|1': {'category': 'Chair'},
                    'Armchair|1': {'category': 'Armchair'},
                    'Cup|1': {'category': 'Cup'},
                },
            }
        ],
    }
    colours = episode_mark_colours(episode)
    assert colours['Chair|1'] != colours['Armchair|1']
    assert 'Cup|1' not in colours


def test_resolve_named_objects_uses_confusable_not_type_string():
    episode = {
        'object_catalog': {
            'Chair|1': {'category': 'Chair', 'synset': 'straight_chair.n.01'},
            'Armchair|1': {'category': 'Armchair', 'synset': 'straight_chair.n.01'},
        },
        'steps': [
            {
                'step': 0,
                'image_path': '/img_0.png',
                'visible_objects': {
                    'Chair|1': {
                        'category': 'Chair',
                        'position': [0.0, 0.0, 1.0],
                        'bbox_area': 2000,
                        'min_side': 40,
                        'visible_pixels': 500,
                    },
                    'Armchair|1': {
                        'category': 'Armchair',
                        'position': [3.0, 0.0, 1.0],
                        'bbox_area': 2000,
                        'min_side': 40,
                        'visible_pixels': 500,
                    },
                },
            }
        ],
        'episode_meta': {},
    }
    names = resolve_named_objects(episode, [0], ['Chair|1'])
    assert names is not None
    assert names['Chair|1'].mode in ('proximity', 'mark')
    assert names['Chair|1'].object_type == 'Chair'
