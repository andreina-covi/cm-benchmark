"""Tests for SPOC episode folder / annotations path discovery."""

from pathlib import Path

import pytest

from cm_benchmark.generator.episode_paths import (
    is_structural_object,
    list_collection_episodes,
    resolve_annotations_dir,
    resolve_episode_paths,
    resolve_image_path,
)


def test_is_structural_object():
    assert is_structural_object(obj_type='Floor')
    assert is_structural_object(object_id='Wall|2|1')
    assert not is_structural_object(object_id='Window|0')
    assert not is_structural_object(obj_type='Cup', object_id='Cup|1')


def test_resolve_annotations_dir_accepts_episode_root(tmp_path: Path):
    root = tmp_path / 'run'
    ann = root / 'annotations'
    imgs = root / 'images'
    ann.mkdir(parents=True)
    imgs.mkdir()
    (ann / 'navigation-house_9.csv').write_text('timestep\n0\n')
    (ann / 'episode_meta-house_9.json').write_text(
        '{"scene_id":"house_9","images_dir":"images","annotations_dir":"annotations"}'
    )
    found_ann, found_root = resolve_annotations_dir(root)
    assert found_ann == ann.resolve()
    assert found_root == root.resolve()


def test_resolve_annotations_dir_accepts_annotations_folder(tmp_path: Path):
    root = tmp_path / 'run'
    ann = root / 'annotations'
    ann.mkdir(parents=True)
    (ann / 'navigation-house_9.csv').write_text('timestep\n0\n')
    found_ann, found_root = resolve_annotations_dir(ann)
    assert found_ann == ann.resolve()
    assert found_root == root.resolve()


def test_resolve_episode_paths_from_root(tmp_path: Path):
    root = tmp_path / '07_16_run'
    ann = root / 'annotations'
    imgs = root / 'images'
    ann.mkdir(parents=True)
    imgs.mkdir()
    (imgs / 'img_0.png').write_bytes(b'x')
    (ann / 'navigation-house_001030.csv').write_text(
        'timestep,obj-id,path\n0,Cup|1,img_0.png\n'
    )
    (ann / 'objects-house_001030.csv').write_text(
        'obj-id,obj-type,obj-pos-x,obj-pos-y,obj-pos-z,obj-rot-x,obj-rot-y,obj-rot-z,'
        'receptacleObjectIds,bBox-center-x,bBox-center-y,bBox-center-z,size-x,size-y,size-z\n'
        'Cup|1,Cup,0,0,0,0,0,0,[],0,0,0,0.1,0.1,0.1\n'
    )
    (ann / 'episode_meta-house_001030.json').write_text(
        '{"scene_id":"house_001030","images_dir":"images","annotations_dir":"annotations"}'
    )
    paths = resolve_episode_paths(root)
    assert paths['scene_id'] == 'house_001030'
    assert paths['annotations_dir'] == ann.resolve()
    assert paths['images_dir'] == imgs.resolve()
    assert paths['navigation'].name.startswith('navigation-')


def test_resolve_image_path_relative_and_basename(tmp_path: Path):
    imgs = tmp_path / 'images'
    imgs.mkdir()
    target = imgs / 'img_3.png'
    target.write_bytes(b'png')
    assert resolve_image_path('img_3.png', images_dir=imgs, timestep=3) == str(target.resolve())
    assert resolve_image_path(
        '/missing/elsewhere/img_3.png', images_dir=imgs, timestep=99
    ) == str(target.resolve())
    assert resolve_image_path(None, images_dir=imgs, timestep=3) == str(target.resolve())


def test_list_collection_episodes_walks_timestamp_children(tmp_path: Path):
    """A root of <timestamp>/{annotations,images} yields one episode per child."""
    root = tmp_path / 'navigation'
    for stamp, scene in (
        ('09_23_2026_16_31_04_526137', 'house_007514'),
        ('09_23_2026_16_35_09_464480', 'house_001030'),
    ):
        ann = root / stamp / 'annotations'
        (root / stamp / 'images').mkdir(parents=True)
        ann.mkdir()
        (ann / f'navigation-{scene}.csv').write_text('timestep\n0\n')
        (ann / f'objects-{scene}.csv').write_text('obj-id\n')
    found = list_collection_episodes(root)
    assert [p.name for p in found] == [
        '09_23_2026_16_31_04_526137',
        '09_23_2026_16_35_09_464480',
    ]
    from cm_benchmark.generator.episode_paths import discover_scene_id

    assert discover_scene_id(found[0] / 'annotations') == 'house_007514'
    assert discover_scene_id(found[1] / 'annotations') == 'house_001030'


def test_list_collection_episodes_keeps_a_single_episode_folder(tmp_path: Path):
    root = tmp_path / '09_23_2026_16_31_04_526137'
    ann = root / 'annotations'
    ann.mkdir(parents=True)
    (root / 'images').mkdir()
    (ann / 'navigation-house_007514.csv').write_text('timestep\n0\n')
    assert list_collection_episodes(root) == [root.resolve()]
    assert list_collection_episodes(ann) == [ann.resolve()]
