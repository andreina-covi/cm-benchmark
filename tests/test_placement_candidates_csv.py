"""Nav-generator ingest of placement_candidates CSVs (keep/drop at read)."""

from __future__ import annotations

from pathlib import Path

from cm_benchmark.generator.ai2thor_nav_generator import (
    Ai2ThorNavGenerator,
    _is_floor_placement_dest,
    _receptacle_parent_ok,
)

FIXTURES = Path(__file__).parent / 'fixtures'
NAV_CSV = FIXTURES / 'navigation_tiny.csv'
OBJ_CSV = FIXTURES / 'objects_tiny.csv'


def test_floor_dest_detection():
    assert _is_floor_placement_dest('Sofa|1', 'floor')
    assert _is_floor_placement_dest('Floor', None)
    assert _is_floor_placement_dest('Floor|0', 'unknown')
    assert not _is_floor_placement_dest('Floor', 'receptacle')
    assert not _is_floor_placement_dest('CounterTop|1', 'receptacle')
    assert not _is_floor_placement_dest('CounterTop|1', None)


def test_receptacle_parent_ok_rules():
    assert _receptacle_parent_ok('CounterTop|1', 'receptacle', 'CounterTop|1')
    assert not _receptacle_parent_ok('CounterTop|1', 'receptacle', 'Table|1')
    assert not _receptacle_parent_ok('CounterTop|1', 'receptacle', None)
    assert _receptacle_parent_ok('Sofa|1', 'floor', 'Floor')
    assert _receptacle_parent_ok('Floor', None, None)


def _generator(tmp_path, scene_id='house_test'):
    return Ai2ThorNavGenerator(
        path_navigation=str(NAV_CSV),
        path_objects=str(OBJ_CSV),
        output_path=str(tmp_path),
        scene_id=scene_id,
    )


def test_read_drops_failed_and_parent_mismatch(tmp_path, capsys):
    csv = tmp_path / 'placement_candidates.csv'
    csv.write_text(
        'obj-id,anchor-id,anchor-kind,parent-receptacle,pos-x,pos-y,pos-z,success,radius_m\n'
        'Cup|1,CounterTop|1,receptacle,CounterTop|1,1,0.9,1,true,1.2\n'
        'Cup|2,CounterTop|1,receptacle,CounterTop|1,1,0.9,1,false,1.2\n'
        'Cup|3,CounterTop|1,receptacle,Table|1,1,0.9,1,true,1.2\n'
        'Cup|4,CounterTop|1,receptacle,,1,0.9,1,true,1.2\n'
        'Cup|5,Floor,floor,Floor,1,0.9,1,true,1.2\n'
        'Cup|6,Sofa|1,floor,Floor,1,0.9,1,true,1.2\n'
    )
    gen = _generator(tmp_path)
    rows = gen._read_placement_candidates(csv)
    ids = [r['obj_id'] for r in rows]
    assert ids == ['Cup|1', 'Cup|5', 'Cup|6']
    assert all(r['success'] is True for r in rows)
    out = capsys.readouterr().out
    assert 'placement_candidates house_test: kept 3, dropped 3' in out
    assert 'success!=true: 1' in out
    assert 'receptacle parent!=anchor: 2' in out


def test_read_legacy_missing_success_column_keeps_row(tmp_path):
    csv = tmp_path / 'placement_candidates.csv'
    csv.write_text(
        'obj-id,anchor-id,anchor-kind,parent-receptacle,pos-x,pos-y,pos-z\n'
        'Cup|1,CounterTop|1,receptacle,CounterTop|1,1,0.9,1\n'
    )
    gen = _generator(tmp_path)
    rows = gen._read_placement_candidates(csv)
    assert len(rows) == 1
    assert rows[0]['success'] is True
    assert rows[0]['obj_id'] == 'Cup|1'


def test_dropped_rows_not_in_episode_json(tmp_path):
    csv = tmp_path / 'placement_candidates.csv'
    csv.write_text(
        'obj-id,anchor-id,anchor-kind,parent-receptacle,pos-x,pos-y,pos-z,success,radius_m\n'
        'Cup|1,CounterTop|1,receptacle,CounterTop|1,1,0.9,1,true,1.2\n'
        'Cup|2,CounterTop|1,receptacle,Table|1,1,0.9,1,true,1.2\n'
        'Cup|3,CounterTop|1,receptacle,CounterTop|1,1,0.9,1,false,1.2\n'
    )
    gen = _generator(tmp_path)
    gen.placement_candidates = gen._read_placement_candidates(csv)
    episode = {'scene': 'house_test'}
    gen.enrich_episode_data(episode)
    kept = episode['placement_candidates']
    assert [r['obj_id'] for r in kept] == ['Cup|1']
    assert all(r.get('success') is True for r in kept)
