"""Sparse object_state_track helpers."""

from cm_benchmark.generator.ai2thor_nav_generator import sparsify_state_entries, state_at_step


def _entry(step, x=0.0, fov=True, receptacle='A'):
    return {
        'step': step,
        'position': [x, 1.0, 0.0],
        'rotation': [0.0, 0.0, 0.0],
        'visible': True,
        'in_camera_fov': fov,
        'parent_receptacle': receptacle,
        'is_inside_receptacle': True,
        'receptacle_is_open': None,
    }


def test_sparsify_keeps_only_changes():
    dense = [
        _entry(0, x=0.0, fov=True),
        _entry(1, x=0.0, fov=True),   # unchanged → drop
        _entry(2, x=0.0, fov=True),   # unchanged → drop
        _entry(3, x=0.0, fov=False),  # fov change → keep
        _entry(4, x=0.0, fov=False),  # unchanged → drop
        _entry(5, x=1.0, fov=False),  # pose change → keep
    ]
    sparse = sparsify_state_entries(dense)
    assert [e['step'] for e in sparse] == [0, 3, 5]


def test_state_at_step_carry_forward():
    sparse = [
        _entry(0, x=0.0, fov=True),
        _entry(3, x=0.0, fov=False),
        _entry(5, x=1.0, fov=False),
    ]
    assert state_at_step(sparse, 0)['in_camera_fov'] is True
    assert state_at_step(sparse, 1)['in_camera_fov'] is True
    assert state_at_step(sparse, 2)['in_camera_fov'] is True
    assert state_at_step(sparse, 3)['in_camera_fov'] is False
    assert state_at_step(sparse, 4)['position'] == [0.0, 1.0, 0.0]
    assert state_at_step(sparse, 9)['position'] == [1.0, 1.0, 0.0]
    assert state_at_step(sparse, -1) is None


def test_sparsify_region_trajectory():
    from cm_benchmark.generator.ai2thor_nav_generator import sparsify_timestep_series, series_at_step

    dense = [
        {'timestep': 0, 'region_id': 'room|2', 'region_type': 'Kitchen'},
        {'timestep': 1, 'region_id': 'room|2', 'region_type': 'Kitchen'},
        {'timestep': 50, 'region_id': 'room|2', 'region_type': 'Kitchen'},
        {'timestep': 103, 'region_id': 'room|4', 'region_type': 'LivingRoom'},
        {'timestep': 104, 'region_id': 'room|4', 'region_type': 'LivingRoom'},
    ]
    sparse = sparsify_timestep_series(dense, key_fields=('region_id', 'region_type'))
    assert [r['timestep'] for r in sparse] == [0, 103]
    assert series_at_step(sparse, 80)['region_id'] == 'room|2'
    assert series_at_step(sparse, 103)['region_type'] == 'LivingRoom'
