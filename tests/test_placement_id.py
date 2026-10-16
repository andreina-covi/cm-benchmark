"""Invisible displacement from placement_candidates (not displacement_events)."""

from __future__ import annotations

from cm_benchmark.generation.planner import id_placement_funnel, plan_invisible_displacement


def _cup():
    return {
        'category': 'Cup',
        'position': [0.5, 0.9, 0.5],
        'bbox_area': 200.0,
        'min_side': 12.0,
        'visible_pixels': 80.0,
        'size': [0.1, 0.1, 0.1],
    }


def _counter():
    return {
        'category': 'CounterTop',
        'position': [2.0, 0.9, 1.0],
        'bbox_area': 5000.0,
        'min_side': 60.0,
        'visible_pixels': 4000.0,
        'size': [1.0, 0.8, 0.6],
        'aabb_min': [1.5, 0.5, 0.7],
        'aabb_max': [2.5, 1.3, 1.3],
    }


def _ego(target, rel=('front',)):
    return {'source': 'agent', 'target': target, 'angle_relation': ['', '', rel[0]]}


def _placement_episode(*, dest, anchor='CounterTop|1', radius=1.2, extra_vis=None, extra_catalog=None):
    vis0 = {'Cup|1': _cup(), 'CounterTop|1': _counter()}
    vis_later = {'CounterTop|1': _counter()}
    if extra_vis:
        vis0.update(extra_vis)
        vis_later.update(extra_vis)
    catalog = {
        'Cup|1': {
            'category': 'Cup',
            'bbox_center': [0.5, 0.9, 0.5],
            'aabb_min': [0.45, 0.85, 0.45],
            'aabb_max': [0.55, 0.95, 0.55],
            'parent_receptacle': 'Table|1',
        },
        'CounterTop|1': {
            'category': 'CounterTop',
            'bbox_center': [2.0, 0.9, 1.0],
            'aabb_min': [1.5, 0.5, 0.7],
            'aabb_max': [2.5, 1.3, 1.3],
        },
        'Table|1': {
            'category': 'Table',
            'bbox_center': [0.5, 0.8, 0.5],
            'aabb_min': [0.0, 0.4, 0.0],
            'aabb_max': [1.0, 0.9, 1.0],
        },
    }
    if extra_catalog:
        catalog.update(extra_catalog)
    return {
        'camera': {'width': 396, 'height': 224, 'fov_vertical_deg': 59},
        'episode_meta': {'fov_stride': 1, 'camera': {'width': 396, 'height': 224, 'fov_v': 59}},
        'object_catalog': catalog,
        'steps': [
            {
                'step': 0,
                'image_path': '/img_0.png',
                'agent': {'position': [0, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': vis0,
                'edges_egocentric': [_ego(oid) for oid in vis0],
            },
            {
                'step': 1,
                'image_path': '/img_1.png',
                'agent': {'position': [1, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': vis_later,
                'edges_egocentric': [_ego(oid) for oid in vis_later],
            },
            {
                'step': 2,
                'image_path': '/img_2.png',
                'agent': {'position': [2, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': vis_later,
                'edges_egocentric': [_ego(oid) for oid in vis_later],
            },
        ],
        'agent_trajectory': [
            {'step': 0, 'position': [0, 1, 0], 'rotation': [0, 0, 0]},
            {'step': 1, 'position': [1, 1, 0], 'rotation': [0, 0, 0]},
            {'step': 2, 'position': [2, 1, 0], 'rotation': [0, 0, 0]},
        ],
        'object_state_track': {
            'Cup|1': {
                'category': 'Cup',
                'entries': [
                    {
                        'step': 0,
                        'position': [0.5, 0.9, 0.5],
                        'visible': True,
                        'in_camera_fov': True,
                        'parent_receptacle': 'Table|1',
                    },
                    {
                        'step': 1,
                        'position': [0.5, 0.9, 0.5],
                        'visible': False,
                        'in_camera_fov': False,
                        'parent_receptacle': 'Table|1',
                    },
                    {
                        'step': 2,
                        'position': [0.5, 0.9, 0.5],
                        'visible': False,
                        'in_camera_fov': False,
                        'parent_receptacle': 'Table|1',
                    },
                ],
            }
        },
        'placement_candidates': [
            {
                'obj_id': 'Cup|1',
                'anchor_id': anchor,
                'anchor_kind': 'receptacle' if anchor != 'Floor' else 'floor',
                'position': dest,
                'success': True,
                'radius_m': radius,
            }
        ],
        'displacement_events': [],
    }


# Behind the camera for yaw=0 (look +Z) at x=1 and x=2.
_BEHIND = [0.0, 0.9, -3.0]


def test_placement_emits_item_and_uses_csv_radius():
    episode = _placement_episode(dest=_BEHIND)
    facts = plan_invisible_displacement(episode, max_items=3)
    ok = [f for f in facts if f.status == 'ok']
    assert ok, [f.reason for f in facts]
    fact = ok[0]
    assert fact.queried_object_id == 'Cup|1'
    assert 'placement_candidates' in fact.answer_source[0]
    assert fact.extra.get('new_location')
    assert fact.displacement_event.get('to_position') == _BEHIND


def test_placement_rejects_current_receptacle():
    episode = _placement_episode(dest=_BEHIND, anchor='Table|1')
    facts = plan_invisible_displacement(episode, max_items=3)
    assert not any(f.status == 'ok' for f in facts)


def test_placement_disambiguates_confusable_anchor():
    extra_vis = {
        'CounterTop|2': {
            'category': 'CounterTop',
            'position': [4.0, 0.9, 1.0],
            'bbox_area': 5000.0,
            'min_side': 60.0,
            'visible_pixels': 4000.0,
            'synset': 'counter.n.01',
            'aabb_min': [3.5, 0.5, 0.7],
            'aabb_max': [4.5, 1.3, 1.3],
        }
    }
    extra_cat = {
        'CounterTop|1': {
            'category': 'CounterTop',
            'synset': 'counter.n.01',
            'bbox_center': [2.0, 0.9, 1.0],
            'aabb_min': [1.5, 0.5, 0.7],
            'aabb_max': [2.5, 1.3, 1.3],
        },
        'CounterTop|2': {
            'category': 'CounterTop',
            'synset': 'counter.n.01',
            'bbox_center': [4.0, 0.9, 1.0],
            'aabb_min': [3.5, 0.5, 0.7],
            'aabb_max': [4.5, 1.3, 1.3],
        },
    }
    episode = _placement_episode(
        dest=_BEHIND, extra_vis=extra_vis, extra_catalog=extra_cat
    )
    episode['object_catalog']['CounterTop|1']['synset'] = 'counter.n.01'
    facts = plan_invisible_displacement(episode, max_items=3)
    ok = [f for f in facts if f.status == 'ok']
    assert ok, [f.reason for f in facts]
    loc = ok[0].extra.get('new_location') or ''
    assert 'Counter' in loc
    assert 'close to' in loc or 'marked with' in loc


def test_placement_rejects_parent_not_anchor():
    episode = _placement_episode(dest=_BEHIND)
    episode['placement_candidates'][0]['parent_receptacle'] = 'Table|1'
    facts = plan_invisible_displacement(episode, max_items=3)
    assert not any(f.status == 'ok' for f in facts)


def test_placement_rejects_in_frustum_dest():
    # Ahead of yaw-0 camera at both move and query.
    episode = _placement_episode(dest=[1.0, 0.9, 4.0])
    facts = plan_invisible_displacement(episode, max_items=3)
    assert not any(f.status == 'ok' for f in facts)


def test_placement_floor_uses_csv_radius_not_config():
    episode = _placement_episode(dest=_BEHIND, anchor='Floor', radius=0.05)
    # Counter is ~4 m from dest — too far for a 0.05 m CSV radius.
    facts = plan_invisible_displacement(episode, max_items=3)
    assert not any(f.status == 'ok' for f in facts)

    episode = _placement_episode(dest=_BEHIND, anchor='Floor', radius=8.0)
    facts = plan_invisible_displacement(episode, max_items=3)
    ok = [f for f in facts if f.status == 'ok']
    assert ok
    assert ok[0].extra.get('floor_anchor_radius_m') == 8.0
    assert ok[0].extra.get('floor_anchor_id') == 'CounterTop|1'


def test_last_hide_at_query_still_yields_a_window():
    """A query where the object is hidden only on that frame is still a window."""
    from cm_benchmark.generation.planner import _id_windows_for_object

    episode = _placement_episode(dest=_BEHIND)
    # Steps 1 and 2 share a visible-object dict in the fixture; copy before editing.
    episode['steps'][1]['visible_objects'] = dict(episode['steps'][1]['visible_objects'])
    episode['steps'][1]['visible_objects']['Cup|1'] = _cup()
    episode['steps'][1]['edges_egocentric'] = list(episode['steps'][1]['edges_egocentric'])
    episode['steps'][1]['edges_egocentric'].append(_ego('Cup|1'))
    seen = episode['object_state_track']['Cup|1']['entries'][1]
    seen['visible'] = True
    seen['in_camera_fov'] = True
    windows = _id_windows_for_object(episode, 'Cup|1')
    # Latest encode is the step-1 sighting; the only later hide is the query.
    assert windows == [(1, 2, 2)]


def test_question_filter_keeps_raw_encode_window():
    """Stripping the prop from question-visible objects must not zero the window."""
    from cm_benchmark.generation.planner import _id_windows_for_object

    episode = _placement_episode(dest=_BEHIND)
    for step in episode['steps']:
        raw = step['visible_objects']
        step['visible_frames'] = {
            oid: {k: o[k] for k in ('bbox_area', 'min_side', 'visible_pixels') if k in o}
            for oid, o in raw.items()
        }
        step['visible_objects'] = {oid: o for oid, o in raw.items() if oid != 'Cup|1'}
    windows = _id_windows_for_object(episode, 'Cup|1')
    assert windows
    assert any(move == query for _enc, move, query in windows)


def test_floor_placement_survives_failed_swap_partner():
    """A floor trial is recall when the occupying object cannot be a swap."""
    episode = _placement_episode(dest=_BEHIND, anchor='Floor', radius=8.0)
    episode['object_catalog']['Statue|1'] = {
        'category': 'Statue',
        'aabb_min': [-0.2, 0.0, -3.2],
        'aabb_max': [0.2, 2.0, -2.8],
    }
    funnel = dict(id_placement_funnel(episode))
    assert funnel['anchor_nameable'] >= 1
    assert funnel['swap_partner_ok'] == funnel['anchor_nameable']
    facts = plan_invisible_displacement(episode, max_items=3)
    ok = [f for f in facts if f.status == 'ok']
    assert ok, [f.reason for f in facts]
    assert all(f.extra.get('template_mode') == 'recall_direction' for f in ok)


def test_id_windows_use_latest_hide_not_first():
    """Move is the latest not-visible step in (encode, query], so the window is short."""
    from cm_benchmark.generation.planner import _id_windows_for_object

    episode = _placement_episode(dest=_BEHIND)
    windows = _id_windows_for_object(episode, 'Cup|1')
    assert windows
    assert all(move == query for _enc, move, query in windows)
    assert (0, 2, 2) in windows
    assert (0, 1, 2) not in windows


def test_late_move_keeps_dest_hidden_only_at_query():
    """A dest inside the frustum at the first hide still passes when the query faces away."""
    episode = _placement_episode(dest=[1.0, 0.9, 4.0])
    episode['steps'][2]['agent']['rotation'] = [0, 180, 0]
    episode['agent_trajectory'][2]['rotation'] = [0, 180, 0]
    facts = plan_invisible_displacement(episode, max_items=3)
    ok = [f for f in facts if f.status == 'ok']
    assert ok, [f.reason for f in facts]
    assert ok[0].displacement_event.get('at_timestep') == ok[0].query_step


def test_colliding_distractor_replaced_with_unused_ego_label():
    """Same ego label as the answer becomes an unused bank label; the answer stays once."""
    from cm_benchmark.generation.constructs import EGO_DIRECTION_OPTIONS
    from cm_benchmark.generation.planner import _ego_pool_from_specs, _try_ego_direction_fact, _unused_ego_label

    assert _unused_ego_label(set(EGO_DIRECTION_OPTIONS)) is None
    episode = _placement_episode(dest=_BEHIND)
    # Both poses are behind the yaw-0 camera, so they share one ego label.
    specs = [
        {'role': 'chosen', 'position': [0.0, 0.9, -3.0], 'is_answer': True},
        {'role': 'original_location', 'position': [0.2, 0.9, -4.0], 'is_answer': False},
    ]
    answer, pool, seeds = _ego_pool_from_specs(episode, specs, 2)
    assert answer in EGO_DIRECTION_OPTIONS
    assert pool.count(answer) == 1
    assert len(pool) == len(set(pool)) == 2
    assert any(lab != answer for lab in pool)
    assert 'ego_bank_filler' in seeds
    fact = _try_ego_direction_fact(
        episode,
        {'obj_id': 'Cup|1', 'at_timestep': 2, 'hidden_during': True},
        obj_id='Cup|1',
        object_type='Cup',
        enc_idx=0,
        query_step=2,
        at_t=2,
        images=['/img_0.png', '/img_2.png'],
        specs=specs,
        template_mode='recall_direction',
        answer_source=['placement_candidates'],
        extra_fields={},
    )
    assert fact is not None and fact.status == 'ok'
    assert fact.options_pool.count(fact.answer_label) == 1
    assert len(fact.options_pool) == 4


def test_id_funnel_reports_gates_in_order():
    episode = _placement_episode(dest=_BEHIND)
    funnel = id_placement_funnel(episode)
    names = [g for g, _n in funnel]
    assert names == [
        'successful_candidates',
        'encode_hidden_query_window',
        'displaced_object_nameable',
        'min_displacement',
        'dest_not_current_receptacle',
        'receptacle_parent_eq_anchor',
        'dest_outside_frustum',
        'anchor_nameable',
        'swap_partner_ok',
        'diagnostic_options',
    ]
    by_gate = dict(funnel)
    assert by_gate['successful_candidates'] == 1
    assert by_gate['diagnostic_options'] >= 1


def test_events_path_still_used_without_placement():
    from cm_benchmark.generation.planner import _plan_id_from_events

    cup = _cup()
    counter = _counter()
    episode = {
        'steps': [
            {
                'step': 0,
                'image_path': '/img_0.png',
                'agent': {'position': [0, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {'Cup|1': cup, 'CounterTop|1': counter},
                'edges_egocentric': [_ego('Cup|1'), _ego('CounterTop|1')],
            },
            {
                'step': 2,
                'image_path': '/img_2.png',
                'agent': {'position': [2, 1, 0], 'rotation': [0, 90, 0]},
                'visible_objects': {'CounterTop|1': counter},
                'edges_egocentric': [_ego('CounterTop|1', ('right',))],
            },
        ],
        'agent_trajectory': [
            {'step': 0, 'position': [0, 1, 0], 'rotation': [0, 0, 0]},
            {'step': 2, 'position': [2, 1, 0], 'rotation': [0, 90, 0]},
        ],
        'object_state_track': {
            'Cup|1': {
                'category': 'Cup',
                'entries': [
                    {'step': 0, 'position': [0.5, 0.9, 0.5], 'visible': True, 'in_camera_fov': True},
                    {'step': 2, 'position': [3.0, 0.9, 0.5], 'visible': False, 'in_camera_fov': False},
                ],
            }
        },
        'displacement_events': [
            {
                'event_id': 'disp_0',
                'obj_id': 'Cup|1',
                'hidden_during': True,
                'at_timestep': 2,
                'from_position': [0.5, 0.9, 0.5],
                'to_position': [3.0, 0.9, 0.5],
                'from_receptacle': 'Table|1',
                'to_receptacle': 'CounterTop|1',
                'moved_via': 'place',
            }
        ],
        'displacement_candidates': [
            {
                'event_id': 'disp_0',
                'obj_id': 'Cup|1',
                'candidate_role': 'chosen',
                'candidate_position': [3.0, 0.9, 0.5],
                'candidate_receptacle': 'CounterTop|1',
            },
            {
                'event_id': 'disp_0',
                'obj_id': 'Cup|1',
                'candidate_role': 'original_location',
                'candidate_position': [0.5, 0.9, 0.5],
                'candidate_receptacle': 'Table|1',
            },
        ],
    }
    facts = _plan_id_from_events(episode, max_items=2)
    assert any(f.status == 'ok' for f in facts)
    assert plan_invisible_displacement(episode, max_items=2)
