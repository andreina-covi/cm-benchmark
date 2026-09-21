"""Tests for first-draft taxonomy Q&A generation."""

from copy import deepcopy
from pathlib import Path

import pytest

from cm_benchmark.generator.ai2thor_nav_generator import Ai2ThorNavGenerator
from cm_benchmark.generation.constructs import object_type_from_id, select_template
from cm_benchmark.generation.pipeline import draft_items_for_episode
from cm_benchmark.generation.planner import plan_episode
from cm_benchmark.generation.templates import _core_question, build_verbose_preamble

FIXTURES = Path(__file__).parent / 'fixtures'
NAV_CSV = FIXTURES / 'navigation_tiny.csv'
OBJ_CSV = FIXTURES / 'objects_tiny.csv'
EPISODE_DIR = FIXTURES / 'episode_tiny'


@pytest.fixture
def tiny_episode(tmp_path):
    gen = Ai2ThorNavGenerator(
        path_navigation=str(NAV_CSV),
        path_objects=str(OBJ_CSV),
        output_path=str(tmp_path),
        output_filename='episode.json',
    )
    return gen.collect_episode_data(extra_data={'scene': 'TinyScene'})


@pytest.fixture
def folder_episode(tmp_path):
    gen = Ai2ThorNavGenerator(
        csv_path_folder=str(EPISODE_DIR),
        output_path=str(tmp_path),
        output_filename='folder_episode.json',
    )
    return gen.collect_episode_data(extra_data={'scene': 'ignore'})


def _corridor_episode(cells, grid: float = 0.25, scene_id: str = 'corridor') -> dict:
    """Synthetic episode whose navigable space is a corridor along ``cells``.

    The agent walks the cells in order; a Fridge sits at the first cell and a
    Toilet at the last, both distinguishable at the near end of the walk.
    """

    def ego(target):
        return {
            'source': 'agent',
            'target': target,
            'distance_metric': 1.0,
            'distance_label': 'near',
            'visible': True,
            'angle_relation': ['', '', 'front'],
            'inferred': False,
        }

    def detection(category, position):
        return {
            'category': category,
            'position': position,
            'bbox': [100, 60, 220, 190],
            'bbox_area': 15600.0,
            'min_side': 120.0,
            'visible_pixels': 12000.0,
            'occupancy_ratio': 0.6,
            'obj_distance': 1.0,
            'local_point': [160, 125],
        }

    cells = [(round(x, 3), round(z, 3)) for x, z in cells]
    open_cells = set()
    for x, z in cells:
        for dx in (-grid, 0.0, grid):
            for dz in (-grid, 0.0, grid):
                open_cells.add((round(x + dx, 3), round(z + dz, 3)))
    nodes = [
        {'node_id': f'n{i}', 'x': x, 'y': 1.0, 'z': z}
        for i, (x, z) in enumerate(sorted(open_cells))
    ]
    nav_graph = {
        'snapshots': {
            'episode_start': {
                'params': {
                    'grid_size': grid,
                    'agent_move_m': grid,
                    'agent_rotation_deg': 45.0,
                    'edge_connectivity': '8',
                },
                'nodes': nodes,
                'edges': [],
            }
        }
    }
    trajectory, steps = [], []
    yaw = 0.0
    for i, (x, z) in enumerate(cells):
        if i > 0:
            px, pz = cells[i - 1]
            yaw = 0.0 if z > pz else (180.0 if z < pz else 90.0)
        pose = {'position': [x, 1.0, z], 'rotation': [0.0, yaw, 0.0]}
        trajectory.append({'step': i, 'image_path': f'/img_{i}.png', **pose})
        visible, ego_edges = {}, []
        if i <= 1:
            first = cells[0]
            visible['Fridge|1'] = detection('Fridge', [first[0], 1.0, first[1]])
            ego_edges.append(ego('Fridge|1'))
        if i >= len(cells) - 2:
            last = cells[-1]
            visible['Toilet|1'] = detection('Toilet', [last[0], 1.0, last[1]])
            ego_edges.append(ego('Toilet|1'))
        steps.append(
            {
                'step': i,
                'image_path': f'/img_{i}.png',
                'action': 'move_ahead',
                'degrees': 45,
                'agent': pose,
                'visible_objects': visible,
                'non_visible_objects': {},
                'edges_egocentric': ego_edges,
                'edges_allocentric': [],
                'edges_object_frame': [],
                'edges_inferred': [],
            }
        )
    return {
        'episode_id': scene_id,
        'scene_id': scene_id,
        'environment': 'ai2thor',
        'episode_meta': {
            'camera': {'width': 396, 'height': 224, 'fov_vertical_deg': 59},
            'agent': {'rotation_deg': 45, 'movement_constant': grid},
        },
        'nav_graph': nav_graph,
        'agent_trajectory': trajectory,
        'steps': steps,
        'agent_actions': [
            {'step': i, 'action': 'move_ahead', 'degrees': 45} for i in range(len(cells))
        ],
    }


@pytest.fixture
def u_corridor_episode():
    """U-shaped corridor (+Z 12 cells, +X 6, -Z 12): two real corners, big detour.

    The collected episodes on hand are single straight rooms, which legitimately
    produce no route item; this scene exercises the path that does.
    """
    g = 0.25
    cells = (
        [(0.0, k * g) for k in range(13)]
        + [(k * g, 12 * g) for k in range(1, 7)]
        + [(6 * g, (12 - k) * g) for k in range(1, 13)]
    )
    return _corridor_episode(cells, grid=g, scene_id='u_corridor')


def _two_room_episode() -> dict:
    """Two rooms joined by one open door; the agent only ever walks room A.

    Room B is sighted through the doorway, so every A-B pair is never-walked but
    viewed — the survey_based_route_planning setup. The collected episodes on
    hand have no through-door evidence at all, so this is the only scene that
    exercises the construct end to end.
    """
    g = 0.25

    def frange(a, b):
        out, x = [], a
        while x <= b + 1e-9:
            out.append(round(x, 3))
            x += g
        return out

    def ego(target):
        return {
            'source': 'agent',
            'target': target,
            'distance_metric': 2.0,
            'distance_label': 'near',
            'visible': True,
            'angle_relation': ['', '', 'front'],
            'inferred': False,
        }

    def detection(category, position):
        return {
            'category': category,
            'position': position,
            'bbox': [100, 60, 220, 190],
            'bbox_area': 15600.0,
            'min_side': 120.0,
            'visible_pixels': 12000.0,
            'occupancy_ratio': 0.6,
            'obj_distance': 2.0,
            'local_point': [160, 125],
        }

    cells = set()
    for x in frange(0.0, 2.0):
        for z in frange(0.0, 3.0):
            cells.add((x, z))
    for z in frange(1.25, 1.75):
        cells.add((2.25, z))
    for x in frange(2.5, 4.5):
        for z in frange(0.0, 3.0):
            cells.add((x, z))
    nodes = [
        {'node_id': f'n{i}', 'x': x, 'y': 1.0, 'z': z}
        for i, (x, z) in enumerate(sorted(cells))
    ]
    nav_graph = {
        'snapshots': {
            'episode_start': {
                'params': {
                    'grid_size': g,
                    'agent_move_m': g,
                    'agent_rotation_deg': 45.0,
                    'edge_connectivity': '8',
                },
                'nodes': nodes,
                'edges': [],
            }
        }
    }
    landmarks = {
        'Fridge|1': ('Fridge', 'roomA', [0.0, 1.0, 0.0]),
        'CounterTop|1': ('CounterTop', 'roomA', [0.0, 1.0, 3.0]),
        'Toilet|1': ('Toilet', 'roomB', [4.5, 1.0, 0.0]),
        'Bed|1': ('Bed', 'roomB', [4.5, 1.0, 3.0]),
        'Sofa|1': ('Sofa', 'roomB', [3.0, 1.0, 1.5]),
    }
    walk = [(0.0, z) for z in frange(0.0, 3.0)]
    walk += [(x, 3.0) for x in frange(0.25, 2.0)]
    walk += [(2.0, z) for z in reversed(frange(1.5, 2.75))]
    door_step = len(walk) - 1
    # Toilet is sighted over more frames, so it sorts first by salience even
    # though its pair is the easiest. That makes `ids` enumeration order and
    # difficulty order disagree, which is what the ranking test needs.
    shown_at = {
        'Fridge|1': {8, 9},
        'CounterTop|1': {16, 17, door_step - 1, door_step},
        'Toilet|1': set(range(door_step - 5, door_step + 1)),
        'Bed|1': {door_step - 1, door_step},
        'Sofa|1': {door_step - 1, door_step},
    }
    trajectory, steps = [], []
    yaw = 0.0
    for i, (x, z) in enumerate(walk):
        if i > 0:
            px, pz = walk[i - 1]
            yaw = 0.0 if z > pz else (180.0 if z < pz else (90.0 if x > px else 270.0))
        pose = {'position': [x, 1.0, z], 'rotation': [0.0, yaw, 0.0]}
        trajectory.append({'step': i, 'image_path': f'/img_{i}.png', **pose})
        visible, ego_edges = {}, []
        for oid, frames in shown_at.items():
            if i in frames:
                category, _region, position = landmarks[oid]
                visible[oid] = detection(category, position)
                ego_edges.append(ego(oid))
        steps.append(
            {
                'step': i,
                'image_path': f'/img_{i}.png',
                'action': 'move_ahead',
                'degrees': 45,
                'agent': pose,
                'visible_objects': visible,
                'non_visible_objects': {},
                'edges_egocentric': ego_edges,
                'edges_allocentric': [],
                'edges_object_frame': [],
                'edges_inferred': [],
            }
        )
    layout = {
        'landmarks': [
            {
                'landmark_id': oid,
                'obj-type': category,
                'region_id': region,
                'position': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
            }
            for oid, (category, region, pos) in landmarks.items()
        ],
        'passages': [
            {
                'passage_id': 'door|1',
                'passage_type': 'door',
                'from_region': 'roomA',
                'to_region': 'roomB',
                'position': {'x': 2.25, 'y': 1.0, 'z': 1.5},
            }
        ],
    }
    return {
        'episode_id': 'two_room',
        'scene_id': 'two_room',
        'environment': 'ai2thor',
        'episode_meta': {
            'camera': {'width': 396, 'height': 224, 'fov_vertical_deg': 59},
            'agent': {'rotation_deg': 45, 'movement_constant': g},
        },
        'nav_graph': nav_graph,
        'agent_trajectory': trajectory,
        'steps': steps,
        'world_layout': layout,
        'passage_state': [
            {
                'passage_id': 'door|1',
                'is_open': True,
                'timestep': door_step,
                'from_region': 'roomA',
                'to_region': 'roomB',
            }
        ],
        'agent_actions': [
            {'step': i, 'action': 'move_ahead', 'degrees': 45} for i in range(len(walk))
        ],
    }


@pytest.fixture
def two_room_episode():
    return _two_room_episode()


@pytest.fixture
def straight_corridor_episode():
    """Single straight run: no detour and no turns, so route must stay unsupported."""
    g = 0.25
    return _corridor_episode(
        [(0.0, k * g) for k in range(25)], grid=g, scene_id='straight_corridor'
    )


@pytest.fixture
def delayed_episode(tiny_episode):
    """Tiny episode extended so SWM/SU exercise multi-step delay with translation."""
    episode = deepcopy(tiny_episode)
    # Ensure every step has a distinct floor position (tiny starts rotate-only).
    for i, pose in enumerate(episode.get('agent_trajectory') or []):
        new_pos = (0.25 * i, 1.0, 0.0)
        pose['position'] = new_pos
        step = next(
            (s for s in episode['steps'] if int(s['step']) == int(pose['step'])),
            None,
        )
        if step is not None:
            step.setdefault('agent', {})['position'] = new_pos
            step['agent']['rotation'] = pose.get('rotation')

    last_pos = list(episode['agent_trajectory'][-1]['position'])
    base = int(episode['steps'][-1]['step'])
    for i, step_idx in enumerate((base + 1, base + 2), start=1):
        step = deepcopy(episode['steps'][-1])
        step['step'] = step_idx
        step['image_path'] = f'/img_{step_idx}.png'
        step['action'] = 'MoveAhead'
        new_pos = (float(last_pos[0]) + 0.25 * i, float(last_pos[1]), float(last_pos[2]))
        step['agent'] = {
            **(step.get('agent') or {}),
            'position': new_pos,
            'rotation': (step.get('agent') or {}).get('rotation')
            or episode['agent_trajectory'][-1]['rotation'],
        }
        episode['steps'].append(step)

        pose = deepcopy(episode['agent_trajectory'][-1])
        pose.update(step=step_idx, image_path=step['image_path'], position=new_pos)
        episode['agent_trajectory'].append(pose)
        episode['agent_actions'].append(
            {'step': step_idx, 'action': 'MoveAhead', 'degrees': None}
        )
    return episode


def test_egocentric_item_has_answer_source(tiny_episode):
    items = draft_items_for_episode(
        tiny_episode,
        constructs=['egocentric_encoding'],
        max_per_construct=1,
        styles=('concise',),
    )
    assert items
    item = items[0]
    assert item['status'] == 'ok'
    assert item['construct'] == 'egocentric_encoding'
    assert item['answer'] in item['options']
    assert item['answer_source']
    assert 'equal-wedge' in item['answer_source'][0]
    from cm_benchmark.generation.geometry import ego_label_from_world_pose, agent_pose_at_step

    step_idx = item['query_step']
    oid = item['queried_object_id']
    ag_pos, ag_rot = agent_pose_at_step(tiny_episode, step_idx)
    obj_pos = tiny_episode['steps'][step_idx]['visible_objects'][oid]['position']
    assert item['options'][item['answer']] == ego_label_from_world_pose(
        ag_pos,
        ag_rot,
        obj_pos,
        tiny_episode,
        ahead_half_width=20.0,
    )


def test_concise_verbose_pair_share_answer(tiny_episode):
    items = draft_items_for_episode(
        tiny_episode,
        constructs=['egocentric_encoding'],
        max_per_construct=1,
        styles=('concise', 'verbose'),
    )
    assert len(items) == 2
    a, b = items
    assert {a['question_style'], b['question_style']} == {'concise', 'verbose'}
    assert a['answer'] == b['answer']
    assert a['answer_source'] == b['answer_source']
    assert a['paired_item_id'] == b['item_id'] or b['paired_item_id'] == a['item_id']
    verbose = a if a['question_style'] == 'verbose' else b
    answer_text = verbose['options'][verbose['answer']]
    assert answer_text.lower() not in verbose['question'].lower()
    assert len(verbose['question']) > len(
        (a if a['question_style'] == 'concise' else b)['question']
    )


def test_invisible_displacement_swap_mode(folder_episode):
    facts = plan_episode(
        folder_episode, constructs=['invisible_displacement'], max_per_construct=4
    )
    swap = [
        f
        for f in facts
        if f.status == 'ok' and (f.extra or {}).get('template_mode') == 'swap'
    ]
    assert swap, f'expected swap facts, got {[f.reason for f in facts if f.status != "ok"]}'

    items = draft_items_for_episode(
        folder_episode,
        constructs=['invisible_displacement'],
        max_per_construct=4,
        styles=('concise',),
    )
    swap_items = [
        i
        for i in items
        if i.get('status') == 'ok'
        and (
            'used to be' in i['question']
            or 'previous location' in i['question']
        )
    ]
    assert swap_items
    for item in swap_items:
        assert item['frame_of_reference'] == 'egocentric'
        assert item.get('displacement_event', {}).get('moved_via') == 'swap'
        assert 'moved onto' not in item['question']
        assert any(
            pid in (item.get('displacement_event') or {}).get('swap_partner_id', '')
            for pid in ('Cup|1', 'Plate|1')
        )


def test_invisible_displacement_dual_modes(folder_episode):
    facts = plan_episode(
        folder_episode, constructs=['invisible_displacement'], max_per_construct=4
    )
    assert facts
    ok = [f for f in facts if f.status == 'ok']
    assert ok, f'expected ok ID facts, got {[f.reason for f in facts]}'
    modes = {(f.extra or {}).get('template_mode') for f in ok}
    assert modes <= {'recall_direction', 'swap'}
    assert 'recall_direction' in modes or 'swap' in modes

    items = draft_items_for_episode(
        folder_episode,
        constructs=['invisible_displacement'],
        max_per_construct=4,
        styles=('concise',),
    )
    ok_items = [i for i in items if i.get('status') == 'ok']
    assert ok_items
    for item in ok_items:
        assert item.get('displacement_event')
        assert item['encoding_step'] is not None
        assert item['query_step'] is not None
        assert item['frame_of_reference'] == 'egocentric'
        labels = list(item['options'].values())
        assert len(labels) == len(set(labels))
        assert any('you' in lab for lab in labels)
        # Destination cue required: landmark name or partner object
        q = item['question']
        assert 'moved onto' in q or 'moved to' in q or 'used to be' in q or 'previous location' in q
        assert not (
            q.strip().endswith('now?')
            and 'moved' not in q
            and 'previous location' not in q
        )


def test_invisible_displacement_rejects_duplicate_ego_labels(folder_episode):
    """If candidate poses collapse to one ego label at every query step, skip."""
    episode = folder_episode
    # Force all candidate positions identical → duplicate ego labels
    for row in episode.get('displacement_candidates') or []:
        row['candidate_position'] = (0.2, 1.0, 0.5)
    for ev in episode.get('displacement_events') or []:
        ev['from_position'] = (0.2, 1.0, 0.5)
        ev['to_position'] = (0.2, 1.0, 0.5)
    facts = plan_episode(
        episode, constructs=['invisible_displacement'], max_per_construct=4
    )
    ok = [f for f in facts if f.status == 'ok']
    assert not ok
    assert any(f.status == 'unsupported' for f in facts)


def test_invisible_displacement_skips_colliding_distractors(folder_episode):
    """A distractor that shares the answer's ego label is dropped, not fatal."""
    facts = plan_episode(
        folder_episode, constructs=['invisible_displacement'], max_per_construct=4
    )
    ok = [f for f in facts if f.status == 'ok']
    assert ok
    for fact in ok:
        labels = fact.options_pool or []
        assert len(labels) == len(set(labels))
        assert len(labels) == 4
        assert fact.answer_label in labels
        assert (fact.extra or {}).get('template_mode') in ('recall_direction', 'swap')


def test_rotate_only_window_is_filtered(folder_episode):
    """No floor-plane translation → SWM/ID rejected; SU also needs net pose change."""
    from cm_benchmark.generation.planner import _has_real_move_between

    episode = folder_episode
    # Collapse all agent poses to one place and freeze heading (no net pose change).
    fixed = (0.0, 1.0, 0.0)
    fixed_rot = [0.0, 0.0, 0.0]
    for step in episode.get('steps') or []:
        agent = step.setdefault('agent', {})
        agent['position'] = fixed
        agent['rotation'] = list(fixed_rot)
    for pose in episode.get('agent_trajectory') or []:
        pose['position'] = fixed
        pose['rotation'] = list(fixed_rot)

    assert not _has_real_move_between(episode, 0, 2)

    for construct in (
        'spatial_working_memory',
        'spatial_updating',
        'invisible_displacement',
    ):
        facts = plan_episode(episode, constructs=[construct], max_per_construct=4)
        assert not any(f.status == 'ok' for f in facts), construct


def test_images_between_drops_stationary_intermediates(folder_episode):
    from cm_benchmark.generation.planner import _images_between

    episode = folder_episode
    # step 0 and 1 same position, step 2 translated (fixture)
    paths = _images_between(episode, 0, 2)
    assert paths[0].endswith('img_0.png')
    assert paths[-1].endswith('img_2.png')
    # Intermediate rotate-only frame at same (x,z) as step 0 is dropped.
    assert not any(p.endswith('img_1.png') for p in paths)


def test_perspective_taking_abc_landmarks(folder_episode):
    """Perspective taking uses relational A→B heading — no intrinsic front required."""
    facts = plan_episode(
        folder_episode, constructs=['perspective_taking'], max_per_construct=1
    )
    assert facts
    if facts[0].status == 'unsupported':
        reason = facts[0].reason or ''
        assert 'landmark' in reason or 'ABC' in reason or 'triple' in reason
        return
    fact = facts[0]
    assert fact.extra.get('A') and fact.extra.get('B') and fact.extra.get('C')
    assert fact.answer_label in {
        'ahead of you',
        'to your right',
        'behind you',
        'to your left',
    }
    q = _core_question(fact).lower()
    assert 'imagine standing' in q
    assert 'facing' in q


def test_perspective_taking_tiny_may_be_ok_or_unsupported(tiny_episode):
    items = draft_items_for_episode(
        tiny_episode,
        constructs=['perspective_taking'],
        styles=('concise',),
    )
    assert len(items) == 1
    assert items[0]['status'] in ('ok', 'unsupported')
    if items[0]['status'] == 'ok':
        assert 'Imagine standing' in items[0]['question']


def test_allocentric_encoding_unsupported_without_facing(tiny_episode):
    items = draft_items_for_episode(
        tiny_episode,
        constructs=['allocentric_encoding'],
        styles=('concise',),
    )
    assert len(items) == 1
    assert items[0]['status'] == 'unsupported'
    assert 'facing' in items[0]['distractor_rationale'].get('reason', '')


def test_swm_encoding_uses_last_distinguishable_not_weak_last_seen():
    """SWM encode walks back past a weak FOV scrape to a clear earlier sighting."""
    from cm_benchmark.generation.planner import (
        _last_distinguishable_sighting,
        plan_spatial_working_memory,
    )

    def _vis(area, side, pix, pos=(0.0, 1.0, 1.0)):
        return {
            'category': 'CounterTop',
            'position': list(pos),
            'bbox_area': area,
            'min_side': side,
            'visible_pixels': pix,
        }

    # step 0: clear sighting; step 2: weak scrape still in visible_objects;
    # step 4: object only in non_visible with last_seen=2
    episode = {
        'steps': [
            {
                'step': 0,
                'image_path': '/img_0.png',
                'agent': {'position': [0, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {'CounterTop|1': _vis(2000, 40, 500)},
                'non_visible_objects': {},
                'edges_egocentric': [
                    {
                        'source': 'agent',
                        'target': 'CounterTop|1',
                        'angle_relation': ['', '', 'front'],
                    }
                ],
            },
            {
                'step': 1,
                'image_path': '/img_1.png',
                'agent': {'position': [0.5, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {},
                'non_visible_objects': {
                    'CounterTop|1': {
                        'category': 'CounterTop',
                        'last_seen_step': 0,
                        'position': [0.0, 1.0, 1.0],
                    }
                },
                'edges_egocentric': [],
            },
            {
                'step': 2,
                'image_path': '/img_2.png',
                'agent': {'position': [1.0, 1, 0], 'rotation': [0, 0, 0]},
                # Weak scrape: in FOV list but fails QUERY_FOV gates
                'visible_objects': {'CounterTop|1': _vis(50, 5, 20)},
                'non_visible_objects': {},
                'edges_egocentric': [
                    {
                        'source': 'agent',
                        'target': 'CounterTop|1',
                        'angle_relation': ['right', '', ''],
                    }
                ],
            },
            {
                'step': 3,
                'image_path': '/img_3.png',
                'agent': {'position': [1.5, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {},
                'non_visible_objects': {
                    'CounterTop|1': {
                        'category': 'CounterTop',
                        'last_seen_step': 2,
                        'position': [0.0, 1.0, 1.0],
                    }
                },
                'edges_egocentric': [],
            },
            {
                'step': 4,
                'image_path': '/img_4.png',
                'agent': {'position': [2.0, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {},
                'non_visible_objects': {
                    'CounterTop|1': {
                        'category': 'CounterTop',
                        'last_seen_step': 2,
                        'position': [0.0, 1.0, 1.0],
                    }
                },
                'edges_egocentric': [],
            },
        ],
        'agent_trajectory': [
            {'step': i, 'position': [0.5 * i, 1, 0], 'rotation': [0, 0, 0], 'image_path': f'/img_{i}.png'}
            for i in range(5)
        ],
        'agent_actions': [
            {'step': i, 'action': 'MoveAhead', 'degrees': None} for i in range(1, 5)
        ],
        'displacement_events': [],
    }

    assert _last_distinguishable_sighting(episode, 'CounterTop|1', 4) == 0
    facts = plan_spatial_working_memory(episode, max_items=5, min_delay=2)
    ok = [f for f in facts if f.status == 'ok' and f.queried_object_id == 'CounterTop|1']
    assert ok, 'expected SWM fact using clear encode, not weak last_seen=2'
    assert all(f.encoding_step == 0 for f in ok)
    assert all(f.encoding_step != 2 for f in ok)
    assert any(f.query_step >= 3 for f in ok)


def test_id_encode_requires_distinguishable_rejects_track_only():
    """Track FOV without visible distinguishable metrics must not encode ID."""
    from cm_benchmark.generation.planner import (
        _ID_ENCODE_FOV,
        _last_distinguishable_sighting,
        plan_invisible_displacement,
    )

    counter = {
        'category': 'CounterTop',
        'position': [2.0, 0.9, 1.0],
        'bbox_area': 5000.0,
        'min_side': 60.0,
        'visible_pixels': 4000.0,
    }
    episode = {
        'steps': [
            {
                'step': 0,
                'image_path': '/img_0.png',
                'agent': {'position': [0, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {},
                'edges_egocentric': [],
            },
            {
                'step': 1,
                'image_path': '/img_1.png',
                'agent': {'position': [1, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {'CounterTop|1': counter},
                'edges_egocentric': [
                    {
                        'source': 'agent',
                        'target': 'CounterTop|1',
                        'angle_relation': ['', '', 'front'],
                    }
                ],
            },
            {
                'step': 2,
                'image_path': '/img_2.png',
                'agent': {'position': [2, 1, 0], 'rotation': [0, 90, 0]},
                'visible_objects': {'CounterTop|1': counter},
                'edges_egocentric': [
                    {
                        'source': 'agent',
                        'target': 'CounterTop|1',
                        'angle_relation': ['right', '', ''],
                    }
                ],
            },
        ],
        'agent_trajectory': [
            {'step': 0, 'position': [0, 1, 0], 'rotation': [0, 0, 0], 'image_path': '/img_0.png'},
            {'step': 1, 'position': [1, 1, 0], 'rotation': [0, 0, 0], 'image_path': '/img_1.png'},
            {'step': 2, 'position': [2, 1, 0], 'rotation': [0, 90, 0], 'image_path': '/img_2.png'},
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
                    },
                    {
                        'step': 2,
                        'position': [3.0, 0.9, 0.5],
                        'visible': False,
                        'in_camera_fov': False,
                    },
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
    assert _last_distinguishable_sighting(episode, 'Cup|1', 2, **_ID_ENCODE_FOV) is None
    facts = plan_invisible_displacement(episode, max_items=2)
    assert not any(f.status == 'ok' for f in facts)


def test_id_encode_soft_distinguishable_and_pads_to_four_options():
    """Soft-FOV distinguishable encode + invisible query; options padded to 4."""
    from cm_benchmark.generation.planner import (
        _pad_id_ego_options,
        plan_invisible_displacement,
    )

    cup = {
        'category': 'Cup',
        'position': [0.5, 0.9, 0.5],
        'bbox_area': 200.0,
        'min_side': 12.0,
        'visible_pixels': 80.0,
    }
    counter = {
        'category': 'CounterTop',
        'position': [2.0, 0.9, 1.0],
        'bbox_area': 5000.0,
        'min_side': 60.0,
        'visible_pixels': 4000.0,
    }
    episode = {
        'steps': [
            {
                'step': 0,
                'image_path': '/img_0.png',
                'agent': {'position': [0, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {'Cup|1': cup, 'CounterTop|1': counter},
                'edges_egocentric': [
                    {
                        'source': 'agent',
                        'target': 'Cup|1',
                        'angle_relation': ['', '', 'front'],
                    },
                    {
                        'source': 'agent',
                        'target': 'CounterTop|1',
                        'angle_relation': ['', '', 'front'],
                    },
                ],
            },
            {
                'step': 1,
                'image_path': '/img_1.png',
                'agent': {'position': [1, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {'CounterTop|1': counter},
                'edges_egocentric': [
                    {
                        'source': 'agent',
                        'target': 'CounterTop|1',
                        'angle_relation': ['', '', 'front'],
                    }
                ],
            },
            {
                'step': 2,
                'image_path': '/img_2.png',
                'agent': {'position': [2, 1, 0], 'rotation': [0, 90, 0]},
                'visible_objects': {'CounterTop|1': counter},
                'edges_egocentric': [
                    {
                        'source': 'agent',
                        'target': 'CounterTop|1',
                        'angle_relation': ['right', '', ''],
                    }
                ],
            },
        ],
        'agent_trajectory': [
            {'step': 0, 'position': [0, 1, 0], 'rotation': [0, 0, 0], 'image_path': '/img_0.png'},
            {'step': 1, 'position': [1, 1, 0], 'rotation': [0, 0, 0], 'image_path': '/img_1.png'},
            {'step': 2, 'position': [2, 1, 0], 'rotation': [0, 90, 0], 'image_path': '/img_2.png'},
        ],
        'agent_actions': [
            {'step': 0, 'action': 'Pass', 'degrees': None},
            {'step': 1, 'action': 'MoveAhead', 'degrees': None},
            {'step': 2, 'action': 'MoveAhead', 'degrees': None},
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
                    },
                    {
                        'step': 2,
                        'position': [3.0, 0.9, 0.5],
                        'visible': False,
                        'in_camera_fov': False,
                    },
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
    padded, seeds = _pad_id_ego_options(
        'to your right', ['to your right', 'behind you'], ['original_location']
    )
    assert len(padded) == 4
    assert 'to your right' in padded and 'behind you' in padded
    assert 'opposite_direction' in seeds

    facts = plan_invisible_displacement(episode, max_items=2)
    ok = [f for f in facts if f.status == 'ok']
    assert ok, f'expected ID ok, got {[(f.status, f.reason) for f in facts]}'
    assert ok[0].encoding_step == 0
    assert ok[0].queried_object_id == 'Cup|1'
    assert len(ok[0].options_pool) == 4
    assert len(set(ok[0].options_pool)) == 4


def test_survey_rejects_when_agent_near_source_landmark():
    """Allocentric survey collapses if agent stands at the source landmark."""
    from cm_benchmark.generation.planner import (
        SURVEY_MIN_AGENT_SOURCE_DIST_M,
        _agent_near_landmark,
    )

    episode = {
        'steps': [
            {
                'step': 0,
                'agent': {'position': [1.0, 1.0, 1.0], 'rotation': [0, 0, 0]},
            }
        ],
        'agent_trajectory': [
            {'step': 0, 'position': [1.0, 1.0, 1.0], 'rotation': [0, 0, 0]}
        ],
    }
    near = {'x': 1.2, 'y': 0.5, 'z': 1.1}  # ~0.22 m
    far = {'x': 5.0, 'y': 0.5, 'z': 5.0}
    assert _agent_near_landmark(episode, near, [0])
    assert not _agent_near_landmark(episode, far, [0])
    assert not _agent_near_landmark(episode, near, [0], min_dist_m=0.1)
    assert SURVEY_MIN_AGENT_SOURCE_DIST_M == 1.0


def test_route_min_hop_count_is_two_not_r2r():
    """Route hop minimum is 2 (one edge), not R2R's 4–6."""
    from cm_benchmark.generation.planner import ROUTE_MIN_HOP_COUNT

    assert ROUTE_MIN_HOP_COUNT == 2


def test_route_reference_is_smoothed_and_scores_valid_success(u_corridor_episode):
    """A cornered scene yields a route whose stored reference is a valid success.

    The corridor is three straight runs, so the reference must read as long
    ``move_ahead`` runs with two 90° corners — not one rotation per lattice hop.
    """
    from cm_benchmark.generation.planner import (
        ROUTE_MIN_TURN_COUNT,
        _load_nav_graph_or_none,
        _rotation_deg,
        plan_route_knowledge,
    )
    from cm_benchmark.generation.nav_graph import score_route_action_sequence

    facts = [f for f in plan_route_knowledge(u_corridor_episode, max_items=2)
             if f.status == 'ok']
    assert facts, 'a U-shaped corridor must support route_knowledge'
    fact = facts[0]
    extra = fact.extra
    assert extra['turn_count'] >= ROUTE_MIN_TURN_COUNT
    # Three straight legs, two corners of two 45° increments each.
    assert extra['turn_count'] == 2
    actions = extra['action_sequence']
    assert actions.count('move_ahead') >= 24
    assert actions.count('rotate_right') + actions.count('rotate_left') == 4

    graph = _load_nav_graph_or_none(u_corridor_episode)
    score = score_route_action_sequence(
        graph,
        extra['source_node'],
        extra['goal_node'],
        actions,
        [tuple(e) for e in extra['traversed_edges']],
        start_heading_deg=extra['start_heading_deg'],
        rotation_deg=_rotation_deg(u_corridor_episode),
    )
    assert score['outcome'] == 'valid_success'
    assert score['illegal_edges'] == []


def test_route_rejects_straight_and_turnless_pairs(straight_corridor_episode):
    """Straight-line pairs are rejected by name, so small scenes stay honest."""
    from cm_benchmark.generation.planner import (
        ROUTE_MIN_GEODESIC_EUCLIDEAN_RATIO,
        ROUTE_MIN_TURN_COUNT,
        plan_route_knowledge,
    )

    facts = plan_route_knowledge(straight_corridor_episode, max_items=2)
    assert all(f.status == 'unsupported' for f in facts)
    reason = facts[0].reason or ''
    assert (
        f'geodesic_euclidean_ratio_lt_{ROUTE_MIN_GEODESIC_EUCLIDEAN_RATIO:g}' in reason
        or f'turns_lt_{ROUTE_MIN_TURN_COUNT}' in reason
    )


def test_route_emits_hardest_pairs_first(u_corridor_episode):
    """Emitted pairs are ranked by difficulty, not by first-in-walk-order."""
    from cm_benchmark.generation.planner import plan_route_knowledge

    facts = [f for f in plan_route_knowledge(u_corridor_episode, max_items=5)
             if f.status == 'ok']
    ranks = [(f.extra['turn_count'], f.extra['geodesic_m']) for f in facts]
    assert ranks == sorted(ranks, reverse=True)


def test_survey_emits_hardest_pairs_first(two_room_episode):
    """Survey ranks by difficulty like route, instead of returning enumeration order.

    ``ids`` is salience-ordered, so the first pair the loop reaches can be the
    easiest in the scene; the fixture is built so that is true.
    """
    from cm_benchmark.generation.planner import (
        ROUTE_LANDMARK_SNAP_M,
        _build_landmark_node_map,
        _load_nav_graph_or_none,
        plan_survey_based_route_planning,
        select_landmark_candidates,
    )

    every = [
        f
        for f in plan_survey_based_route_planning(two_room_episode, max_items=99)
        if f.status == 'ok'
    ]
    assert len(every) >= 3, 'fixture must offer several survey pairs to rank'
    ranks = [
        (f.extra['turn_count'], f.extra['geodesic_m'], f.extra['hop_count'])
        for f in every
    ]
    assert ranks == sorted(ranks, reverse=True)

    graph = _load_nav_graph_or_none(two_room_episode)
    landmarks = select_landmark_candidates(two_room_episode, unique_category=False)
    id_to_node, _names, _meta = _build_landmark_node_map(
        graph, landmarks, max_distance_m=ROUTE_LANDMARK_SNAP_M
    )
    ids = [lm['obj_id'] for lm in landmarks if lm['obj_id'] in id_to_node]
    position = {oid.split('|')[0]: i for i, oid in enumerate(ids)}

    def enumeration_key(fact):
        a = position[fact.extra['source'].split()[0]]
        b = position[fact.extra['goal'].split()[0]]
        return (min(a, b), max(a, b))

    first_in_enumeration = sorted(every, key=enumeration_key)[0]
    top = plan_survey_based_route_planning(two_room_episode, max_items=2)
    assert [f.extra['turn_count'] for f in top] == [r[0] for r in ranks[:2]]
    # The old early-return would have emitted this one; ranking must not.
    assert first_in_enumeration.extra['turn_count'] < ranks[0][0]
    assert first_in_enumeration.answer_label not in [f.answer_label for f in top]


def test_survey_reference_scores_valid_success(two_room_episode):
    """Survey stores a viewed-edge path, so the scorer must accept it."""
    from cm_benchmark.generation.planner import (
        _load_nav_graph_or_none,
        _rotation_deg,
        plan_survey_based_route_planning,
    )
    from cm_benchmark.generation.nav_graph import score_survey_action_sequence

    graph = _load_nav_graph_or_none(two_room_episode)
    facts = [
        f
        for f in plan_survey_based_route_planning(two_room_episode, max_items=99)
        if f.status == 'ok'
    ]
    assert facts
    for fact in facts:
        extra = fact.extra
        score = score_survey_action_sequence(
            graph,
            extra['source_node'],
            extra['goal_node'],
            extra['action_sequence'],
            [tuple(e) for e in extra['viewed_edges']],
            [tuple(e) for e in extra['traversed_edges']],
            start_heading_deg=extra['start_heading_deg'],
            rotation_deg=_rotation_deg(two_room_episode),
        )
        assert score['outcome'] == 'valid_success'
        assert score['illegal_edges'] == []
        assert score['novel_edges'], 'survey must leave the traversed edge set'


def test_scene_geodesic_floor_is_per_scene_and_adds_to_absolute_band():
    """Scene-relative floor: a percentile of this scene's own pair distribution."""
    import networkx as nx
    from cm_benchmark.generation.planner import (
        MIN_PAIR_GEODESIC_M,
        SCENE_PAIR_GEODESIC_PERCENTILE,
        _class4_pair_reject_reason,
        calibrate_scene_geodesic_floor_m,
    )

    assert SCENE_PAIR_GEODESIC_PERCENTILE == 0.5

    # Chain of five nodes 2 m apart: sorted pair geodesics are
    # [2, 2, 2, 2, 4, 4, 4, 6, 6, 8]. Index is int(p * (n - 1)) = 4, the same
    # nearest-rank convention calibrate_view_radius_m uses.
    g = nx.Graph()
    for i in range(5):
        g.add_node(f'n{i}', pos=(2.0 * i, 1.0, 0.0))
    for i in range(4):
        g.add_edge(f'n{i}', f'n{i + 1}', weight=2.0)
    nodes = [f'n{i}' for i in range(5)]

    floor = calibrate_scene_geodesic_floor_m(g, nodes)
    assert floor == 4.0
    assert floor > MIN_PAIR_GEODESIC_M

    near = ({'x': 0.0, 'z': 0.0}, {'x': 2.0, 'z': 0.0}, 'n0', 'n1')  # geodesic 2 m
    far = ({'x': 0.0, 'z': 0.0}, {'x': 6.0, 'z': 0.0}, 'n0', 'n3')  # geodesic 6 m
    # Both clear the absolute band; only the far pair clears the scene floor.
    assert _class4_pair_reject_reason(g, *near) is None
    assert (
        _class4_pair_reject_reason(g, *near, min_scene_geodesic_m=floor)
        == 'geodesic_below_scene_median'
    )
    assert _class4_pair_reject_reason(g, *far, min_scene_geodesic_m=floor) is None

    # Undefined distribution leaves only the absolute band in force.
    assert calibrate_scene_geodesic_floor_m(g, ['n0']) is None
    assert calibrate_scene_geodesic_floor_m(g, []) is None


def test_scene_geodesic_floor_is_stricter_than_the_absolute_floor(u_corridor_episode):
    """On a real scene the scene-relative floor binds where the 1 m band does not."""
    from cm_benchmark.generation.planner import (
        MIN_PAIR_GEODESIC_M,
        ROUTE_LANDMARK_SNAP_M,
        _build_landmark_node_map,
        _load_nav_graph_or_none,
        calibrate_scene_geodesic_floor_m,
        select_landmark_candidates,
    )

    graph = _load_nav_graph_or_none(u_corridor_episode)
    landmarks = select_landmark_candidates(u_corridor_episode, unique_category=False)
    id_to_node, _names, _meta = _build_landmark_node_map(
        graph, landmarks, max_distance_m=ROUTE_LANDMARK_SNAP_M
    )
    floor = calibrate_scene_geodesic_floor_m(graph, list(id_to_node.values()))
    assert floor is not None
    assert floor >= MIN_PAIR_GEODESIC_M


def test_class4_pair_gate_is_geodesic_metres_not_percentile():
    """Class-4 pair length is 1–30 m geodesic (OVON/GOAT/HSSD) plus a ratio gate."""
    from cm_benchmark.generation.planner import (
        MIN_PAIR_GEODESIC_M,
        MAX_PAIR_GEODESIC_M,
        ROUTE_MIN_GEODESIC_EUCLIDEAN_RATIO,
        SURVEY_MIN_GEODESIC_EUCLIDEAN_RATIO,
        _class4_pair_reject_reason,
    )
    import networkx as nx

    assert MIN_PAIR_GEODESIC_M == 1.0
    assert MAX_PAIR_GEODESIC_M == 30.0
    assert SURVEY_MIN_GEODESIC_EUCLIDEAN_RATIO == 1.05
    # Habitat's PointNav generator applies geo/eucl >= 1.1 to every episode.
    assert ROUTE_MIN_GEODESIC_EUCLIDEAN_RATIO == 1.1

    g = nx.Graph()
    g.add_node('n0', pos=(0.0, 1.0, 0.0))
    g.add_node('n1', pos=(4.0, 1.0, 0.0))
    g.add_edge('n0', 'n1', weight=4.0)
    # Straight corridor, geo=4 m, ratio=1.0: length passes, both ratio gates reject.
    assert (
        _class4_pair_reject_reason(
            g, {'x': 0.0, 'z': 0.0}, {'x': 4.0, 'z': 0.0}, 'n0', 'n1', min_ratio=None
        )
        is None
    )
    for ratio in (SURVEY_MIN_GEODESIC_EUCLIDEAN_RATIO, ROUTE_MIN_GEODESIC_EUCLIDEAN_RATIO):
        assert (
            _class4_pair_reject_reason(
                g,
                {'x': 0.0, 'z': 0.0},
                {'x': 4.0, 'z': 0.0},
                'n0',
                'n1',
                min_ratio=ratio,
            )
            == f'geodesic_euclidean_ratio_lt_{ratio:g}'
        )
    g.add_node('n_close', pos=(0.4, 1.0, 0.0))
    g.add_edge('n0', 'n_close', weight=0.4)
    assert (
        _class4_pair_reject_reason(
            g,
            {'x': 0.0, 'z': 0.0},
            {'x': 0.4, 'z': 0.0},
            'n0',
            'n_close',
            min_ratio=None,
        )
        == 'geodesic_lt_1m'
    )
    g.add_node('n2', pos=(0.0, 1.0, 3.0))
    g.add_edge('n1', 'n2', weight=5.0)
    # Detour 4+5=9 over eucl 3 → ratio 3.0; both keep.
    assert (
        _class4_pair_reject_reason(
            g,
            {'x': 0.0, 'z': 0.0},
            {'x': 0.0, 'z': 3.0},
            'n0',
            'n2',
            min_ratio=SURVEY_MIN_GEODESIC_EUCLIDEAN_RATIO,
        )
        is None
    )


def test_swm_reserves_recall_count_slots():
    """Count/load mode must not be starved by relation filling max_items first."""
    from cm_benchmark.generation.planner import plan_spatial_working_memory

    # Minimal episode: two Chairs seen early, then navigate so delay+move hold.
    def step(i, visible, non_vis=None, edges=None, x=None):
        return {
            'step': i,
            'image_path': f'/img_{i}.png',
            'agent': {
                'position': [float(x if x is not None else i), 1.0, 0.0],
                'rotation': [0, 0, 0],
            },
            'visible_objects': visible,
            'non_visible_objects': non_vis or {},
            'edges_egocentric': edges or [],
        }

    chair = lambda oid, z: {
        'category': 'Chair',
        'position': [0.0, 1.0, float(z)],
        'bbox_area': 2000.0,
        'min_side': 40.0,
        'visible_pixels': 500.0,
    }
    episode = {
        'steps': [
            step(
                0,
                {'Chair|1': chair('Chair|1', 1.0), 'Chair|2': chair('Chair|2', 2.0)},
                edges=[
                    {
                        'source': 'agent',
                        'target': 'Chair|1',
                        'angle_relation': ['', '', 'front'],
                    },
                    {
                        'source': 'agent',
                        'target': 'Chair|2',
                        'angle_relation': ['right', '', ''],
                    },
                ],
            ),
            step(1, {}, non_vis={
                'Chair|1': {'category': 'Chair', 'last_seen_step': 0, 'position': [0, 1, 1]},
                'Chair|2': {'category': 'Chair', 'last_seen_step': 0, 'position': [0, 1, 2]},
            }),
            step(2, {}, non_vis={
                'Chair|1': {'category': 'Chair', 'last_seen_step': 0, 'position': [0, 1, 1]},
                'Chair|2': {'category': 'Chair', 'last_seen_step': 0, 'position': [0, 1, 2]},
            }),
        ],
        'agent_trajectory': [
            {
                'step': i,
                'position': [float(i), 1.0, 0.0],
                'rotation': [0, 0, 0],
                'image_path': f'/img_{i}.png',
            }
            for i in range(3)
        ],
        'agent_actions': [
            {'step': i, 'action': 'MoveAhead', 'degrees': None} for i in range(1, 3)
        ],
        'displacement_events': [],
    }
    facts = plan_spatial_working_memory(episode, max_items=3, min_delay=2)
    modes = {(f.extra or {}).get('template_mode') for f in facts if f.status == 'ok'}
    assert 'recall_count' in modes, f'expected recall_count among {facts}'
    count = next(f for f in facts if (f.extra or {}).get('template_mode') == 'recall_count')
    assert count.answer_label == '2'
    assert count.extra.get('load_n_objects') == 2


def test_merge_role_images_dedupes_with_combined_labels():
    from cm_benchmark.generation.planner import merge_role_images

    paths, roles = merge_role_images(
        [
            (['/a.png'], 'A · stand (Table)'),
            (['/a.png'], 'B · face (Sofa)'),
            (['/c.png'], 'C · locate (Fridge)'),
        ]
    )
    assert paths == ['/a.png', '/c.png']
    assert roles[0] == 'A · stand (Table) + B · face (Sofa)'
    assert roles[1] == 'C · locate (Fridge)'


def test_draft_item_persists_image_roles_and_context(folder_episode):
    items = draft_items_for_episode(
        folder_episode,
        constructs=['perspective_taking'],
        max_per_construct=1,
        styles=('concise',),
    )
    ok = [i for i in items if i.get('status') == 'ok']
    if not ok:
        pytest.skip('no perspective_taking on tiny fixture')
    item = ok[0]
    assert item.get('image_roles')
    assert len(item['image_roles']) == len(item.get('image_paths') or [])
    ctx = item.get('context') or {}
    assert ctx.get('A') and ctx.get('B') and ctx.get('C')


def test_referring_display_name_requires_distinguishable_and_unique():
    from cm_benchmark.generation.planner import _referring_display_name

    def _vis(area, side, pix, pos):
        return {
            'category': 'Chair',
            'position': list(pos),
            'bbox_area': area,
            'min_side': side,
            'visible_pixels': pix,
        }

    lm = {
        'category': 'Table',
        'position': [0.0, 1.0, 3.0],
        'bbox_area': 5000.0,
        'min_side': 50.0,
        'visible_pixels': 2000.0,
    }
    episode = {
        'steps': [
            {
                'step': 0,
                'agent': {'position': [0, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {
                    'Chair|1': _vis(2000, 40, 500, (0.0, 1.0, 1.0)),
                    'Chair|2': _vis(2000, 40, 500, (2.0, 1.0, 1.0)),
                    'Table|1': lm,
                },
                'edges_egocentric': [
                    {'source': 'agent', 'target': 'Chair|1', 'angle_relation': ['', '', 'front']},
                    {'source': 'agent', 'target': 'Chair|2', 'angle_relation': ['right', '', '']},
                    {'source': 'agent', 'target': 'Table|1', 'angle_relation': ['', '', 'front']},
                ],
            }
        ]
    }
    # Unique landmark category → bare name
    assert _referring_display_name(episode, 0, 'Table|1') == 'Table'
    # Duplicate chairs with clear proximity margin to Table for Chair|1
    name = _referring_display_name(episode, 0, 'Chair|1')
    assert name is not None
    assert name.startswith('Chair')
    assert 'Table' in name


def test_perspective_taking_rejects_weak_landmarks():
    """PT must not mention landmarks that fail QUERY_FOV distinguishability."""
    from cm_benchmark.generation.planner import plan_perspective_taking

    weak = {
        'category': 'Cup',
        'position': [1.0, 1.0, 1.0],
        'bbox_area': 50.0,
        'min_side': 5.0,
        'visible_pixels': 20.0,
    }
    strong = {
        'category': 'Sofa',
        'position': [2.0, 1.0, 2.0],
        'bbox_area': 5000.0,
        'min_side': 50.0,
        'visible_pixels': 2000.0,
    }
    episode = {
        'steps': [
            {
                'step': 0,
                'image_path': '/a.png',
                'agent': {'position': [0, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {
                    'Cup|1': weak,
                    'Sofa|1': strong,
                    'Fridge|1': {
                        **strong,
                        'category': 'Fridge',
                        'position': [3.0, 1.0, 0.0],
                    },
                },
                'edges_egocentric': [
                    {'source': 'agent', 'target': 'Cup|1', 'angle_relation': ['', '', 'front']},
                    {'source': 'agent', 'target': 'Sofa|1', 'angle_relation': ['right', '', '']},
                    {'source': 'agent', 'target': 'Fridge|1', 'angle_relation': ['left', '', '']},
                ],
            }
        ],
        'world_layout': {'landmarks': []},
        'agent_trajectory': [
            {'step': 0, 'position': [0, 1, 0], 'rotation': [0, 0, 0], 'image_path': '/a.png'}
        ],
    }
    facts = plan_perspective_taking(episode, max_items=2)
    # Weak Cup must never appear as A/B/C
    for f in facts:
        if f.status != 'ok':
            continue
        blob = ' '.join(
            str((f.extra or {}).get(k) or '') for k in ('A', 'B', 'C')
        )
        assert 'Cup' not in blob


def test_border_scrape_cup_rejected_for_egocentric():
    """Thin left-edge Cup strip must not become an egocentric query target."""
    from cm_benchmark.generation.constructs import (
        bbox_not_border_scrape,
        fov_metrics_ok,
    )
    from cm_benchmark.generation.planner import (
        _distinguishable_encoding_sighting,
        plan_egocentric_encoding,
    )

    cup = {
        'category': 'Cup',
        'position': [0.1, 0.9, 0.5],
        'bbox': [0, 177, 25, 223],
        'bbox_area': 1150.0,
        'min_side': 25.0,
        'visible_pixels': 664.0,
        'occupancy_ratio': 0.577,
    }
    # Absolute floors alone (old 800/24/200) would pass; border scrape must fail.
    assert fov_metrics_ok(
        cup, min_bbox_area=800, min_side=24, min_visible_pixels=200
    )
    assert not bbox_not_border_scrape(cup, (396, 224))
    assert not fov_metrics_ok(
        cup,
        min_bbox_area=800,
        min_side=24,
        min_visible_pixels=200,
        image_wh=(396, 224),
    )

    episode = {
        'episode_meta': {'camera': {'width': 396, 'height': 224}},
        'steps': [
            {
                'step': 0,
                'image_path': '/img_0.png',
                'agent': {'position': [0, 1, 0], 'rotation': [0, 0, 0]},
                'visible_objects': {'Cup|2|9': cup},
                'edges_egocentric': [
                    {
                        'source': 'agent',
                        'target': 'Cup|2|9',
                        'angle_relation': ['left', '', ''],
                    }
                ],
            }
        ],
    }
    assert not _distinguishable_encoding_sighting(episode, 0, 'Cup|2|9')
    facts = plan_egocentric_encoding(episode, max_items=3)
    assert all(f.queried_object_id != 'Cup|2|9' for f in facts)


def test_swm_question_states_explicit_delay_k(delayed_episode):
    facts = plan_episode(
        delayed_episode, constructs=['spatial_working_memory'], max_per_construct=2
    )
    relation = [f for f in facts if (f.extra or {}).get('template_mode') == 'recall_relation']
    assert relation, 'expected at least one recall_relation SWM fact'
    fact = relation[0]
    assert fact.status == 'ok'
    assert fact.queried_object_id == 'Cup|1'
    k = fact.extra['k']
    assert k == fact.query_step - fact.encoding_step
    assert k >= 2
    # One image per navigated pose in the window (stationary intermediates dropped).
    assert 2 <= len(fact.image_paths) <= k + 1
    from cm_benchmark.generation.planner import _has_real_move_between

    assert _has_real_move_between(
        delayed_episode, fact.encoding_step, fact.query_step
    )
    q = _core_question(fact)
    assert str(k) in q
    assert 'steps' in q.lower() or 'ago' in q.lower()


def test_swm_delay_range_is_configurable(delayed_episode):
    items = draft_items_for_episode(
        delayed_episode,
        constructs=['spatial_working_memory'],
        max_per_construct=1,
        swm_min_delay=3,
        swm_max_delay=3,
        styles=('concise',),
    )
    ok = [item for item in items if item.get('status') == 'ok']
    assert ok, 'expected an SWM item at the requested delay'
    item = ok[0]
    assert item['query_step'] - item['encoding_step'] == 3
    assert len(item['image_paths']) == 4
    assert len(item['agent_trajectory']) == 4
    assert len(item['agent_actions']) == 3


def test_spatial_updating_mentions_now(delayed_episode):
    items = draft_items_for_episode(
        delayed_episode,
        constructs=['spatial_updating'],
        max_per_construct=1,
        styles=('concise',),
    )
    if not items or items[0].get('status') == 'unsupported':
        pytest.skip('no spatial_updating fact on delayed episode')
    item = items[0]
    q = item['question'].lower()
    assert 'relative to you now' in q or 'now' in q
    assert len(item.get('image_paths') or []) >= 2
    assert item['query_step'] - item['encoding_step'] >= 2
    # Online sequential protocol: no bundled multi-image time-order cue.
    assert 'time order' not in q
    assert 'images are shown' not in q
    assert 'steps' in q or 'navigation' in q


def test_spatial_updating_delay_range_is_configurable(delayed_episode):
    items = draft_items_for_episode(
        delayed_episode,
        constructs=['spatial_updating'],
        max_per_construct=1,
        su_min_delay=3,
        su_max_delay=3,
        styles=('concise',),
    )
    ok = [item for item in items if item.get('status') == 'ok']
    assert ok, 'expected a spatial_updating item at the requested delay'
    item = ok[0]
    assert item['query_step'] - item['encoding_step'] == 3
    assert len(item['image_paths']) == 4
    assert len(item['agent_trajectory']) == 4
    assert len(item['agent_actions']) == 3


def test_route_knowledge_is_retrace_not_plan(folder_episode):
    """Class-4 route items retrace walked A→B as collected-format actions."""
    facts = plan_episode(
        folder_episode, constructs=['route_knowledge'], max_per_construct=2
    )
    assert facts
    if facts[0].status == 'unsupported':
        reason = facts[0].reason or ''
        assert (
            'nav_graph' in reason
            or 'landmark' in reason
            or 'trajectory' in reason
            or 'walk' in reason
        )
        return
    fact = facts[0]
    assert fact.extra.get('source')
    assert fact.extra.get('goal')
    assert fact.extra.get('path_nodes')
    assert fact.extra.get('answer_format') == 'action_sequence'
    assert fact.extra.get('scoring') == 'success_validity_efficiency'
    assert fact.extra.get('graph_scope') == 'traversed'
    assert fact.extra.get('traversed_edges') is not None
    assert fact.extra.get('action_sequence')
    assert 'move_ahead' in (fact.answer_label or '')
    q = _core_question(fact).lower()
    assert 'in order' in q
    assert 'move_ahead' not in q
    assert 'rotate_left' not in q
    assert f"from the {fact.extra['source']}".lower() in q
    assert f"to the {fact.extra['goal']}".lower() in q
    assert 'which of these' not in q


def test_survey_based_route_planning_unsupported_without_novel_path(folder_episode):
    facts = plan_episode(
        folder_episode, constructs=['survey_based_route_planning'], max_per_construct=1
    )
    assert facts
    # Tiny fixture walk may cover all landmark pairs; accept unsupported reasons.
    if facts[0].status == 'unsupported':
        reason = facts[0].reason or ''
        assert (
            'nav_graph' in reason
            or 'novel' in reason
            or 'landmark' in reason
            or 'untraversed' in reason
            or 'through_opening' in reason
            or 'through-opening' in reason
            or 'opening' in reason
        )
        return
    fact = facts[0]
    assert fact.extra.get('answer_format') == 'action_sequence'
    assert fact.extra.get('scoring') == 'success_validity_efficiency'
    assert fact.extra.get('graph_scope') == 'viewed'
    assert fact.extra.get('viewed_edges') is not None
    assert fact.extra.get('traversed_edges') is not None
    assert fact.extra.get('action_sequence')
    assert 'move_ahead' in (fact.answer_label or '')
    roles = fact.extra.get('image_roles') or []
    assert any('source sighted' in r for r in roles)
    assert any('goal sighted' in r for r in roles)


def test_object_type_skips_undefined_category():
    from cm_benchmark.generation.constructs import object_type_from_id

    assert (
        object_type_from_id(
            'ObjaScooter|4|5', {'ObjaScooter|4|5': {'category': 'Undefined'}}
        )
        == 'ObjaScooter'
    )
    assert (
        object_type_from_id('FloorLamp|4|2', {'FloorLamp|4|2': {'category': 'FloorLamp'}})
        == 'FloorLamp'
    )
    assert object_type_from_id('Cup|1') == 'Cup'


def test_verbose_preamble_does_not_leak_answer(tiny_episode):
    facts = plan_episode(tiny_episode, constructs=['egocentric_encoding'], max_per_construct=1)
    fact = facts[0]
    preamble = build_verbose_preamble(tiny_episode, fact)
    assert fact.answer_label.lower() not in preamble.lower()


def test_verbose_preamble_omits_spatial_relations(delayed_episode):
    """SWM/ID verbose may name co-visible types — never ego bearings/relations."""
    facts = plan_episode(
        delayed_episode, constructs=['spatial_working_memory'], max_per_construct=1
    )
    if not facts or facts[0].status != 'ok':
        pytest.skip('no SWM fact on delayed episode')
    fact = facts[0]
    preamble = build_verbose_preamble(delayed_episode, fact).lower()
    assert 'you can see' in preamble or 'also visible' in preamble
    assert '(to your' not in preamble
    assert 'ahead of you' not in preamble
    assert 'behind you' not in preamble
    assert 'to your left' not in preamble
    assert 'to your right' not in preamble


def test_verbose_scene_detail_scoped_to_taxonomy_exception(tiny_episode):
    """Class-1 verbose must not dump other object names (shared_rules default)."""
    facts = plan_episode(tiny_episode, constructs=['egocentric_encoding'], max_per_construct=1)
    fact = facts[0]
    if fact.status != 'ok':
        pytest.skip('no egocentric fact')
    step = tiny_episode['steps'][fact.query_step]
    other_types = [
        object_type_from_id(oid, {oid: odata}).lower()
        for oid, odata in (step.get('visible_objects') or {}).items()
        if oid != fact.queried_object_id
    ]
    preamble = build_verbose_preamble(tiny_episode, fact).lower()
    assert 'you can see' not in preamble
    assert 'also visible' not in preamble
    for typ in other_types:
        assert typ not in preamble, f'unexpected other-object name {typ!r} in class-1 verbose'


def test_core_question_covers_active_template_placeholders():
    """Guard against KeyError when formatting templates."""
    from cm_benchmark.generation.planner import PlannedFact

    from cm_benchmark.generation.constructs import template_count

    for construct, modes in (
        ('egocentric_encoding', [None]),
        ('invisible_displacement', ['recall_direction', 'swap']),
        ('spatial_updating', [None]),
        ('route_knowledge', [None]),
        ('survey_based_route_planning', [None]),
        ('perspective_taking', [None]),
        ('spatial_working_memory', ['recall_relation', 'recall_count']),
    ):
        for mode in modes:
            n_tmpl = max(1, template_count(construct, mode))
            for tidx in range(n_tmpl):
                tmpl = select_template(construct, template_mode=mode, index=tidx)
                assert isinstance(tmpl, str)
                fact = PlannedFact(
                    construct=construct,
                    status='ok',
                    query_step=5,
                    encoding_step=2,
                    queried_object_id='Cup|1',
                    reference_object_id='Table|1',
                    answer_label='to your left',
                    extra={
                        'object_type': 'Cup',
                        'object_category': 'Cup',
                        'reference_object': 'Table',
                        'source': 'Kitchen',
                        'goal': 'LivingRoom',
                        'A': 'Armchair',
                        'B': 'Sofa',
                        'C': 'Lamp',
                        'new_location': 'Shelf',
                        'other_object_type': 'Plate',
                        'condition': 'the door is closed',
                        'k': 3,
                        'template_mode': mode,
                        'template_index': tidx,
                        'disambiguator': '',
                    },
                )
                q = _core_question(fact)
                assert '{object' not in q
                assert '{k}' not in q
                assert '{disambiguator}' not in q
                assert '{new_location}' not in q
                assert '{other_object_type}' not in q
                assert '{A}' not in q and '{C}' not in q
                assert tmpl.split('{')[0] in q or 'Cup' in q or 'Kitchen' in q or 'Armchair' in q
                # Unique category → no referring phrase / no double spaces around type.
                if '{disambiguator}' in tmpl:
                    assert 'Cup  ' not in q
                    assert 'close to' not in q.lower()


def test_class4_templates_match_taxonomy_wording():
    from cm_benchmark.generation.constructs import pick_template_index, template_count

    route = select_template('route_knowledge')
    assert isinstance(route, str)
    assert '{source}' in route and '{goal}' in route
    assert 'move_ahead' not in route
    assert 'which of these' not in route.lower()

    assert template_count('survey_based_route_planning') == 2
    t0 = select_template('survey_based_route_planning', index=0)
    t1 = select_template('survey_based_route_planning', index=1)
    assert t0 != t1
    assert 'move_ahead' not in t0 and 'move_ahead' not in t1
    assert pick_template_index('survey_based_route_planning', None, 'a', 'b') == (
        pick_template_index('survey_based_route_planning', None, 'a', 'b')
    )


def test_class4_slide_panel_shows_source_goal_path_and_scoring():
    from cm_benchmark.generation.slide_copy import class4_slide_panel

    route = class4_slide_panel(
        'route_knowledge',
        {
            'source': 'Chair',
            'goal': 'Table',
            'path_nodes': ['n0', 'n1', 'n2', 'n3'],
            'action_sequence': ['move_ahead', 'rotate_left', 'move_ahead'],
            'hop_count': 3,
        },
    )
    assert 'Chair' in route['task'] and 'Table' in route['task']
    assert 'Retrace' in route['task']
    assert 'n0' not in route['path'] and 'n3' not in route['path']
    assert '↑' in route['path'] and '←' in route['path']
    assert 'bbox center' in route['actions']
    assert 'validity' in route['scoring']

    survey = class4_slide_panel(
        'survey_based_route_planning',
        {'source': 'Door', 'goal': 'Window', 'path_nodes': ['a', 'b']},
        answer='move_ahead → rotate_right → move_ahead',
    )
    assert 'Plan' in survey['task']
    assert 'Door' in survey['task'] and 'Window' in survey['task']
    assert 'viewed' in survey['graph']
    assert 'unwalked' in survey['scoring']
    assert '↑' in survey['path'] and '→' in survey['path']


def test_core_question_includes_disambiguator_only_when_set():
    from cm_benchmark.generation.planner import PlannedFact

    fact = PlannedFact(
        construct='egocentric_encoding',
        status='ok',
        query_step=0,
        encoding_step=0,
        queried_object_id='Chair|1',
        answer_label='ahead of you',
        extra={
            'object_type': 'Chair',
            'disambiguator': ' close to the Bread',
        },
    )
    q = _core_question(fact)
    assert 'the Chair close to the Bread relative to you' in q


def test_resolve_referring_disambiguator_unique_and_duplicate():
    from cm_benchmark.generation.constructs import resolve_referring_disambiguator

    def _vis(cat, pos, *, area=2000, side=40, pix=500):
        return {
            'category': cat,
            'position': pos,
            'bbox_area': area,
            'min_side': side,
            'visible_pixels': pix,
        }

    unique_step = {
        'visible_objects': {
            'Cup|1': _vis('Cup', [0.0, 0.0, 1.0]),
            'Bread|1': _vis('Bread', [0.5, 0.0, 1.0]),
        }
    }
    assert resolve_referring_disambiguator(unique_step, 'Cup|1') == ''

    # Clear margin: Chair|1 next to Bread, Chair|2 far away
    dup_step = {
        'visible_objects': {
            'Chair|1': _vis('Chair', [0.0, 0.0, 1.0]),
            'Chair|2': _vis('Chair', [3.0, 0.0, 1.0]),
            'Bread|1': _vis('Bread', [0.1, 0.0, 1.0]),
        }
    }
    phrase = resolve_referring_disambiguator(
        dup_step, 'Chair|1', {'heading': 0.0}
    )
    assert phrase == ' close to the Bread'

    # Tiny margin only — reject (sibling almost as close)
    tight = {
        'visible_objects': {
            'Chair|1': _vis('Chair', [0.0, 0.0, 1.0]),
            'Chair|2': _vis('Chair', [0.4, 0.0, 1.0]),
            'Bread|1': _vis('Bread', [0.1, 0.0, 1.0]),
        }
    }
    assert resolve_referring_disambiguator(tight, 'Chair|1') is None

    # Landmark too small in FOV — reject
    tiny_lm = {
        'visible_objects': {
            'Chair|1': _vis('Chair', [0.0, 0.0, 1.0]),
            'Chair|2': _vis('Chair', [3.0, 0.0, 1.0]),
            'Bread|1': _vis('Bread', [0.1, 0.0, 1.0], area=50, side=5, pix=20),
        }
    }
    assert resolve_referring_disambiguator(tiny_lm, 'Chair|1') is None

    # No unique landmark → exclude
    bare_dups = {
        'visible_objects': {
            'Chair|1': _vis('Chair', [0.0, 0.0, 1.0]),
            'Chair|2': _vis('Chair', [2.0, 0.0, 1.0]),
        }
    }
    assert resolve_referring_disambiguator(bare_dups, 'Chair|1') is None


@pytest.mark.parametrize(
    'construct', ['spatial_working_memory', 'spatial_updating']
)
def test_trajectory_hooks_are_scoped_to_item_frames(delayed_episode, construct):
    items = draft_items_for_episode(
        delayed_episode,
        constructs=[construct],
        max_per_construct=1,
        styles=('concise',),
    )
    ok = [i for i in items if i.get('status') == 'ok']
    if not ok:
        pytest.skip(f'no ok {construct} item')
    item = ok[0]

    poses = item.get('agent_trajectory')
    assert poses, 'expected pose per item frame'
    assert len(poses) == len(item['image_paths'])
    assert len(poses) <= len(delayed_episode['agent_trajectory'])
    for pose in poses:
        assert set(pose) <= {'step', 'x', 'z', 'heading'}
        assert {'x', 'z', 'heading'} <= set(pose)
        assert 'position' not in pose and 'rotation' not in pose
        assert 'image_path' not in pose

    actions = item.get('agent_actions') or []
    steps = [a['step'] for a in actions]
    assert all(item['encoding_step'] < s <= item['query_step'] for s in steps)
    assert len(actions) <= item['query_step'] - item['encoding_step']
    for act in actions:
        assert 'action' in act
        assert set(act) <= {'step', 'action', 'degrees'}
        assert 'position' not in act and 'rotation' not in act


def test_route_knowledge_does_not_leak_action_list(folder_episode):
    items = draft_items_for_episode(
        folder_episode,
        constructs=['route_knowledge'],
        max_per_construct=1,
        styles=('concise',),
    )
    ok = [i for i in items if i.get('status') == 'ok']
    if not ok:
        pytest.skip('no ok route item')
    assert ok[0].get('agent_actions') is None
    assert ok[0].get('options') == {}
    assert 'move_ahead' in (ok[0].get('answer') or '')


def test_write_draft_json(folder_episode, tmp_path):
    from cm_benchmark.generation.episode_io import write_draft_items

    items = draft_items_for_episode(
        folder_episode,
        constructs=['egocentric_encoding', 'invisible_displacement', 'perspective_taking'],
        max_per_construct=1,
        styles=('concise',),
    )
    out = write_draft_items(items, tmp_path / 'draft.json')
    assert out.is_file()
    import json

    data = json.loads(out.read_text())
    assert data['n_items'] == len(items)
