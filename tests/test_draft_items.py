"""Tests for first-draft taxonomy Q&A generation."""

from copy import deepcopy
from pathlib import Path

import pytest

from cm_benchmark.generator.ai2thor_nav_generator import Ai2ThorNavGenerator
from cm_benchmark.generation.constructs import select_template
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
    assert item['answer_source'][0].startswith('steps[')
    step = tiny_episode['steps'][item['query_step']]
    edge = next(
        e
        for e in step['edges_egocentric']
        if e['target'] == item['queried_object_id']
    )
    from cm_benchmark.generation.constructs import angle_relation_to_ego_label

    assert item['options'][item['answer']] == angle_relation_to_ego_label(
        edge['angle_relation']
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
        assert len(labels) >= 2
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
    """Class-4 route items retrace walked A→B via derive_turns (not full dump)."""
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
    assert ' → ' in (fact.answer_label or '')
    q = _core_question(fact).lower()
    assert 'sequence of turns' in q or 'traveling' in q or 'matches' in q
    assert 'plan a route' not in q


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
        )
        return
    fact = facts[0]
    assert fact.extra.get('direction')
    assert fact.extra.get('distance_label')
    assert 'turn left' not in (fact.answer_label or '').lower() or '@' not in (
        fact.answer_label or ''
    )


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


def test_core_question_covers_active_template_placeholders():
    """Guard against KeyError when formatting templates."""
    from cm_benchmark.generation.planner import PlannedFact

    for construct, modes in (
        ('egocentric_encoding', [None]),
        ('invisible_displacement', ['recall_direction', 'swap']),
        ('spatial_updating', [None]),
        ('route_knowledge', [None]),
        ('survey_based_route_planning', ['direction_distance', 'conditional_detour']),
        ('perspective_taking', [None]),
        ('spatial_working_memory', ['recall_relation', 'recall_count']),
    ):
        for mode in modes:
            tmpl = select_template(construct, template_mode=mode)
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
                },
            )
            q = _core_question(fact)
            assert '{object' not in q
            assert '{k}' not in q
            assert '{new_location}' not in q
            assert '{other_object_type}' not in q
            assert '{A}' not in q and '{C}' not in q
            assert tmpl.split('{')[0] in q or 'Cup' in q or 'Kitchen' in q or 'Armchair' in q


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
    assert [p.get('image_path') for p in poses] == item['image_paths']
    assert len(poses) <= len(delayed_episode['agent_trajectory'])

    actions = item.get('agent_actions') or []
    steps = [a['step'] for a in actions]
    assert all(item['encoding_step'] < s <= item['query_step'] for s in steps)
    assert len(actions) <= item['query_step'] - item['encoding_step']


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
