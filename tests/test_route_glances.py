"""Route answers skip SPOC glance turns (``scripted_glance`` in agent_actions)."""

from cm_benchmark.generation.route_replay import recorded_actions_between


def _episode():
    # step 1-2: glance out and back; 3: policy move; 4-5: second glance
    rows = [
        (1, 'move_ahead', False),
        (2, 'rotate_left', True),
        (3, 'rotate_right', True),
        (4, 'move_ahead', False),
        (5, 'rotate_right', True),
        (6, 'rotate_left', True),
        (7, 'rotate_left', False),
    ]
    actions = []
    for step, act, glance in rows:
        rec = {'step': step, 'action': act, 'action_success': True}
        if glance:
            rec['scripted_glance'] = True
        actions.append(rec)
    return {'agent_actions': actions}


def test_glance_turns_dropped_from_route():
    ep = _episode()
    assert recorded_actions_between(ep, 0, 7, skip_glances=True) == [
        'move_ahead',
        'move_ahead',
        'rotate_left',
    ]


def test_default_keeps_every_action():
    assert len(recorded_actions_between(_episode(), 0, 7)) == 7


def test_window_cutting_a_glance_keeps_its_turns():
    # Starts after the glance turned out (step 2): the turn back is needed.
    assert recorded_actions_between(_episode(), 2, 7, skip_glances=True) == [
        'rotate_right',
        'move_ahead',
        'rotate_left',
    ]
    # Ends mid-glance (step 5): the first glance is dropped, the cut one kept.
    assert recorded_actions_between(_episode(), 0, 5, skip_glances=True) == [
        'move_ahead',
        'move_ahead',
        'rotate_right',
    ]
