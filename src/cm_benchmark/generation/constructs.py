"""Construct templates and helpers for first-draft items (mirrors taxonomy YAML)."""

from __future__ import annotations

from typing import Optional

# Horizontal option bank used by egocentric / SWM / updating drafts
EGO_DIRECTION_OPTIONS = [
    'ahead of you',
    'to your right',
    'behind you',
    'to your left',
]

OPPOSITE = {
    'ahead of you': 'behind you',
    'behind you': 'ahead of you',
    'to your left': 'to your right',
    'to your right': 'to your left',
}

ORTHOGONAL = {
    'ahead of you': 'to your right',
    'behind you': 'to your left',
    'to your left': 'ahead of you',
    'to your right': 'behind you',
}

# Template banks keyed by construct; modes selected via fact.extra['template_mode'].
CONSTRUCT_TEMPLATES = {
    'egocentric_encoding': [
        'Where is the {object_type} relative to you right now?',
    ],
    'allocentric_encoding': [
        'Where is the {object_type} in relation to the {reference_object}?',
    ],
    'spatial_working_memory': {
        # Delay k is a difficulty axis — must be stated in the question (not implicit).
        'recall_relation': [
            (
                'You last saw the {object_type} {k} steps ago: '
                'where was it relative to you at that time?'
            )
        ],
        'recall_count': [
            'How many distinct {object_category} have you seen up to this point?',
        ],
    },
    'invisible_displacement': {
        'displacement_update': [
            (
                'While out of view, the {object_type} was moved to {new_location}. '
                'Where is it relative to you now?'
            ),
        ],
        'swap': [
            (
                'The {object_type} was moved to the previous location of the {other_object_type}. '
                'Where is the {object_type} relative to you now?'
            ),
        ],
    },
    'spatial_updating': [
        'You last saw the {object_type} {k} steps ago. Where is it relative to you now?',
    ],
    'perspective_taking': [
        'Given the direction the {reference_entity} is facing, which object is to its {relation}?',
    ],
    'route_knowledge': [
        (
            'What was the sequence of turns along the route you traveled from {source} to {goal}?'
        ),
    ],
    'survey_based_route_planning': [
        (
            "Using your knowledge of the environment's layout, "
            "what route should you take from {source} to {goal}?"
        ),
        (
            "Using your knowledge of the environment's layout, "
            "what route should you take from {source} to {goal}, "
            'given that {condition}?'
        )
    ],
}


def select_template(construct: str, template_mode: Optional[str] = None, index: int = 0) -> str:
    """Pick a question template for a construct / mode."""
    bank = CONSTRUCT_TEMPLATES.get(construct)
    if bank is None:
        return '(no template)'
    if isinstance(bank, dict):
        mode = template_mode
        if mode not in bank:
            mode = next(iter(bank))
        templates = bank.get(mode) or ['(no template)']
        return templates[index % len(templates)]
    return bank[index % len(bank)]


def frame_sequence_cue(n_images: int) -> str:
    """Deprecated: sequential/online protocol does not bundle images with the question.

    Kept as a no-op so callers do not accidentally reintroduce multi-image cues.
    """
    return ''


def online_temporal_preamble(construct: str, k: int) -> str:
    """Optional short cue for online models (stream already observed; no image bundle)."""
    k = max(1, int(k))
    if construct == 'spatial_working_memory':
        return f'Considering what you saw {k} steps ago. '
    if construct == 'spatial_updating':
        return f'Considering the last {k} navigation steps. '
    return ''


_BAD_CATEGORIES = frozenset(
    {
        '',
        'undefined',
        'none',
        'null',
        'nan',
        'unknown',
        'n/a',
        'na',
    }
)


def _category_usable(cat) -> bool:
    if cat is None:
        return False
    text = str(cat).strip()
    if not text:
        return False
    return text.lower() not in _BAD_CATEGORIES


def object_type_from_id(obj_id: str, visible_or_memory: Optional[dict] = None) -> str:
    """Human-readable object name for questions.

    Prefer a real ``category`` from GT when present. Simulator placeholders such as
    ``Undefined`` (common for some Objaverse assets) fall back to the id stem
    (``ObjaScooter|4|5`` → ``ObjaScooter``).
    """
    if visible_or_memory and obj_id in visible_or_memory:
        cat = visible_or_memory[obj_id].get('category')
        if _category_usable(cat):
            return str(cat).strip()
    stem = str(obj_id).split('|')[0].strip()
    return stem if stem else str(obj_id)


def angle_relation_to_ego_label(angle_relation) -> Optional[str]:
    """Map GT angle_relation (x, y, z) to a multiple-choice ego direction label.

    Only horizontal labels are used in MC pools (left/right/ahead/behind).
    Vertical-only relations return None so the planner can skip them.
    """
    if not angle_relation or len(angle_relation) < 3:
        return None
    x_dir, _y_dir, z_dir = angle_relation[0], angle_relation[1], angle_relation[2]
    if x_dir == 'left':
        return 'to your left'
    if x_dir == 'right':
        return 'to your right'
    if z_dir == 'front':
        return 'ahead of you'
    if z_dir == 'behind':
        return 'behind you'
    return None


def find_ego_edge(step: dict, obj_id: str) -> Optional[dict]:
    for edge in step.get('edges_egocentric') or []:
        if edge.get('target') == obj_id and edge.get('source') == 'agent':
            return edge
    return None


def find_inferred_edge(step: dict, obj_id: str) -> Optional[dict]:
    for edge in step.get('edges_inferred') or []:
        if edge.get('target') == obj_id:
            return edge
    return None


def find_allocentric_edge(step: dict) -> Optional[dict]:
    for edge in step.get('edges_allocentric') or []:
        src, tgt = edge.get('source'), edge.get('target')
        if src and tgt and src != 'agent' and tgt != 'agent':
            return edge
    return None


def step_by_index(episode: dict, step_idx: int) -> Optional[dict]:
    for step in episode.get('steps') or []:
        if int(step.get('step')) == int(step_idx):
            return step
    return None


def humanize_receptacle(receptacle_id: Optional[str]) -> str:
    if not receptacle_id or str(receptacle_id).lower() in ('none', 'null', ''):
        return 'on the floor'
    typ = str(receptacle_id).split('|')[0]
    return f'on/in the {typ}'
