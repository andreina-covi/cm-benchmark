"""Construct templates and helpers for first-draft items (mirrors taxonomy YAML)."""

from __future__ import annotations

from typing import Optional

from cm_benchmark.generation.schema import CONSTRUCT_CLASS, CONSTRUCT_FOR

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

CONSTRUCT_TEMPLATES = {
    'egocentric_encoding': [
        'From your current view, where is the {object_type} relative to you?',
        'In which direction is the {object_type} from you?',
        'Where is the {object_type} relative to you?',
    ],
    'allocentric_encoding': [
        'Relative to the {reference_object}, where is the {object_type}?',
    ],
    # Multi-image constructs: wording must situate "now" / "earlier" in the frame sequence
    'spatial_working_memory': [
        'Looking at these images in order, where was the {object_type} relative to you when you last saw it (an earlier image, not the last)?',
        'Across these views in time order, where was the {object_type} relative to you when it was last visible?',
    ],
    'invisible_displacement': [
        'Looking at these images in order, where is the {object_type} now (after the last image)?',
        'After the sequence of views (earliest to latest), where is the {object_type}?',
    ],
    'spatial_updating': [
        'Looking at these images in order, where is the {object_type} relative to you now (in the last image)?',
        'After these views in time order, from your pose in the last image, where is the {object_type} relative to you?',
    ],
    'perspective_taking': [
        'From where the {reference_entity} faces, which object is on its left?',
    ],
    'route_knowledge': [
        'You need to go from {source} to {goal}. Which action sequence matches the route between these places in the views shown?',
        'Plan a route from {source} to {goal} along the path you walked. Which sequence is correct?',
    ],
    'survey_knowledge': [
        'Using the layout of the space, to go from {source} to {goal}, which connection should you use?',
        'Plan a route from {source} to {goal}. Which passage or room link is correct?',
    ],
}


def frame_sequence_cue(n_images: int) -> str:
    """Explicit temporal cue when the item attaches more than one frame."""
    if n_images <= 1:
        return ''
    if n_images == 2:
        return (
            'The two images are shown in time order '
            '(first = earlier view, second = later view). '
        )
    return (
        f'The {n_images} images are shown in time order '
        f'(first = earliest, last = latest). '
    )


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
    """Map GT angle_relation (x, y, z) to a multiple-choice ego direction label."""
    if not angle_relation or len(angle_relation) < 3:
        return None
    x_dir, y_dir, z_dir = angle_relation[0], angle_relation[1], angle_relation[2]
    if x_dir == 'left':
        return 'to your left'
    if x_dir == 'right':
        return 'to your right'
    if z_dir == 'front':
        return 'ahead of you'
    if z_dir == 'behind':
        return 'behind you'
    if y_dir == 'above':
        return 'above you'
    if y_dir == 'below':
        return 'below you'
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
