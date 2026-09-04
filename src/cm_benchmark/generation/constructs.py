"""Construct templates and helpers for first-draft items (mirrors taxonomy YAML)."""

from __future__ import annotations

from typing import Optional
import math

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

MIRRORED_LR = {
    'ahead of you': 'ahead of you',
    'behind you': 'behind you',
    'to your left': 'to your right',
    'to your right': 'to your left',
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
        # Destination cue required: named landmark (direct) or partner object (swap).
        # Never ask "Where is X now?" alone — the hidden move is unwitnessed.
        # Wording is for online sequential models (stream already seen; no image bundle).
        # k is stored on the item for analysis; the probe is destination tracking, not delay.
        'recall_direction': [
            (
                'While out of view, the {object_type} was moved onto {new_location}. '
                'Where is it relative to you now?'
            ),
        ],
        'swap': [
            (
                'The {object_type} was moved to where the {other_object_type} used to be. '
                'Where is the {object_type} relative to you now?'
            ),
        ],
    },
    'spatial_updating': [
        (
            'Considering the last {k} navigation steps, '
            'where is the {object_type} relative to you now?'
        ),
    ],
    'perspective_taking': [
        'Imagine standing at the {A}, facing the {B}. From that position, where is the {C}?',
    ],
    'route_knowledge': [
        (
            'Which of these matches the sequence of turns you made traveling '
            'from {source} to {goal}?'
        ),
    ],
    'survey_based_route_planning': {
        'direction_distance': [
            (
                "Using your knowledge of the environment's layout, "
                'where is the {goal} relative to the {source}?'
            ),
        ],
        'conditional_detour': [
            (
                "Using your knowledge of the environment's layout, where would you "
                'first head to reach the {goal} from the {source}, given that {condition}?'
            ),
        ],
    },
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


def humanize_receptacle(
    receptacle_id: Optional[str], floor_anchor_landmark: Optional[str] = None
) -> Optional[str]:
    """Returns None if Floor with no anchor — caller must reject that candidate,
    never silently emit a bare 'on the floor' cue."""
    is_floor = (
        not receptacle_id
        or str(receptacle_id).lower() in ('none', 'null', '', 'floor')
    )
    if is_floor:
        return (
            f'the floor near the {floor_anchor_landmark}'
            if floor_anchor_landmark
            else None
        )
    typ = str(receptacle_id).split('|')[0]
    return f'on/in the {typ}'


def _pose_xz_heading(pose: dict) -> tuple[float, float, float]:
    """Normalize pose dict to (x, z, heading_deg). Accepts x/z or position."""
    if 'x' in pose and 'z' in pose:
        x, z = float(pose['x']), float(pose['z'])
    else:
        pos = pose.get('position') or pose.get('pos')
        x, z = float(pos[0]), float(pos[2])
    if 'heading' in pose:
        heading = float(pose['heading'])
    else:
        rot = pose.get('rotation') or pose.get('rot') or [0, 0, 0]
        heading = float(rot[1] if len(rot) > 1 else rot[0])
    return x, z, heading


def net_pose_changed(
    pose_a: dict, pose_b: dict, pos_tol: float = 0.1, heading_tol_deg: float = 5.0
) -> bool:
    """Real move check from actual positions — never trust action count alone."""
    ax, az, ah = _pose_xz_heading(pose_a)
    bx, bz, bh = _pose_xz_heading(pose_b)
    pos_delta = ((bx - ax) ** 2 + (bz - az) ** 2) ** 0.5
    heading_delta = abs((bh - ah + 180) % 360 - 180)
    return pos_delta > pos_tol or heading_delta > heading_tol_deg


def imagined_perspective_label(pos_a: dict, pos_b: dict, pos_c: dict) -> Optional[str]:
    """Direction of C from an imagined viewpoint standing at A, facing B.

    Heading is RELATIONAL (A→B) — no object-intrinsic-front data needed.
    Positions are dicts with x/z (y optional).
    """
    try:
        ax, az = float(pos_a['x']), float(pos_a['z'])
        bx, bz = float(pos_b['x']), float(pos_b['z'])
        cx, cz = float(pos_c['x']), float(pos_c['z'])
    except (KeyError, TypeError, ValueError):
        return None
    if abs(bx - ax) < 1e-9 and abs(bz - az) < 1e-9:
        return None
    if abs(cx - ax) < 1e-9 and abs(cz - az) < 1e-9:
        return None
    heading = math.degrees(math.atan2(bx - ax, bz - az))
    angle = math.degrees(math.atan2(cx - ax, cz - az))
    rel = (angle - heading + 360) % 360
    if rel < 45 or rel >= 315:
        return 'ahead of you'
    if rel < 135:
        return 'to your right'
    if rel < 225:
        return 'behind you'
    return 'to your left'


def xyz_as_dict(pos) -> Optional[dict]:
    """Normalize list/tuple/dict world pose to ``{x,y,z}``."""
    if pos is None:
        return None
    if isinstance(pos, dict) and 'x' in pos and 'z' in pos:
        return {
            'x': float(pos['x']),
            'y': float(pos.get('y', 0.0)),
            'z': float(pos['z']),
        }
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        return {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])}
    return None
