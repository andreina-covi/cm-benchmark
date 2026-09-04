"""Construct templates and helpers for first-draft items (mirrors taxonomy YAML)."""

from __future__ import annotations

from typing import Any, Optional
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
# {disambiguator} is '' when the category is unique, else a leading-space phrase
# like " close to the Bread" (taxonomy referring-expression exception).
CONSTRUCT_TEMPLATES = {
    'egocentric_encoding': [
        'Where is the {object_type}{disambiguator} relative to you right now?',
    ],
    'allocentric_encoding': [
        'Where is the {object_type} in relation to the {reference_object}?',
    ],
    'spatial_working_memory': {
        # Delay k is a difficulty axis — must be stated in the question (not implicit).
        'recall_relation': [
            (
                'You last saw the {object_type}{disambiguator} {k} steps ago: '
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
                'While out of view, the {object_type}{disambiguator} was moved onto {new_location}. '
                'Where is it relative to you now?'
            ),
        ],
        'swap': [
            (
                'The {object_type}{disambiguator} was moved to where the {other_object_type} used to be. '
                'Where is the {object_type} relative to you now?'
            ),
        ],
    },
    'spatial_updating': [
        (
            'Considering the last {k} navigation steps, '
            'where is the {object_type}{disambiguator} relative to you now?'
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


def angle_to_ego_label(angle_deg: float, ahead_half_width: float = 45.0) -> str:
    """Map bearing degrees to an ego MC label with a tunable ahead wedge.

    ``ahead_half_width`` is half the ahead sector in degrees (total ahead =
    2 * width). Side/behind wedges share the remaining circle symmetrically:

      ahead:  [360-w, 360) ∪ [0, w)
      right:  [w, 180-w)
      behind: [180-w, 180+w)
      left:   [180+w, 360-w)

    Use ``AHEAD_HALF_WIDTH_FOV`` (20°) when the object must be in the current
    camera FOV (egocentric_encoding, disambiguator landmarks). Use
    ``AHEAD_HALF_WIDTH_FULL`` (45°) when the queried pose can place the object
    anywhere (spatial_updating, perspective_taking, hidden-object query poses).
    """
    w = float(ahead_half_width)
    a = float(angle_deg) % 360.0
    if a < w or a >= 360.0 - w:
        return 'ahead of you'
    if a < 180.0 - w:
        return 'to your right'
    if a < 180.0 + w:
        return 'behind you'
    return 'to your left'


# FOV-constrained constructs: object visible now → bearing stays near ahead;
# a 45° half-width collapses almost everything to "ahead of you".
AHEAD_HALF_WIDTH_FOV = 20.0
# Full-circle constructs: after real/imagined pose change, behind is legitimate.
AHEAD_HALF_WIDTH_FULL = 45.0


def bearing_deg_xz(dx: float, dz: float) -> Optional[float]:
    """Horizontal bearing degrees from local/world xz offset (0 = +Z ahead)."""
    if abs(float(dx)) < 1e-12 and abs(float(dz)) < 1e-12:
        return None
    return math.degrees(math.atan2(float(dx), float(dz))) % 360.0


def local_offset_to_ego_label(
    local_xyz, *, ahead_half_width: float = AHEAD_HALF_WIDTH_FULL
) -> Optional[str]:
    """MC ego label from a local (dx, dy, dz) offset."""
    try:
        x = float(local_xyz[0])
        z = float(local_xyz[2])
    except (TypeError, ValueError, IndexError):
        return None
    bearing = bearing_deg_xz(x, z)
    if bearing is None:
        return None
    return angle_to_ego_label(bearing, ahead_half_width=ahead_half_width)


def angle_relation_to_ego_label(
    angle_relation, *, ahead_half_width: float = AHEAD_HALF_WIDTH_FULL
) -> Optional[str]:
    """Map a stored relation triple or local offset to an MC label.

    Prefer ``local_offset_to_ego_label`` / ``angle_to_ego_label`` at call sites.
    For legacy exclusive triples (one of left/right/front/behind), map directly;
    composite triples from the old 15° scheme are not recovered — return None
    so callers recompute from poses.
    """
    if not angle_relation or len(angle_relation) < 3:
        return None
    # Numeric local offset (dx, dy, dz)
    try:
        x, _y, z = float(angle_relation[0]), float(angle_relation[1]), float(angle_relation[2])
        return local_offset_to_ego_label((x, _y, z), ahead_half_width=ahead_half_width)
    except (TypeError, ValueError):
        pass
    x_dir, _y_dir, z_dir = angle_relation[0], angle_relation[1], angle_relation[2]
    bits = [b for b in (x_dir, z_dir) if b]
    if len(bits) > 1:
        return None
    if x_dir == 'left':
        return 'to your left'
    if x_dir == 'right':
        return 'to your right'
    if z_dir == 'front':
        return 'ahead of you'
    if z_dir == 'behind':
        return 'behind you'
    return None


_MC_TO_REFERRING = {
    'ahead of you': 'ahead of',
    'to your right': 'to the right of',
    'behind you': 'behind',
    'to your left': 'to the left of',
}


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


def translated_egocentric_label(
    agent_heading: float,
    landmark_pos: dict,
    target_pos: dict,
    *,
    ahead_half_width: float = AHEAD_HALF_WIDTH_FOV,
) -> Optional[str]:
    """Direction of target from landmark in the agent's yaw frame (referring only).

    Default FOV half-width: disambiguator landmarks are co-visible in-frame.
    Maps ``angle_to_ego_label`` output to landmark-relative phrases.
    """
    dx = float(target_pos['x']) - float(landmark_pos['x'])
    dz = float(target_pos['z']) - float(landmark_pos['z'])
    bearing = bearing_deg_xz(dx, dz)
    if bearing is None:
        return None
    rel = (bearing - float(agent_heading)) % 360.0
    return _MC_TO_REFERRING[
        angle_to_ego_label(rel, ahead_half_width=ahead_half_width)
    ]


def _visible_catalog(step: dict) -> dict[str, dict]:
    """obj_id -> {category, position{x,y,z}, metrics...} from step.visible_objects."""
    out: dict[str, dict] = {}
    for oid, odata in (step.get('visible_objects') or {}).items():
        if not isinstance(odata, dict):
            continue
        pos = xyz_as_dict(odata.get('position'))
        if pos is None:
            continue
        cat = object_type_from_id(oid, {oid: odata})
        out[oid] = {
            'category': cat,
            'position': pos,
            'bbox_area': odata.get('bbox_area'),
            'min_side': odata.get('min_side'),
            'visible_pixels': odata.get('visible_pixels'),
            'occupancy_ratio': odata.get('occupancy_ratio'),
        }
    return out


# Referring expressions must be obvious in the image, not just metrically true.
# Reject "close to" when the nearest sibling is almost as near (tiny margins).
MIN_DISAMBIG_MARGIN_M = 0.6
MIN_DISAMBIG_MARGIN_RATIO = 1.5  # nearest sibling must be ≥ this × target–landmark dist

# Landmarks / recalled query targets need a clearer FOV footprint than the
# soft Q&A filter (which still keeps small props). Tuned above HousePlant@s5
# (bbox≈432, side≈16) which was not human-distinguishable in review.
QUERY_FOV_MIN_BBOX_AREA = 800.0
QUERY_FOV_MIN_SIDE = 24.0
QUERY_FOV_MIN_VISIBLE_PIXELS = 200.0


def fov_metrics_ok(
    odata: Optional[dict],
    *,
    min_bbox_area: float = QUERY_FOV_MIN_BBOX_AREA,
    min_side: float = QUERY_FOV_MIN_SIDE,
    min_visible_pixels: float = QUERY_FOV_MIN_VISIBLE_PIXELS,
) -> bool:
    """True if FOV detection is large enough to treat as distinguishable.

    When no bbox metrics are present (legacy fixtures), do not invent a reject.
    When some metrics are present, enforce only those that are non-null.
    """
    if not isinstance(odata, dict):
        return False
    checks: list[tuple[Any, float]] = []
    if odata.get('bbox_area') is not None:
        checks.append((odata.get('bbox_area'), min_bbox_area))
    if odata.get('min_side') is not None:
        checks.append((odata.get('min_side'), min_side))
    if odata.get('visible_pixels') is not None:
        checks.append((odata.get('visible_pixels'), min_visible_pixels))
    if not checks:
        return True
    try:
        return all(float(val) >= float(thr) for val, thr in checks)
    except (TypeError, ValueError):
        return False


def duplicate_category_group(step: dict, target_obj_id: str) -> list[str]:
    """Visible object ids sharing the target's category (includes target)."""
    catalog = _visible_catalog(step)
    if target_obj_id not in catalog:
        return []
    cat = catalog[target_obj_id]['category']
    return [oid for oid, o in catalog.items() if o['category'] == cat]


def format_disambiguator_phrase(info: dict) -> str:
    """Leading-space phrase for templates, e.g. `` close to the Bread``."""
    relation = str(info.get('relation') or '').strip()
    landmark = str(info.get('landmark_type') or '').strip()
    if not relation or not landmark:
        return ''
    if relation == 'close to':
        return f' close to the {landmark}'
    if relation == 'behind':
        return f' behind the {landmark}'
    return f' {relation} the {landmark}'


def find_disambiguator(
    step: dict,
    target_obj_id: str,
    duplicate_group: list[str],
    agent_pose: Optional[dict] = None,
    *,
    min_margin_m: float = MIN_DISAMBIG_MARGIN_M,
    min_margin_ratio: float = MIN_DISAMBIG_MARGIN_RATIO,
) -> Optional[dict]:
    """Landmark + relation that uniquely identifies target among duplicate_group.

    Proximity is accepted only when the margin is large enough to be obvious in
    the image (absolute meters AND relative ratio). Landmarks must themselves
    pass ``fov_metrics_ok``. Returns None if nothing qualifies — caller skips.
    """
    if len(duplicate_group) < 2:
        return None

    catalog = _visible_catalog(step)
    raw_vis = step.get('visible_objects') or {}
    target = catalog.get(target_obj_id)
    if target is None:
        return None
    target_pos = target['position']

    cat_counts: dict[str, int] = {}
    for o in catalog.values():
        cat_counts[o['category']] = cat_counts.get(o['category'], 0) + 1

    siblings = set(duplicate_group) - {target_obj_id}
    landmarks = [
        (oid, o)
        for oid, o in catalog.items()
        if oid not in duplicate_group
        and cat_counts.get(o['category'], 0) == 1
        and fov_metrics_ok(raw_vis.get(oid))
    ]

    def dist(a: dict, b: dict) -> float:
        return ((a['x'] - b['x']) ** 2 + (a['z'] - b['z']) ** 2) ** 0.5

    # 1) proximity — target closest with a visually clear margin over siblings
    best = None
    for oid, o in landmarks:
        d_target = dist(target_pos, o['position'])
        if d_target <= 1e-6:
            continue
        sib_dists = [
            dist(catalog[s]['position'], o['position']) for s in siblings if s in catalog
        ]
        if not sib_dists:
            continue
        d_sib = min(sib_dists)
        margin = d_sib - d_target
        if margin < float(min_margin_m):
            continue
        if d_sib < float(min_margin_ratio) * d_target:
            continue
        if best is None or margin > best[1]:
            best = (
                {
                    'landmark_id': oid,
                    'relation': 'close to',
                    'landmark_type': o['category'],
                    'margin_m': margin,
                },
                margin,
            )
    if best:
        return best[0]

    # 2) fallback — landmark→target direction in agent yaw, unique among siblings
    heading = None
    if agent_pose and agent_pose.get('heading') is not None:
        try:
            heading = float(agent_pose['heading'])
        except (TypeError, ValueError):
            heading = None
    if heading is None:
        return None

    for oid, o in landmarks:
        rel = translated_egocentric_label(heading, o['position'], target_pos)
        if rel is None:
            continue
        sib_rels = {
            translated_egocentric_label(heading, o['position'], catalog[s]['position'])
            for s in siblings
            if s in catalog
        }
        if rel not in sib_rels:
            return {
                'landmark_id': oid,
                'relation': rel,
                'landmark_type': o['category'],
            }
    return None


def resolve_referring_disambiguator(
    step: dict,
    target_obj_id: str,
    agent_pose: Optional[dict] = None,
) -> Optional[str]:
    """Phrase for templates, or None to skip the candidate.

    - Unique category in the step → ``''`` (no phrase; not ambiguous).
    - Duplicate category + unique landmark phrase → leading-space phrase.
    - Duplicate category + no unique phrase → ``None`` (exclude from queries).
    """
    group = duplicate_category_group(step, target_obj_id)
    if not group:
        return None
    if len(group) == 1:
        return ''
    info = find_disambiguator(step, target_obj_id, group, agent_pose)
    if info is None:
        return None
    return format_disambiguator_phrase(info)


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


def imagined_perspective_label(
    pos_a: dict,
    pos_b: dict,
    pos_c: dict,
    *,
    ahead_half_width: float = AHEAD_HALF_WIDTH_FULL,
) -> Optional[str]:
    """Direction of C from an imagined viewpoint standing at A, facing B.

    Heading is RELATIONAL (A→B). Full-circle half-width by default — the
    imagined pose is not FOV-constrained.
    """
    try:
        ax, az = float(pos_a['x']), float(pos_a['z'])
        bx, bz = float(pos_b['x']), float(pos_b['z'])
        cx, cz = float(pos_c['x']), float(pos_c['z'])
    except (KeyError, TypeError, ValueError):
        return None
    heading = bearing_deg_xz(bx - ax, bz - az)
    angle = bearing_deg_xz(cx - ax, cz - az)
    if heading is None or angle is None:
        return None
    rel = (angle - heading) % 360.0
    return angle_to_ego_label(rel, ahead_half_width=ahead_half_width)


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
