"""Construct templates and helpers for generated items (mirrors taxonomy YAML)."""

from __future__ import annotations

from typing import Any, Optional
import math

# Horizontal option bank used by egocentric / SWM / updating items
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
            'How many distinct {object_category}s have you seen up to this point?',
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
            'List, in order, the navigation actions you took to travel '
            'from the {source} to the {goal}.'
        ),
    ],
    'survey_based_route_planning': [
        (
            "Using your knowledge of the environment's layout, list the "
            'navigation actions that take you from the {source} to the {goal}.'
        ),
        (
            "Based on what you've seen of this space, list the navigation "
            'actions, in order, that go from the {source} to the {goal}.'
        ),
    ],
}


def _template_bank(construct: str, template_mode: Optional[str] = None) -> list:
    bank = CONSTRUCT_TEMPLATES.get(construct)
    if bank is None:
        return []
    if isinstance(bank, dict):
        mode = template_mode if template_mode in bank else next(iter(bank), None)
        return list(bank.get(mode) or [])
    return list(bank)


def _template_text(entry) -> str:
    """Normalize a bank entry to a format string (unwrap accidental 1-tuples)."""
    if isinstance(entry, tuple):
        return ''.join(str(p) for p in entry) if entry else '(no template)'
    return str(entry)


def template_count(construct: str, template_mode: Optional[str] = None) -> int:
    return len(_template_bank(construct, template_mode))


def pick_template_index(
    construct: str,
    template_mode: Optional[str] = None,
    *keys: Any,
) -> int:
    """Stable paraphrase index from item keys (source/goal ids, …)."""
    n = template_count(construct, template_mode)
    if n <= 1:
        return 0
    seed = '|'.join(str(k) for k in keys if k is not None)
    return sum(ord(c) for c in seed) % n


def select_template(construct: str, template_mode: Optional[str] = None, index: int = 0) -> str:
    """Pick a question template for a construct / mode."""
    templates = _template_bank(construct, template_mode)
    if not templates:
        return '(no template)'
    return _template_text(templates[index % len(templates)])


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
    camera FOV (egocentric_encoding, disambiguator, SWM encode). Use
    ``AHEAD_HALF_WIDTH_FULL`` (45°) for full-circle pose updates
    (spatial_updating, invisible-displacement query). Perspective-taking uses
    ``imagined_perspective_label`` / signed-angle bins instead of this wedge.
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
# Full-circle constructs after a real pose change (spatial_updating, ID query).
AHEAD_HALF_WIDTH_FULL = 45.0

# Perspective-taking (stand at A, face B, locate C): signed-angle bins with no
# "ahead" class. Reject samples within this margin of 0° / ±135° boundaries.
PT_DIRECTION_BOUNDARY_MARGIN_DEG = 10.0
PT_DIRECTION_BACK_DEG = 135.0


def bearing_deg_xz(dx: float, dz: float) -> Optional[float]:
    """Horizontal bearing degrees from local/world xz offset (0 = +Z ahead)."""
    if abs(float(dx)) < 1e-12 and abs(float(dz)) < 1e-12:
        return None
    return math.degrees(math.atan2(float(dx), float(dz))) % 360.0


def signed_rel_bearing_deg_xz(
    fwd_dx: float, fwd_dz: float, tgt_dx: float, tgt_dz: float
) -> Optional[float]:
    """Signed horizontal angle from forward to target, degrees in (-180, 180].

    Uses the same xz / ``atan2(dx, dz)`` frame as ``bearing_deg_xz`` (0 = +Z).
    Sign convention (Y-up, viewed from above): **positive = left (CCW)**,
    **negative = right (CW)**. Example: facing +Z, target on +X → ≈ −90°.
    """
    fx, fz = float(fwd_dx), float(fwd_dz)
    tx, tz = float(tgt_dx), float(tgt_dz)
    f_len = math.hypot(fx, fz)
    t_len = math.hypot(tx, tz)
    if f_len < 1e-12 or t_len < 1e-12:
        return None
    fx, fz = fx / f_len, fz / f_len
    tx, tz = tx / t_len, tz / t_len
    # up · (forward × target) in Y-up: fx*tz - fz*tx
    cross = fx * tz - fz * tx
    dot = fx * tx + fz * tz
    return math.degrees(math.atan2(cross, dot))


def perspective_direction_label_from_signed(
    signed_deg: float,
    *,
    margin_deg: float = PT_DIRECTION_BOUNDARY_MARGIN_DEG,
    back_deg: float = PT_DIRECTION_BACK_DEG,
) -> Optional[str]:
    """Map signed A→B vs A→C angle to left / right / behind; None if ambiguous.

    Bins (no ``ahead of you`` answer class)::

        right : -back < angle < 0
        left  :  0 < angle < +back
        back  : |angle| >= back   → ``behind you``

    Reject when ``|angle|`` is within ``margin_deg`` of 0° or ±back (decision
    boundaries). World object extent is not used here — episode GT does not
    carry trusted world-space AABB extents for angular span checks.
    """
    a = float(signed_deg)
    m = max(0.0, float(margin_deg))
    back = float(back_deg)
    if abs(a) < m:
        return None
    if abs(abs(a) - back) < m:
        return None
    if abs(a) >= back:
        return 'behind you'
    if a < 0.0:
        return 'to your right'
    return 'to your left'


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
# soft Q&A filter when **no** DecisionTree joblib is loaded. These floors are
# a fallback only — prefer refitting ``visibility_filter.joblib`` from labels
# instead of bumping these after each bad item.
QUERY_FOV_MIN_BBOX_AREA = 1200.0
QUERY_FOV_MIN_SIDE = 32.0
QUERY_FOV_MIN_VISIBLE_PIXELS = 300.0
# If the bbox touches the image border, the extent *along that axis* must still
# be large enough that the object is not just a clipped strip (geometric rule;
# kept even when the DecisionTree is active).
QUERY_FOV_EDGE_MARGIN_PX = 2
QUERY_FOV_MIN_EXTENT_IF_CLIPPED = 48.0

# Invisible-displacement objects are small movable props (Cup, Phone, Bread).
# Encode distinguishability: DecisionTree when attached on the episode, else these
# soft floors (not landmark QUERY_FOV). At query they must be invisible.
ID_ENCODE_FOV_MIN_BBOX_AREA = 100.0
ID_ENCODE_FOV_MIN_SIDE = 8.0
ID_ENCODE_FOV_MIN_VISIBLE_PIXELS = 40.0


def camera_size_wh(episode: Optional[dict]) -> Optional[tuple[int, int]]:
    """Return (width, height) from episode_meta.camera when present."""
    if not isinstance(episode, dict):
        return None
    cam = (episode.get('episode_meta') or {}).get('camera') or {}
    try:
        w = int(cam.get('width'))
        h = int(cam.get('height'))
    except (TypeError, ValueError):
        return None
    if w > 0 and h > 0:
        return w, h
    return None


def bbox_not_border_scrape(
    odata: Optional[dict],
    image_wh: Optional[tuple[int, int]],
    *,
    edge_margin_px: int = QUERY_FOV_EDGE_MARGIN_PX,
    min_extent_if_clipped: float = QUERY_FOV_MIN_EXTENT_IF_CLIPPED,
) -> bool:
    """False when the detection is only a thin strip along the image border.

    Example reject: bbox ``[0, 177, 25, 223]`` on a 396×224 frame — left-edge
    clip with width 25 px; a human cannot identify the category.
    When bbox or image size is missing, do not invent a reject.
    """
    if not isinstance(odata, dict) or not image_wh:
        return True
    bbox = odata.get('bbox')
    if not bbox or len(bbox) < 4:
        return True
    try:
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        width = x1 - x0
        height = y1 - y0
        img_w, img_h = int(image_wh[0]), int(image_wh[1])
    except (TypeError, ValueError, IndexError):
        return True
    if width <= 0 or height <= 0 or img_w <= 0 or img_h <= 0:
        return False
    m = max(0, int(edge_margin_px))
    thr = float(min_extent_if_clipped)
    clipped_x = x0 <= m or x1 >= img_w - 1 - m
    clipped_y = y0 <= m or y1 >= img_h - 1 - m
    if clipped_x and width < thr:
        return False
    if clipped_y and height < thr:
        return False
    return True


def fov_metrics_ok(
    odata: Optional[dict],
    *,
    min_bbox_area: float = QUERY_FOV_MIN_BBOX_AREA,
    min_side: float = QUERY_FOV_MIN_SIDE,
    min_visible_pixels: float = QUERY_FOV_MIN_VISIBLE_PIXELS,
    image_wh: Optional[tuple[int, int]] = None,
    min_extent_if_clipped: float = QUERY_FOV_MIN_EXTENT_IF_CLIPPED,
    model: Any = None,
) -> bool:
    """True if FOV detection is distinguishable enough to name / encode.

    When ``model`` (DecisionTree visibility bundle) is set, keep/drop follows
    ``passes_for_questions`` — do not also apply static QUERY floors.
    Otherwise enforce static ``min_*`` floors (fallback).

    When no bbox metrics are present (legacy fixtures), do not invent a reject
    from floors; still apply border-scrape when bbox + image size exist.
    Optional ``image_wh`` rejects thin border-clipped scrapes either way.
    """
    if not isinstance(odata, dict):
        return False

    if model is not None:
        from cm_benchmark.generator.visibility_filters import metrics_dict_from_object

        if not model.passes_for_questions(metrics_dict_from_object(odata)):
            return False
        return bbox_not_border_scrape(
            odata, image_wh, min_extent_if_clipped=min_extent_if_clipped
        )

    checks: list[tuple[Any, float]] = []
    if odata.get('bbox_area') is not None:
        checks.append((odata.get('bbox_area'), min_bbox_area))
    if odata.get('min_side') is not None:
        checks.append((odata.get('min_side'), min_side))
    if odata.get('visible_pixels') is not None:
        checks.append((odata.get('visible_pixels'), min_visible_pixels))
    if not checks:
        # Still apply border-scrape check when bbox + image size exist.
        return bbox_not_border_scrape(
            odata, image_wh, min_extent_if_clipped=min_extent_if_clipped
        )
    try:
        if not all(float(val) >= float(thr) for val, thr in checks):
            return False
    except (TypeError, ValueError):
        return False
    return bbox_not_border_scrape(
        odata, image_wh, min_extent_if_clipped=min_extent_if_clipped
    )


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
    model: Any = None,
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
        and fov_metrics_ok(raw_vis.get(oid), model=model)
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
    *,
    model: Any = None,
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
    info = find_disambiguator(
        step, target_obj_id, group, agent_pose, model=model
    )
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
    margin_deg: float = PT_DIRECTION_BOUNDARY_MARGIN_DEG,
    ahead_half_width: Optional[float] = None,
) -> Optional[str]:
    """Direction of C from an imagined viewpoint standing at A, facing B.

    - Observer: A
    - Forward: A → B
    - Target: A → C
    - Signed angle on the horizontal plane (positive = left, negative = right)
    - Labels: ``to your right`` / ``to your left`` / ``behind you``
    - Returns None near 0° / ±135° boundaries (``margin_deg``) so ambiguous
      QA samples are not generated.

    ``ahead_half_width`` is accepted for call-site compatibility but ignored —
    perspective-taking no longer uses equal ahead wedges.
    """
    del ahead_half_width  # unused; kept for backward-compatible kwargs
    try:
        ax, az = float(pos_a['x']), float(pos_a['z'])
        bx, bz = float(pos_b['x']), float(pos_b['z'])
        cx, cz = float(pos_c['x']), float(pos_c['z'])
    except (KeyError, TypeError, ValueError):
        return None
    signed = signed_rel_bearing_deg_xz(bx - ax, bz - az, cx - ax, cz - az)
    if signed is None:
        return None
    return perspective_direction_label_from_signed(signed, margin_deg=margin_deg)


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
