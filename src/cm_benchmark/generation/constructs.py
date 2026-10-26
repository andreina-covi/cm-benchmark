"""Construct templates and helpers for generated items (mirrors taxonomy YAML)."""

from __future__ import annotations

import re

from typing import Any, Optional
import math

from cm_benchmark.generation.config import get_generation_config, legacy_constant

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
            'List the actions you took from where you saw the {source} '
            'to where you saw the {goal}.'
        ),
    ],
    'survey_based_route_planning': [
        (
            "Using your knowledge of the environment's layout, list the "
            'actions that take you from where you saw the {source} '
            'to where you saw the {goal}.'
        ),
        (
            "Based on what you've seen of this space, list the actions, "
            'in order, that go from where you saw the {source} '
            'to where you saw the {goal}.'
        ),
    ],
}


def _template_bank(construct: str, template_mode: Optional[str] = None) -> list:
    """Templates for a construct; for mode-keyed banks, ``template_mode`` or the first mode."""
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
    """Number of paraphrase templates available for a construct / mode."""
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
    """False for missing or placeholder categories (``Undefined``, ``nan``, …)."""
    return cat is not None and str(cat).strip().lower() not in _BAD_CATEGORIES


_OBJ_TYPE_SPLIT = re.compile(r'(?<=[a-z])(?=[A-Z])|[-_]+')


def words_from_obj_type(obj_type: str) -> str:
    """Split a simulator ``obj-type`` into words: ``AlarmClock`` → ``Alarm Clock``."""
    text = str(obj_type or '').strip()
    if not text:
        return ''
    parts = [p for p in _OBJ_TYPE_SPLIT.split(text) if p]
    return ' '.join(parts) if parts else text


def _strip_obja_asset_prefix(name: str) -> str:
    """Drop the SPOC/Objaverse id prefix so questions say Scooter, not ObjaScooter.

    Longer prefix first: ``Objaverse_Chair`` → ``Chair``, ``ObjaScooter`` → ``Scooter``.
    """
    text = str(name).strip()
    if text:
        lower = text.lower()
        for prefix in ('objaverse', 'obja'):
            if lower.startswith(prefix) and len(text) > len(prefix):
                rest = text[len(prefix) :].lstrip('_-')
                if rest:
                    return rest
    return text


def object_type_from_id(obj_id: str, visible_or_memory: Optional[dict] = None) -> str:
    """Human-readable object name for questions.

    Prefer a real ``category`` from GT when present. Simulator placeholders such as
    ``Undefined`` (common for Objaverse assets) fall back to the id stem, with the
    ``Obja`` / ``Objaverse`` asset prefix stripped (``ObjaScooter|4|5`` → ``Scooter``).
    """
    if visible_or_memory and obj_id in visible_or_memory:
        cat = visible_or_memory[obj_id].get('category')
        if _category_usable(cat):
            cleaned = _strip_obja_asset_prefix(str(cat).strip())
            if cleaned:
                return words_from_obj_type(cleaned)
    stem = str(obj_id).split('|')[0].strip()
    cleaned = _strip_obja_asset_prefix(stem) if stem else ''
    return words_from_obj_type(cleaned or stem or str(obj_id))


def angle_to_ego_label(angle_deg: float, ahead_half_width: Optional[float] = None) -> str:
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
    if ahead_half_width is None:
        ahead_half_width = get_generation_config().directions.ahead_half_width_full_deg
    w = float(ahead_half_width)
    a = float(angle_deg) % 360.0
    if a < w or a >= 360.0 - w:
        return 'ahead of you'
    if a < 180.0 - w:
        return 'to your right'
    if a < 180.0 + w:
        return 'behind you'
    return 'to your left'


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
    margin_deg: Optional[float] = None,
    back_deg: Optional[float] = None,
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
    dcfg = get_generation_config().directions
    if margin_deg is None:
        margin_deg = dcfg.pt_boundary_margin_deg
    if back_deg is None:
        back_deg = dcfg.pt_back_deg
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


def _sector_boundaries_deg(ahead_half_width: float) -> tuple[float, ...]:
    """The four ego-sector edges (degrees) for an ahead half-width ``w``."""
    w = float(ahead_half_width) % 360.0
    return (w, (180.0 - w) % 360.0, (180.0 + w) % 360.0, (360.0 - w) % 360.0)


def angular_span_crosses_sector(
    bearing_deg: float,
    ang_width_deg: Optional[float],
    *,
    ahead_half_width: Optional[float] = None,
) -> bool:
    """True when center±half ``ang_width_deg`` touches a sector edge.

    Kept as a fallback when neither a 2D bbox nor a 3D AABB is available.
    Prefer ``bbox_edges_cross_sector`` (visible) or corner bearings (hidden).
    """
    if not get_generation_config().directions.refuse_straddling_objects:
        return False
    if ahead_half_width is None:
        ahead_half_width = get_generation_config().directions.ahead_half_width_full_deg
    if ang_width_deg is None:
        return False
    try:
        half = abs(float(ang_width_deg)) / 2.0
    except (TypeError, ValueError):
        return False
    if half <= 0.0 or math.isnan(half) or math.isinf(half):
        return False
    bearing = float(bearing_deg) % 360.0
    for boundary in _sector_boundaries_deg(ahead_half_width):
        delta = abs((boundary - bearing + 180.0) % 360.0 - 180.0)
        if delta <= half:
            return True
    return False


def bearings_cross_sector(
    bearings_deg: list[float] | tuple[float, ...],
    *,
    ahead_half_width: Optional[float] = None,
) -> bool:
    """True when the bearings do not all fall in one ego sector."""
    if not get_generation_config().directions.refuse_straddling_objects:
        return False
    if ahead_half_width is None:
        ahead_half_width = get_generation_config().directions.ahead_half_width_full_deg
    labels: set[str] = set()
    for bearing in bearings_deg:
        try:
            labels.add(
                angle_to_ego_label(float(bearing), ahead_half_width=ahead_half_width)
            )
        except (TypeError, ValueError):
            continue
        if len(labels) > 1:
            return True
    return False


def pixel_column_to_bearing_deg(
    column: float, frame_w: float, fov_h_deg: float
) -> Optional[float]:
    """Horizontal ego bearing (0 = ahead) for an image column via pinhole FOV."""
    try:
        w = float(frame_w)
        fov_h = float(fov_h_deg)
        u = float(column)
    except (TypeError, ValueError):
        return None
    if w <= 0.0 or fov_h <= 0.0 or math.isnan(w) or math.isnan(fov_h):
        return None
    fx = w / (2.0 * math.tan(math.radians(fov_h / 2.0)))
    if fx <= 0.0 or math.isnan(fx) or math.isinf(fx):
        return None
    cx = w / 2.0
    return math.degrees(math.atan2(u - cx, fx)) % 360.0


def bbox_edges_cross_sector(
    bbox,
    frame_w: float,
    fov_h_deg: float,
    *,
    ahead_half_width: Optional[float] = None,
) -> bool:
    """True when the 2D bbox left/right edges fall in different ego sectors.

    Used for objects visible in the answer frame: the visible extent must not
    straddle a sector boundary under the same ``ahead_half_width`` as the answer.
    """
    if not get_generation_config().directions.refuse_straddling_objects:
        return False
    try:
        cmin, _rmin, cmax, _rmax = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )
    except (TypeError, ValueError, IndexError):
        return False
    left = pixel_column_to_bearing_deg(cmin, frame_w, fov_h_deg)
    right = pixel_column_to_bearing_deg(cmax, frame_w, fov_h_deg)
    if left is None or right is None:
        return False
    return bearings_cross_sector(
        (left, right), ahead_half_width=ahead_half_width
    )


def local_offset_to_ego_label(
    local_xyz, *, ahead_half_width: Optional[float] = None
) -> Optional[str]:
    """MC ego label from a local (dx, dy, dz) offset."""
    try:
        x = float(local_xyz[0])
        z = float(local_xyz[2])
    except (TypeError, ValueError, IndexError):
        return None
    if ahead_half_width is None:
        ahead_half_width = get_generation_config().directions.ahead_half_width_full_deg
    bearing = bearing_deg_xz(x, z)
    if bearing is None:
        return None
    return angle_to_ego_label(bearing, ahead_half_width=ahead_half_width)


_MC_TO_REFERRING = {
    'ahead of you': 'ahead of',
    'to your right': 'to the right of',
    'behind you': 'behind',
    'to your left': 'to the left of',
}


def find_ego_edge(step: dict, obj_id: str) -> Optional[dict]:
    """The step's ``agent → obj_id`` egocentric edge, or None if the object has none."""
    for edge in step.get('edges_egocentric') or []:
        if edge.get('target') == obj_id and edge.get('source') == 'agent':
            return edge
    return None


def translated_egocentric_label(
    agent_heading: float,
    landmark_pos: dict,
    target_pos: dict,
    *,
    ahead_half_width: Optional[float] = None,
) -> Optional[str]:
    """Direction of target from landmark in the agent's yaw frame (referring only).

    Default FOV half-width: disambiguator landmarks are co-visible in-frame.
    Maps ``angle_to_ego_label`` output to landmark-relative phrases.
    """
    if ahead_half_width is None:
        ahead_half_width = get_generation_config().directions.ahead_half_width_fov_deg
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


def fov_stride_of(episode: Optional[dict]) -> int:
    """``episode_meta.fov_stride`` (or ``config.fov_stride``); default 1."""
    if not isinstance(episode, dict):
        return 1
    meta = episode.get('episode_meta') or {}
    raw = meta.get('fov_stride')
    if raw is None:
        raw = (meta.get('config') or {}).get('fov_stride')
    try:
        stride = int(raw)
    except (TypeError, ValueError):
        return 1
    return stride if stride > 1 else 1


def is_fov_step(episode: Optional[dict], step_idx) -> bool:
    """True when ``step_idx`` has object / visibility rows (stride-aligned)."""
    try:
        si = int(step_idx)
    except (TypeError, ValueError):
        return False
    return si % fov_stride_of(episode) == 0


def _episode_camera(episode: Optional[dict]) -> dict:
    """Prefer top-level ``camera``, else ``episode_meta.camera``."""
    if not isinstance(episode, dict):
        return {}
    cam = episode.get('camera')
    if isinstance(cam, dict) and cam:
        return cam
    return (episode.get('episode_meta') or {}).get('camera') or {}


def camera_size_wh(episode: Optional[dict]) -> Optional[tuple[int, int]]:
    """Return (width, height) from episode camera when present."""
    cam = _episode_camera(episode)
    try:
        w = int(cam.get('width'))
        h = int(cam.get('height'))
    except (TypeError, ValueError):
        return None
    if w > 0 and h > 0:
        return w, h
    return None


def camera_horizontal_fov_deg(episode: Optional[dict]) -> Optional[float]:
    """Horizontal FOV degrees from episode camera (derived from vertical FOV)."""
    cam = _episode_camera(episode)
    size = camera_size_wh(episode)
    if size is None:
        return None
    try:
        fov_v = float(cam.get('fov_vertical_deg') or cam.get('fov_v'))
    except (TypeError, ValueError):
        return None
    if fov_v <= 0.0:
        return None
    from cm_benchmark.utils.spatial_transformer import get_focal_length

    _fx, _fy, fov_h = get_focal_length(size[0], size[1], fov_v)
    try:
        fov_h_f = float(fov_h)
    except (TypeError, ValueError):
        return None
    if fov_h_f <= 0.0 or math.isnan(fov_h_f) or math.isinf(fov_h_f):
        return None
    return fov_h_f


def bbox_not_border_scrape(
    odata: Optional[dict],
    image_wh: Optional[tuple[int, int]],
    *,
    edge_margin_px: Optional[int] = None,
    min_extent_if_clipped: Optional[float] = None,
) -> bool:
    """False when the detection is only a thin strip along the image border.

    Example reject: bbox ``[0, 177, 25, 223]`` on a 396×224 frame — left-edge
    clip with width 25 px; a human cannot identify the category.
    When bbox or image size is missing, do not invent a reject.
    """
    border = get_generation_config().visibility.border
    if edge_margin_px is None:
        edge_margin_px = border.edge_margin_px
    if min_extent_if_clipped is None:
        min_extent_if_clipped = border.min_extent_if_clipped_px
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
    min_bbox_area: Optional[float] = None,
    min_side: Optional[float] = None,
    min_visible_pixels: Optional[float] = None,
    image_wh: Optional[tuple[int, int]] = None,
    min_extent_if_clipped: Optional[float] = None,
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
    query = get_generation_config().visibility.query
    if min_bbox_area is None:
        min_bbox_area = query.min_bbox_area
    if min_side is None:
        min_side = query.min_side
    if min_visible_pixels is None:
        min_visible_pixels = query.min_visible_pixels
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


def duplicate_category_group(
    step: dict, target_obj_id: str, episode: Optional[dict] = None
) -> list[str]:
    """Visible ids confusable with the target (includes target)."""
    vis = step.get('visible_objects') or {}
    if target_obj_id not in vis:
        return []
    from cm_benchmark.generation.synsets import objects_are_confusable

    ep = episode or {'object_catalog': {}, 'steps': [step]}
    return [
        oid
        for oid, odata in vis.items()
        if objects_are_confusable(ep, target_obj_id, oid, vis.get(target_obj_id), odata)
    ]


def format_disambiguator_phrase(info: dict) -> str:
    """Leading-space phrase for templates, e.g. `` close to the Bread``."""
    relation = str(info.get('relation') or '').strip()
    landmark = str(info.get('landmark_type') or '').strip()
    return f' {relation} the {landmark}' if relation and landmark else ''


def find_disambiguator(
    step: dict,
    target_obj_id: str,
    duplicate_group: list[str],
    agent_pose: Optional[dict] = None,
    *,
    min_margin_m: Optional[float] = None,
    min_margin_ratio: Optional[float] = None,
    model: Any = None,
) -> Optional[dict]:
    """Landmark + relation that uniquely identifies target among duplicate_group.

    Proximity is accepted only when the margin is large enough to be obvious in
    the image (absolute meters AND relative ratio). Landmarks must themselves
    pass ``fov_metrics_ok``. Returns None if nothing qualifies — caller skips.
    """
    prox = get_generation_config().disambiguation.active_proximity()
    if min_margin_m is None:
        min_margin_m = prox.margin_m if prox.margin_m is not None else 0.0
    if min_margin_ratio is None:
        min_margin_ratio = prox.ratio if prox.ratio is not None else 0.0
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
    return None


def resolve_referring_disambiguator(
    step: dict,
    target_obj_id: str,
    agent_pose: Optional[dict] = None,
    *,
    model: Any = None,
    episode: Optional[dict] = None,
) -> Optional[str]:
    """Phrase for templates, or None to skip the candidate.

    - Unique category in the step → ``''`` (no phrase; not ambiguous).
    - Duplicate category + proximity or colour-dot mark → leading-space phrase.
    - Duplicate category + neither → ``None`` (exclude from queries).

    Landmark-relative directions are never used. ``agent_pose`` is unused
    (kept so existing call sites stay valid).
    """
    del agent_pose
    from cm_benchmark.generation.disambiguation import resolve_named_objects

    si = int(step.get('step') or 0)
    ep = dict(episode) if isinstance(episode, dict) else {}
    ep['steps'] = [dict(step, step=si)]
    names = resolve_named_objects(ep, [si], [target_obj_id], model=model)
    if names is None or target_obj_id not in names:
        return None
    return names[target_obj_id].phrase


def step_by_index(episode: dict, step_idx: int) -> Optional[dict]:
    """The episode step whose ``step`` equals ``step_idx``, or None."""
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
    pose_a: dict,
    pose_b: dict,
    pos_tol: Optional[float] = None,
    heading_tol_deg: Optional[float] = None,
) -> bool:
    """Real move check from actual positions — never trust action count alone."""
    su = get_generation_config().spatial_updating
    if pos_tol is None:
        pos_tol = su.min_pos_change_m
    if heading_tol_deg is None:
        heading_tol_deg = su.min_heading_change_deg
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
    margin_deg: Optional[float] = None,
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


def __getattr__(name: str):
    """Former module-level quality constants, read from the active config."""
    return legacy_constant(name)
