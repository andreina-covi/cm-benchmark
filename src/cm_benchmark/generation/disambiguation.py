"""Whole-item object naming: proximity, colour-dot marks, no landmark directions.

Duplicate detection uses every shown frame. Proximity uses 3D AABB gap on the
floor plane, then a 2D pixel-bbox check. Marks are the fallback (and the
``all_marks`` ablation) and sit on the exported mask point. A required mark
with no point drops the item. Questions never contain landmark-relative
directions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from cm_benchmark.generation.config import get_generation_config
from cm_benchmark.generation.constructs import (
    camera_size_wh,
    fov_metrics_ok,
    is_fov_step,
    object_type_from_id,
    step_by_index,
    xyz_as_dict,
)
from cm_benchmark.generation.step_derived import raw_visible_frames
from cm_benchmark.generation.synsets import (
    confusable_components,
    confusable_group_of,
    group_is_unique,
    objects_are_confusable,
    type_of,
)
from cm_benchmark.generator.visibility_filters import visibility_model_from_episode

MARK_RGB = {
    'magenta': (255, 0, 200),
    'cyan': (0, 220, 255),
    'lime': (80, 220, 36),
    'yellow': (255, 220, 0),
    'orange': (255, 128, 0),
    'blue': (16, 96, 255),
    'purple': (168, 50, 216),
}


@dataclass
class NameResult:
    """How one object is referred to in the item."""

    object_id: str
    object_type: str
    phrase: str
    mode: str  # none | proximity | mark
    landmark_id: Optional[str] = None
    marks: dict[str, str] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return f'{self.object_type}{self.phrase}'


def aabb_minmax_gap_xz(min_a, max_a, min_b, max_b) -> Optional[float]:
    """Closest-point gap (metres) between two floor-plane AABB min/max corners."""
    a0, a1 = _xz(min_a), _xz(max_a)
    b0, b1 = _xz(min_b), _xz(max_b)
    if None in (a0, a1, b0, b1):
        return None
    min_ax, max_ax = min(a0[0], a1[0]), max(a0[0], a1[0])
    min_az, max_az = min(a0[1], a1[1]), max(a0[1], a1[1])
    min_bx, max_bx = min(b0[0], b1[0]), max(b0[0], b1[0])
    min_bz, max_bz = min(b0[1], b1[1]), max(b0[1], b1[1])
    dx = max(min_ax - max_bx, min_bx - max_ax, 0.0)
    dz = max(min_az - max_bz, min_bz - max_az, 0.0)
    return math.hypot(dx, dz)


def aabb_gap_xz(
    center_a,
    size_a,
    center_b,
    size_b,
) -> Optional[float]:
    """Closest-point gap (metres) between two floor-plane AABBs.

    ``center`` is (x, z) or an xyz dict/list. ``size`` is full (sx, sz) or
    (sx, sy, sz); missing size is treated as a point. Overlap → 0.
    Prefer ``aabb_minmax_gap_xz`` when catalog min/max boxes exist.
    """
    ca = _xz(center_a)
    cb = _xz(center_b)
    if ca is None or cb is None:
        return None
    sa = _half_xz(size_a)
    sb = _half_xz(size_b)
    min_a = (ca[0] - sa[0], ca[1] - sa[1])
    max_a = (ca[0] + sa[0], ca[1] + sa[1])
    min_b = (cb[0] - sb[0], cb[1] - sb[1])
    max_b = (cb[0] + sb[0], cb[1] + sb[1])
    return aabb_minmax_gap_xz(min_a, max_a, min_b, max_b)


def pixel_bbox_gap(box_a, box_b) -> Optional[float]:
    """Closest-point gap in pixels between two ``(cmin, rmin, cmax, rmax)`` boxes."""
    a = _box4(box_a)
    b = _box4(box_b)
    if a is None or b is None:
        return None
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def format_mark_phrase(colour: str) -> str:
    """Leading-space phrase for templates: `` marked with a magenta dot``."""
    return f' marked with a {colour} dot'


def format_proximity_phrase(landmark_type: str) -> str:
    """Leading-space phrase: `` close to the Bread``."""
    return f' close to the {landmark_type}'


def active_margin_m(episode: Optional[dict] = None) -> tuple[float, Optional[float]]:
    """``(margin_m, ratio_or_None)`` for the active proximity setting."""
    cfg = get_generation_config().disambiguation
    setting = cfg.active_proximity()
    if cfg.proximity_setting == 'vsi':
        area = _room_area_m2(episode)
        margin = (
            setting.margin_m_small
            if area is not None and setting.room_area_split_m2 is not None
            and area <= setting.room_area_split_m2
            else setting.margin_m_large
        )
        return float(margin if margin is not None else setting.margin_m_large), None
    margin = float(setting.margin_m or 0.0)
    ratio = None if setting.ratio is None else float(setting.ratio)
    return margin, ratio


def _mask_pixel_count(odata: Optional[dict]) -> float:
    """Mask pixels on a detection. Listed without a count still counts as visible."""
    if not isinstance(odata, dict):
        return 0.0
    pix = odata.get('visible_pixels')
    if pix is None:
        return 1.0
    try:
        return float(pix)
    except (TypeError, ValueError):
        return 0.0


def objects_with_mask_pixels(
    episode: dict, frame_steps: list[int]
) -> dict[str, dict]:
    """Objects with mask pixels > 0 in any shown frame (raw detections)."""
    out: dict[str, dict] = {}
    for si in frame_steps:
        if not is_fov_step(episode, si):
            continue
        step = step_by_index(episode, int(si))
        if step is None:
            continue
        raw = raw_visible_frames(step) or (step.get('visible_objects') or {})
        for oid, odata in raw.items():
            if isinstance(odata, dict) and _mask_pixel_count(odata) > 0:
                out.setdefault(str(oid), odata)
    return out


def passing_objects_in_frames(
    episode: dict, frame_steps: list[int]
) -> dict[int, dict[str, dict]]:
    """``step → {obj_id: odata}`` for objects that pass the item-time visibility filter.

    After ``apply_question_visibility_to_episode``, membership in
    ``visible_objects`` is that filter.
    """
    out: dict[int, dict[str, dict]] = {}
    for si in frame_steps:
        if not is_fov_step(episode, si):
            continue
        step = step_by_index(episode, int(si))
        if step is None:
            continue
        vis = {
            oid: odata
            for oid, odata in (step.get('visible_objects') or {}).items()
            if isinstance(odata, dict)
        }
        out[int(si)] = vis
    return out


def episode_mark_colours(
    episode: dict,
    frame_steps: Optional[list[int]] = None,
    also_ids: Optional[list[str]] = None,
) -> dict[str, str]:
    """Stable ``objectId → colour`` for duplicated types in the item's frames.

    Instances in ``frame_steps`` are coloured, plus ``also_ids`` (confusable
    endpoints that are not in those frames). When frames are omitted, every
    FOV step is used. Two items may reuse a palette colour
    for different objects. A sibling passed in ``also_ids`` is coloured even
    when it is outside the shown frames, so the shown endpoint can be marked.
    Types with more in-frame instances than palette colours get no assignment.
    Assignment is sorted by objectId so it is deterministic.
    """
    palette = list(get_generation_config().disambiguation.mark_palette)
    if frame_steps is None:
        steps = [
            step
            for step in episode.get('steps') or []
            if is_fov_step(episode, step.get('step'))
        ]
    else:
        steps = []
        for si in frame_steps:
            step = step_by_index(episode, int(si))
            if step is not None and is_fov_step(episode, si):
                steps.append(step)
    frame_idx = [int(step['step']) for step in steps if step.get('step') is not None]
    pixel_ids = set(objects_with_mask_pixels(episode, frame_idx))
    passing_ids: set[str] = set()
    for step in steps:
        passing_ids.update(step.get('visible_objects') or {})
    seen = set(pixel_ids)
    for oid in also_ids or []:
        if oid:
            seen.add(oid)
    colours: dict[str, str] = {}
    extra = {oid for oid in (also_ids or []) if oid}
    for group in confusable_components(episode, seen):
        if len(group) < 2:
            continue
        drawable = sorted(
            oid for oid in group if oid in passing_ids or oid in extra
        )
        if not drawable:
            drawable = sorted(group)
        if len(drawable) > len(palette):
            continue
        for i, oid in enumerate(drawable):
            colours[oid] = palette[i]
    return colours


def resolve_named_objects(
    episode: dict,
    frame_steps: list[int],
    object_ids: list[str],
    *,
    model: Any = None,
    also_ids: Optional[list[str]] = None,
) -> Optional[dict[str, NameResult]]:
    """Name every ``object_id`` across ``frame_steps``, or None to skip.

    Unique types → ``none``. Duplicates try proximity (unless ``all_marks``),
    then a colour-dot mark. One failure skips the whole candidate.
    """
    ids = [oid for oid in object_ids if oid]
    if not ids:
        return {}
    frames = passing_objects_in_frames(episode, frame_steps)
    pixel_pool = set(objects_with_mask_pixels(episode, frame_steps))
    extra = [oid for oid in (also_ids or []) if oid]
    pixel_pool.update(extra)
    if model is None:
        model = visibility_model_from_episode(episode)
    colours = episode_mark_colours(episode, frame_steps, also_ids=extra)
    cfg = get_generation_config().disambiguation
    out: dict[str, NameResult] = {}
    for oid in ids:
        typ = _type_of(episode, oid, frames)
        if typ is None:
            return None
        siblings = [x for x in confusable_group_of(episode, oid, pixel_pool) if x != oid]
        if not siblings:
            out[oid] = NameResult(oid, typ, '', 'none')
            continue
        drawable = [
            sib
            for sib in siblings
            if any(sib in vis for vis in frames.values()) and sib in colours
        ]
        result = None
        if not cfg.all_marks:
            result = _try_proximity(
                episode, frames, oid, typ, siblings, model=model
            )
        if result is None:
            result = _try_mark(oid, typ, drawable, colours)
        if result is not None and result.mode == 'mark':
            if not _shown_marks_have_points(frames, result.marks, oid):
                result = None
        if result is None:
            return None
        out[oid] = result
    return out


def aggregate_disambiguation(results: dict[str, NameResult]) -> dict[str, Any]:
    """Item-level fields from per-object results."""
    modes = {r.mode for r in results.values()}
    if 'mark' in modes:
        mode = 'mark'
    elif 'proximity' in modes:
        mode = 'proximity'
    else:
        mode = 'none'
    landmark = next((r.landmark_id for r in results.values() if r.landmark_id), None)
    marks: dict[str, str] = {}
    for r in results.values():
        marks.update(r.marks)
    return {
        'disambiguation': mode,
        'disambiguation_landmark': landmark,
        'marks': marks,
    }


def draw_marked_frames(
    episode: dict,
    image_paths: list[str],
    frame_steps: list[int],
    marks: dict[str, str],
    dest_dir: Path,
) -> list[str]:
    """Write marked copies of ``image_paths``; return the copy paths.

    Marks every ``objectId`` in ``marks`` that passes the visibility filter
    on that frame. Originals are not modified.
    """
    from PIL import Image, ImageDraw

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    frames = passing_objects_in_frames(episode, frame_steps)
    image_wh = camera_size_wh(episode)
    out: list[str] = []
    for i, (path, si) in enumerate(zip(image_paths, frame_steps)):
        vis = frames.get(int(si)) or {}
        src = Path(path)
        dst = dest_dir / f'{int(si)}_{i}_{src.name or "frame.png"}'
        try:
            im = Image.open(src).convert('RGB')
        except OSError:
            out.append(path)
            continue
        draw = ImageDraw.Draw(im)
        w, h = im.size
        for oid, colour in marks.items():
            odata = vis.get(oid)
            if not isinstance(odata, dict):
                continue
            pt = _mark_xy(odata)
            if pt is None:
                continue
            xy = _scale_px(pt, image_wh or (w, h), (w, h))
            _draw_dot(draw, xy, MARK_RGB.get(colour, (255, 0, 200)), w)
        im.save(dst)
        out.append(str(dst))
    return out


def _xz(pos) -> Optional[tuple[float, float]]:
    d = xyz_as_dict(pos) if not (isinstance(pos, (tuple, list)) and len(pos) == 2) else None
    if d is not None:
        return (d['x'], d['z'])
    if isinstance(pos, (tuple, list)) and len(pos) >= 2:
        try:
            if len(pos) >= 3:
                return (float(pos[0]), float(pos[2]))
            return (float(pos[0]), float(pos[1]))
        except (TypeError, ValueError):
            return None
    return None


def _half_xz(size) -> tuple[float, float]:
    if size is None:
        return (0.0, 0.0)
    try:
        if isinstance(size, dict):
            sx = float(size.get('x') or size.get('size-x') or 0.0)
            sz = float(size.get('z') or size.get('size-z') or 0.0)
        else:
            sx = float(size[0])
            sz = float(size[2] if len(size) >= 3 else size[1])
        if sx != sx or sz != sz:
            return (0.0, 0.0)
        return (abs(sx) / 2.0, abs(sz) / 2.0)
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0)


def _box4(bbox) -> Optional[tuple[float, float, float, float]]:
    if not bbox or len(bbox) < 4:
        return None
    try:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None


def _center_and_size(episode: dict, oid: str, odata: dict) -> tuple[Any, Any]:
    catalog = (episode.get('object_catalog') or {}).get(oid) or {}
    center = odata.get('bbox_center') or catalog.get('bbox_center') or odata.get('position')
    size = odata.get('size') or catalog.get('size')
    return center, size


def _aabb_minmax(episode: dict, oid: str, odata: dict):
    """``(aabb_min, aabb_max)`` xyz, or None pair if the catalog has no box."""
    catalog = (episode.get('object_catalog') or {}).get(oid) or {}
    amin = odata.get('aabb_min') or catalog.get('aabb_min')
    amax = odata.get('aabb_max') or catalog.get('aabb_max')
    if amin is not None and amax is not None:
        return amin, amax
    center, size = _center_and_size(episode, oid, odata)
    ca = _xz(center)
    if ca is None:
        return None, None
    hx, hz = _half_xz(size)
    return (ca[0] - hx, 0.0, ca[1] - hz), (ca[0] + hx, 0.0, ca[1] + hz)


def object_aabb_gap_m(episode: dict, oid_a: str, data_a: dict, oid_b: str, data_b: dict) -> Optional[float]:
    """Floor-plane gap between two objects using catalog AABB min/max when present."""
    min_a, max_a = _aabb_minmax(episode, oid_a, data_a)
    min_b, max_b = _aabb_minmax(episode, oid_b, data_b)
    return aabb_minmax_gap_xz(min_a, max_a, min_b, max_b)


def _type_of(episode: dict, oid: str, frames: dict[int, dict[str, dict]]) -> Optional[str]:
    for vis in frames.values():
        if oid in vis:
            return type_of(episode, oid, vis[oid])
    for step in episode.get('steps') or []:
        vis = step.get('visible_objects') or {}
        if oid in vis:
            return type_of(episode, oid, vis[oid])
    return type_of(episode, oid)


def _oids_in_frames(frames: dict[int, dict[str, dict]]) -> set[str]:
    return {oid for vis in frames.values() for oid in vis}


def _room_area_m2(episode: Optional[dict]) -> Optional[float]:
    """Area of the agent's current room from ``world_layout.regions``."""
    if not isinstance(episode, dict):
        return None
    regions = (episode.get('world_layout') or {}).get('regions') or []
    rid = None
    for step in episode.get('steps') or []:
        if step.get('current_room'):
            rid = step['current_room']
            break
    if rid is None:
        for rec in (episode.get('object_catalog') or {}).values():
            if rec.get('room_id'):
                rid = rec['room_id']
                break
    for region in regions:
        if rid is not None and region.get('region_id') != rid:
            continue
        area = region.get('area')
        if area is not None:
            try:
                return float(area)
            except (TypeError, ValueError):
                continue
        if rid is not None:
            break
    return None


def _try_proximity(
    episode: dict,
    frames: dict[int, dict[str, dict]],
    target_id: str,
    target_type: str,
    siblings: list[str],
    *,
    model: Any,
) -> Optional[NameResult]:
    margin_m, ratio = active_margin_m(episode)
    pool = _oids_in_frames(frames)
    # Landmark must not be confusable with anything else in the shown frames.
    candidates: list[str] = []
    seen: set[str] = set()
    for vis in frames.values():
        for oid, odata in vis.items():
            if oid in seen or oid == target_id or oid in siblings:
                continue
            if objects_are_confusable(episode, oid, target_id, odata, None):
                continue
            if not group_is_unique(episode, oid, pool):
                continue
            if not fov_metrics_ok(odata, model=model):
                continue
            seen.add(oid)
            candidates.append(oid)
    # Target and landmark must be co-visible in at least one shown frame.
    best = None
    for lid in candidates:
        co_vis = False
        for vis in frames.values():
            if target_id in vis and lid in vis:
                co_vis = True
                break
        if not co_vis:
            continue
        # Target closer than every sibling (any frame they appear in) by margin/ratio.
        # Use the first co-visible frame's poses for the 3D gap.
        t_data = next(vis[target_id] for vis in frames.values() if target_id in vis)
        l_data = next(vis[lid] for vis in frames.values() if lid in vis and target_id in vis)
        d_target = object_aabb_gap_m(episode, target_id, t_data, lid, l_data)
        if d_target is None or d_target <= 1e-6:
            continue
        sib_gaps = []
        for sib in siblings:
            s_data = None
            for vis in frames.values():
                if sib in vis:
                    s_data = vis[sib]
                    break
            if s_data is None:
                # A sibling outside the shown frames cannot be checked, so
                # proximity is not a safe name. The caller falls through to a mark.
                return None
            g = object_aabb_gap_m(episode, sib, s_data, lid, l_data)
            if g is not None:
                sib_gaps.append(g)
        if not sib_gaps:
            continue
        d_sib = min(sib_gaps)
        if d_sib - d_target < margin_m:
            continue
        if ratio is not None and d_sib < ratio * d_target:
            continue
        if not _legible_2d(frames, target_id, lid, siblings, episode):
            continue
        gap = d_sib - d_target
        if best is None or gap > best[1]:
            landmark_type = _type_of(episode, lid, frames) or object_type_from_id(lid)
            best = (
                NameResult(
                    target_id,
                    target_type,
                    format_proximity_phrase(landmark_type),
                    'proximity',
                    landmark_id=lid,
                ),
                gap,
            )
    return None if best is None else best[0]


def _legible_2d(
    frames: dict[int, dict[str, dict]],
    target_id: str,
    landmark_id: str,
    siblings: list[str],
    episode: dict,
) -> bool:
    """True if every co-visible (target, landmark, sibling) frame is 2D-obvious."""
    cfg = get_generation_config().disambiguation
    image_wh = camera_size_wh(episode)
    width = float(image_wh[0]) if image_wh else float(get_generation_config().episode_gt.frame_w)
    min_extra = cfg.min_gap_2d_frac * width
    for vis in frames.values():
        if target_id not in vis or landmark_id not in vis:
            continue
        t_box = vis[target_id].get('bbox')
        l_box = vis[landmark_id].get('bbox')
        if _box4(t_box) is None or _box4(l_box) is None:
            continue
        t_gap = pixel_bbox_gap(t_box, l_box)
        if t_gap is None:
            continue
        for sib in siblings:
            if sib not in vis:
                continue
            s_box = vis[sib].get('bbox')
            if _box4(s_box) is None:
                continue
            s_gap = pixel_bbox_gap(s_box, l_box)
            if s_gap is None:
                continue
            if s_gap < cfg.ratio_2d * t_gap:
                return False
            if s_gap - t_gap < min_extra:
                return False
    return True


def _try_mark(
    target_id: str,
    target_type: str,
    siblings: list[str],
    colours: dict[str, str],
) -> Optional[NameResult]:
    if target_id not in colours:
        return None
    colour = colours[target_id]
    marks = {target_id: colour}
    for oid in siblings:
        if oid in colours:
            marks[oid] = colours[oid]
    return NameResult(
        target_id,
        target_type,
        format_mark_phrase(colour),
        'mark',
        marks=marks,
    )


def _mark_xy(odata: dict) -> Optional[tuple[float, float]]:
    """Exported mask point (``mark-x`` / ``mark-y``), stored as ``mark_xy``."""
    pt = odata.get('mark_xy')
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return None
    try:
        x, y = float(pt[0]), float(pt[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _scale_px(
    pt: tuple[float, float],
    frame_wh: tuple[int, int],
    image_wh: tuple[int, int],
) -> tuple[int, int]:
    """Scale a detection-frame pixel onto the opened image."""
    x, y = pt
    fw, fh = float(frame_wh[0]), float(frame_wh[1])
    w, h = image_wh
    if fw > 1 and fh > 1 and (abs(fw - w) > 0.5 or abs(fh - h) > 0.5):
        x = x * w / fw
        y = y * h / fh
    return int(round(x)), int(round(min(y, h - 1)))


def _shown_marks_have_points(
    frames: dict[int, dict[str, dict]],
    marks: dict[str, str],
    target_id: str,
) -> bool:
    """The target, and every marked sibling on screen, has a mask point."""
    saw_target = False
    for vis in frames.values():
        for mid in marks:
            if mid not in vis:
                continue
            if _mark_xy(vis[mid]) is None:
                return False
            if mid == target_id:
                saw_target = True
    return saw_target


def _draw_dot(draw, xy: tuple[int, int], rgb: tuple[int, int, int], image_w: int) -> None:
    r = max(4, int(round(0.02 * image_w)))
    x, y = xy
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    outline = (0, 0, 0) if lum > 140 else (255, 255, 255)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=rgb, outline=outline, width=max(1, r // 4))
