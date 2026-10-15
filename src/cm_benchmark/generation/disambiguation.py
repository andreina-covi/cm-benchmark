"""Whole-item object naming: proximity, colour-dot marks, no landmark directions.

Duplicate detection uses every shown frame. Proximity uses 3D AABB gap on the
floor plane, then a 2D pixel-bbox check. Marks are the fallback (and the
``all_marks`` ablation). Questions never contain landmark-relative directions.
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
    object_type_from_id,
    step_by_index,
    xyz_as_dict,
)
from cm_benchmark.generator.visibility_filters import visibility_model_from_episode

MARK_RGB = {
    'magenta': (255, 0, 200),
    'cyan': (0, 220, 255),
    'lime': (80, 220, 36),
    'yellow': (255, 220, 0),
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


def aabb_gap_xz(
    center_a,
    size_a,
    center_b,
    size_b,
) -> Optional[float]:
    """Closest-point gap (metres) between two floor-plane AABBs.

    ``center`` is (x, z) or an xyz dict/list. ``size`` is full (sx, sz) or
    (sx, sy, sz); missing size is treated as a point. Overlap → 0.
    """
    ca = _xz(center_a)
    cb = _xz(center_b)
    if ca is None or cb is None:
        return None
    sa = _half_xz(size_a)
    sb = _half_xz(size_b)
    min_ax, max_ax = ca[0] - sa[0], ca[0] + sa[0]
    min_az, max_az = ca[1] - sa[1], ca[1] + sa[1]
    min_bx, max_bx = cb[0] - sb[0], cb[0] + sb[0]
    min_bz, max_bz = cb[1] - sb[1], cb[1] + sb[1]
    dx = max(min_ax - max_bx, min_bx - max_ax, 0.0)
    dz = max(min_az - max_bz, min_bz - max_az, 0.0)
    return math.hypot(dx, dz)


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
        # world_layout.regions have no polygon / area — use the large-room branch.
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


def passing_objects_in_frames(
    episode: dict, frame_steps: list[int]
) -> dict[int, dict[str, dict]]:
    """``step → {obj_id: odata}`` for objects that pass the item-time visibility filter.

    After ``apply_question_visibility_to_episode``, membership in
    ``visible_objects`` is that filter.
    """
    out: dict[int, dict[str, dict]] = {}
    for si in frame_steps:
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


def episode_mark_colours(episode: dict) -> dict[str, str]:
    """Stable ``objectId → colour`` for every duplicated type in the episode.

    Types with more instances than palette colours get no assignment (mark
    unavailable). Assignment is sorted by objectId so it is deterministic.
    """
    palette = list(get_generation_config().disambiguation.mark_palette)
    by_type: dict[str, set[str]] = {}
    for step in episode.get('steps') or []:
        for oid, odata in (step.get('visible_objects') or {}).items():
            typ = object_type_from_id(oid, {oid: odata})
            by_type.setdefault(typ, set()).add(oid)
    colours: dict[str, str] = {}
    for typ, ids in by_type.items():
        ordered = sorted(ids)
        if len(ordered) < 2 or len(ordered) > len(palette):
            continue
        for i, oid in enumerate(ordered):
            colours[oid] = palette[i]
    return colours


def resolve_named_objects(
    episode: dict,
    frame_steps: list[int],
    object_ids: list[str],
    *,
    model: Any = None,
) -> Optional[dict[str, NameResult]]:
    """Name every ``object_id`` across ``frame_steps``, or None to skip.

    Unique types → ``none``. Duplicates try proximity (unless ``all_marks``),
    then a colour-dot mark. One failure skips the whole candidate.
    """
    ids = [oid for oid in object_ids if oid]
    if not ids:
        return {}
    frames = passing_objects_in_frames(episode, frame_steps)
    by_type = _types_in_frames(episode, frames)
    if model is None:
        model = visibility_model_from_episode(episode)
    colours = episode_mark_colours(episode)
    cfg = get_generation_config().disambiguation
    out: dict[str, NameResult] = {}
    for oid in ids:
        typ = _type_of(episode, oid, frames)
        if typ is None:
            return None
        siblings = [x for x in by_type.get(typ, ()) if x != oid]
        if not siblings:
            out[oid] = NameResult(oid, typ, '', 'none')
            continue
        result = None
        if not cfg.all_marks:
            result = _try_proximity(
                episode, frames, oid, typ, siblings, model=model
            )
        if result is None:
            result = _try_mark(oid, typ, siblings, colours)
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
    for path, si in zip(image_paths, frame_steps):
        vis = frames.get(int(si)) or {}
        src = Path(path)
        dst = dest_dir / src.name
        try:
            im = Image.open(src).convert('RGB')
        except OSError:
            out.append(path)
            continue
        draw = ImageDraw.Draw(im)
        w, h = im.size
        for oid, colour in marks.items():
            odata = vis.get(oid)
            if odata is None:
                continue
            box = _box4(odata.get('bbox'))
            if box is None:
                continue
            xy = _dot_xy(box, image_wh or (w, h), (w, h))
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


def _type_of(episode: dict, oid: str, frames: dict[int, dict[str, dict]]) -> Optional[str]:
    for vis in frames.values():
        if oid in vis:
            return object_type_from_id(oid, vis)
    for step in episode.get('steps') or []:
        vis = step.get('visible_objects') or {}
        if oid in vis:
            return object_type_from_id(oid, vis)
    return object_type_from_id(oid)


def _types_in_frames(
    episode: dict, frames: dict[int, dict[str, dict]]
) -> dict[str, set[str]]:
    by_type: dict[str, set[str]] = {}
    for vis in frames.values():
        for oid, odata in vis.items():
            typ = object_type_from_id(oid, {oid: odata})
            by_type.setdefault(typ, set()).add(oid)
    return by_type


def _room_area_m2(episode: Optional[dict]) -> Optional[float]:
    """None — collector regions have no polygon / area (vsi uses 0.30 m)."""
    del episode
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
    type_counts = _types_in_frames(episode, frames)
    # Landmark type must be unique across all shown frames.
    candidates: list[str] = []
    seen: set[str] = set()
    for vis in frames.values():
        for oid, odata in vis.items():
            if oid in seen or oid == target_id or oid in siblings:
                continue
            typ = object_type_from_id(oid, {oid: odata})
            if len(type_counts.get(typ, ())) != 1:
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
        tc, ts = _center_and_size(episode, target_id, t_data)
        lc, ls = _center_and_size(episode, lid, l_data)
        d_target = aabb_gap_xz(tc, ts, lc, ls)
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
                continue
            sc, ss = _center_and_size(episode, sib, s_data)
            g = aabb_gap_xz(sc, ss, lc, ls)
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
    group = [target_id, *siblings]
    if any(oid not in colours for oid in group):
        return None
    colour = colours[target_id]
    marks = {oid: colours[oid] for oid in group}
    return NameResult(
        target_id,
        target_type,
        format_mark_phrase(colour),
        'mark',
        marks=marks,
    )


def _dot_xy(
    box: tuple[float, float, float, float],
    frame_wh: tuple[int, int],
    image_wh: tuple[int, int],
) -> tuple[int, int]:
    cmin, rmin, cmax, rmax = box
    x = (cmin + cmax) / 2.0
    # Inside the bbox, near the top-centre. Clipped at the top → visible centre.
    if rmin <= 1:
        y = (rmin + rmax) / 2.0
    else:
        y = rmin + 0.18 * max(rmax - rmin, 1.0)
    fw, fh = float(frame_wh[0]), float(frame_wh[1])
    w, h = image_wh
    if fw > 1 and fh > 1 and (abs(fw - w) > 0.5 or abs(fh - h) > 0.5):
        x = x * w / fw
        y = y * h / fh
    return int(round(x)), int(round(y))


def _draw_dot(draw, xy: tuple[int, int], rgb: tuple[int, int, int], image_w: int) -> None:
    r = max(4, int(round(0.02 * image_w)))
    x, y = xy
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    outline = (0, 0, 0) if lum > 140 else (255, 255, 255)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=rgb, outline=outline, width=max(1, r // 4))
