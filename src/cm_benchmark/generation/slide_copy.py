"""Slide/review copy for class-4 items. No planner or nav_graph imports.

PIL overlays put a letter marker on the 2D bbox center of source/goal in that
still (no legend panel). The walk itself is text-only: arrow glyphs in the
side panel, nothing painted on the photo.
The pptx builder loads this file by path so the generation stack is not imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import colorsys
import re

# Egocentric collected actions → arrow glyph in the side-panel walk summary.
_ACTION_DIR = {
    'move_ahead': 'up',
    'move_forward': 'up',
    'move_back': 'down',
    'move_backward': 'down',
    'rotate_left': 'left',
    'turn_left': 'left',
    'rotate_right': 'right',
    'turn_right': 'right',
}
_DIR_GLYPH = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
ACTION_LEGEND = '↑ ahead · ↓ back · ← rotate left · → rotate right'


def action_parts(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.replace('->', '→')
        return [p.strip() for p in text.split('→') if p.strip()]
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for entry in raw:
            if isinstance(entry, dict):
                act = entry.get('action')
                if act:
                    out.append(str(act))
            elif entry is not None:
                out.append(str(entry))
        return out
    return [str(raw)]


def class4_pair_length(item: dict) -> tuple[int, int, int]:
    """Sort key for slide examples: longer stored walks first.

    Uses hop_count (nav-graph hops on the stored path), then action count,
    then path-node count. Does not change generation gates.
    """
    ctx = item.get('context') if isinstance(item.get('context'), dict) else item
    if not isinstance(ctx, dict):
        ctx = {}
    nodes = [n for n in (ctx.get('path_nodes') or []) if n]
    hops = ctx.get('hop_count')
    if hops is None:
        hops = max(0, len(nodes) - 1) if nodes else 0
    actions = action_parts(ctx.get('action_sequence') or item.get('answer'))
    try:
        hop_n = int(hops)
    except (TypeError, ValueError):
        hop_n = max(0, len(nodes) - 1)
    return (hop_n, len(actions), len(nodes))


def action_direction(name: str) -> Optional[str]:
    key = str(name).strip().lower().replace('-', '_')
    return _ACTION_DIR.get(key)


def action_glyphs(raw) -> list[str]:
    glyphs: list[str] = []
    for part in action_parts(raw):
        direction = action_direction(part)
        glyphs.append(_DIR_GLYPH[direction] if direction else part)
    return glyphs


def endpoint_glyph(name: str, *, fallback: str = '?') -> str:
    """1–3 letters for the on-image marker (Chair→Cha, Dining Table→DT)."""
    stop = {'the', 'a', 'an', 'of', 'to', 'on', 'in', 'near', 'close', 'and'}
    words = [
        w for w in re.findall(r'[A-Za-z]+', name or '') if w.lower() not in stop
    ]
    if len(words) >= 2:
        return ''.join(w[0] for w in words[:3]).upper()
    if words:
        stem = words[0]
        if len(stem) <= 2:
            return stem.upper()
        return stem[:3].capitalize()
    fb = (fallback or '?').strip()
    return fb[:3] if fb else '?'


def compact_glyph_walk(raw, *, max_show: int = 10) -> str:
    glyphs = action_glyphs(raw)
    if not glyphs:
        return '—'
    if len(glyphs) <= max_show:
        return '  '.join(glyphs)
    hidden = len(glyphs) - 7
    return f"{'  '.join(glyphs[:4])}  …{hidden}…  {'  '.join(glyphs[-3:])}"


def class4_slide_panel(
    construct: str,
    context: Optional[dict] = None,
    *,
    answer: Optional[str] = None,
) -> dict[str, str]:
    """Human-readable class-4 summary: source, goal, action walk, scoring."""
    ctx = context or {}
    src = ctx.get('source') or '?'
    goal = ctx.get('goal') or '?'
    nodes = ctx.get('path_nodes') or []
    hops = ctx.get('hop_count')
    if hops is None and len(nodes) >= 2:
        hops = len(nodes) - 1
    actions = ctx.get('action_sequence') or answer
    walk = compact_glyph_walk(actions)
    hop_bit = f'{hops} hops' if hops is not None else 'path'
    if construct == 'survey_based_route_planning':
        return {
            'task': f'Plan (never walked): {src}  →  {goal}',
            'graph': (
                'Evidence: viewed but never walked · scored on viewed edges '
                f'({hop_bit} on the stored viewed-edge path)'
            ),
            'path': f'Stored walk: {walk}',
            'actions': 'Letter marker at the object bbox center on that still.',
            'scoring': (
                'Score: success = near the goal; validity = stay on viewed edges '
                'and use at least one unwalked hop; efficiency = SPL vs viewed shortest path.'
            ),
        }
    return {
        'task': f'Retrace (walked): {src}  →  {goal}',
        'graph': f'Scored on traversed (experienced) edges · {hop_bit}',
        'path': f'Stored walk: {walk}',
        'actions': 'Letter marker at the object bbox center on that still.',
        'scoring': (
            'Score: success = near the goal; validity = stay on walked edges; '
            'efficiency = SPL. Headline = valid-success + route_efficiency.'
        ),
    }


def overlay_class4_frames(
    image_paths: list[str],
    *,
    roles: Optional[list[str]] = None,
    source_name: str = 'source',
    goal_name: str = 'goal',
    source_mark: Optional[dict] = None,
    goal_mark: Optional[dict] = None,
    out_dir: Path,
) -> list[str]:
    """Write one letter marker per endpoint at its bbox center (no legend)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(image_paths)
    labels = list(roles or [])
    written: list[str] = []
    for i, src in enumerate(image_paths):
        dest = out_dir / f'frame_{i:02d}.png'
        role = labels[i].lower() if i < len(labels) else ''
        is_source = 'source' in role
        is_goal = 'goal' in role
        if not is_source and not is_goal:
            is_source = i == 0
            is_goal = n == 1 or i == 1
        overlay_class4_frame(
            src,
            dest,
            source_name=source_name if is_source else None,
            goal_name=goal_name if is_goal else None,
            source_mark=source_mark if is_source and _mark_applies_to_image(source_mark, src) else None,
            goal_mark=goal_mark if is_goal and _mark_applies_to_image(goal_mark, src) else None,
        )
        written.append(str(dest))
    return written


def overlay_class4_frame(
    src_path: str | Path,
    dest_path: str | Path,
    *,
    source_name: Optional[str] = None,
    goal_name: Optional[str] = None,
    source_mark: Optional[dict] = None,
    goal_mark: Optional[dict] = None,
) -> Path:
    from PIL import Image, ImageDraw

    im = Image.open(src_path).convert('RGB')
    draw = ImageDraw.Draw(im)
    src_xy = _point_xy(source_mark, im.size)
    goal_xy = _point_xy(goal_mark, im.size)
    src_glyph = endpoint_glyph(source_name or '', fallback='S')
    goal_glyph = endpoint_glyph(goal_name or '', fallback='G')
    if source_name and goal_name and src_glyph == goal_glyph:
        src_glyph, goal_glyph = 'S', 'G'
    if source_name and src_xy is not None:
        color = _color_for_index(0)
        font_m, radius = _marker_font_and_radius(src_glyph)
        _draw_labeled_point(draw, src_xy, color, src_glyph, font=font_m, radius=radius)
    if goal_name and goal_xy is not None:
        color = _color_for_index(1)
        font_m, radius = _marker_font_and_radius(goal_glyph)
        _draw_labeled_point(draw, goal_xy, color, goal_glyph, font=font_m, radius=radius)
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest)
    return dest


def _load_font(size: int):
    from PIL import ImageFont

    for name in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ):
        path = Path(name)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _valid_bbox(bbox: Any) -> Optional[tuple[int, int, int, int]]:
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        bbox = (
            bbox.get('cmin'),
            bbox.get('rmin'),
            bbox.get('cmax'),
            bbox.get('rmax'),
        )
    try:
        cmin, rmin, cmax, rmax = bbox
        if any(v is None for v in (cmin, rmin, cmax, rmax)):
            return None
        cmin, rmin, cmax, rmax = (
            int(round(float(cmin))),
            int(round(float(rmin))),
            int(round(float(cmax))),
            int(round(float(rmax))),
        )
    except (TypeError, ValueError):
        return None
    if cmax <= cmin or rmax <= rmin:
        return None
    return cmin, rmin, cmax, rmax


def _same_still(recorded: str, image_path: str) -> bool:
    """True when ``recorded`` and ``image_path`` are the same RGB frame."""
    a = Path(str(recorded))
    b = Path(str(image_path))
    if str(a) == str(b) or a.name == b.name and (
        a.parent == Path('.')
        or b.parent == Path('.')
        or a.parent.name == b.parent.name
    ):
        return True
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _mark_applies_to_image(mark: Optional[dict], image_path: str) -> bool:
    """Use a stored bbox only on the still it was measured on."""
    if not mark:
        return False
    recorded = mark.get('image_path')
    if not recorded:
        return True
    return _same_still(str(recorded), str(image_path))


def _point_xy(mark: Optional[dict], image_size: tuple[int, int]) -> Optional[tuple[int, int]]:
    """Pixel on *this* PNG: 2D bbox center, scaled from the recording frame.

    Episode GT bbox is ``(cmin, rmin, cmax, rmax)`` in camera pixels (column,
    row; origin top-left). Center is ``((cmin+cmax)/2, (rmin+rmax)/2)``.
    ``frame_wh`` is ``episode_meta.camera`` ``(width, height)``. If the file we
    open is a different size, scale so the point stays on the object.

    ``local_point`` is the pinhole projection of the Unity object origin (often
    a floor pivot). It is not the visual center and is not used here.
    """
    if not mark:
        return None
    bbox = _valid_bbox(mark.get('bbox'))
    if bbox is None:
        return None
    cmin, rmin, cmax, rmax = bbox
    x = (cmin + cmax) / 2.0
    y = (rmin + rmax) / 2.0
    w, h = image_size
    frame = mark.get('frame_wh') or mark.get('image_size')
    if isinstance(frame, (list, tuple)) and len(frame) >= 2:
        try:
            fw, fh = float(frame[0]), float(frame[1])
        except (TypeError, ValueError):
            fw, fh = 0.0, 0.0
        if fw > 1 and fh > 1 and (abs(fw - w) > 0.5 or abs(fh - h) > 0.5):
            x = x * w / fw
            y = y * h / fh
    return int(round(x)), int(round(y))


def _color_for_index(index: int) -> tuple[int, int, int]:
    h = (index * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _contrast_text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def _marker_font_and_radius(glyph: str) -> tuple:
    n = len(glyph or '')
    if n >= 3:
        return _load_font(8), 13
    if n == 2:
        return _load_font(9), 11
    return _load_font(10), 9


def _draw_labeled_point(draw, xy: tuple[int, int], color, glyph: str, *, font, radius: int = 9) -> None:
    cx, cy = xy
    draw.ellipse(
        [cx - radius - 1, cy - radius - 1, cx + radius + 1, cy + radius + 1],
        outline=(255, 255, 255),
        width=2,
    )
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=color,
        outline=(0, 0, 0),
        width=1,
    )
    label = (glyph or '').strip()
    if not label:
        return
    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text(
        (cx - tw / 2 - tb[0], cy - th / 2 - tb[1]),
        label,
        fill=_contrast_text_color(color),
        font=font,
    )
