"""Annotate episode frames with numbered colored points + a legend.

Uses episode GT (JSON). Prefers ``visible_objects[*].bbox`` center when present;
otherwise falls back to ``local_point``. Optionally merges bboxes from a
navigation CSV for older GT exports that lack ``bbox``.

Markers on the image are numbered circles (no long text). The side legend maps
``N → object id`` and shows egocentric relations (agent → object).
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont


def _color_for_index(index: int) -> tuple[int, int, int]:
    """Distinct vivid color by marker index (golden-angle spacing)."""
    h = (index * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _contrast_text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    # Relative luminance — pick black or white for the digit
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        path = Path(name)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _edge_lookup(step: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for edge in step.get('edges_egocentric') or []:
        target = edge.get('target')
        if target:
            out[target] = edge
    return out


def _relation_text(edge: Optional[dict], show_relations: bool) -> str:
    if not show_relations or not edge:
        return ''
    parts = []
    ar = edge.get('angle_relation') or []
    if ar:
        parts.append('/'.join(str(x) for x in ar if x))
    dist = edge.get('distance_label')
    if dist:
        parts.append(str(dist))
    return ' | '.join(parts)


def _valid_bbox(bbox: Any) -> Optional[tuple[int, int, int, int]]:
    if bbox is None:
        return None
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


def _marker_xy(
    obj: dict,
    nav_bboxes: Optional[dict[str, tuple[int, int, int, int]]],
    obj_id: str,
) -> Optional[tuple[int, int]]:
    """Prefer bbox center; else projected local_point."""
    bbox = _valid_bbox(obj.get('bbox'))
    if bbox is None and nav_bboxes is not None:
        bbox = nav_bboxes.get(obj_id)
    if bbox is not None:
        cmin, rmin, cmax, rmax = bbox
        return (cmin + cmax) // 2, (rmin + rmax) // 2
    pt = obj.get('local_point') or (None, None)
    if pt[0] is None or pt[1] is None:
        return None
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def _draw_numbered_point(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    color: tuple[int, int, int],
    number: int,
    *,
    font: ImageFont.ImageFont,
    radius: int = 9,
) -> None:
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
    label = str(number)
    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text(
        (cx - tw / 2 - tb[0], cy - th / 2 - tb[1]),
        label,
        fill=_contrast_text_color(color),
        font=font,
    )


def _legend_line(
    number: int,
    obj_id: str,
    obj: dict,
    edge: Optional[dict],
    *,
    show_relations: bool,
    show_local_xyz: bool,
) -> str:
    parts = [f"{number}. {obj_id}"]
    rel = _relation_text(edge, show_relations)
    if rel:
        parts.append(f"agent→obj: {rel}")
    if show_local_xyz:
        lp = obj.get('local_position')
        if lp is not None and len(lp) == 3 and all(v is not None for v in lp):
            parts.append(f"loc=({lp[0]:.2f},{lp[1]:.2f},{lp[2]:.2f})")
    return '  ·  '.join(parts)


def _draw_legend_panel(
    entries: list[tuple[tuple[int, int, int], int, str]],
    *,
    frame_h: int,
    font: ImageFont.ImageFont,
    font_small: ImageFont.ImageFont,
    header: str,
    min_width: int = 260,
) -> Image.Image:
    """Build a right-side legend panel (numbered swatch + text)."""
    pad = 8
    row_h = 18
    title = 'Legend (agent → object)'
    subtitle = 'Relations are egocentric (from the agent)'
    probe = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    max_text_w = max(
        probe.textbbox((0, 0), title, font=font)[2],
        probe.textbbox((0, 0), subtitle, font=font_small)[2],
        probe.textbbox((0, 0), header, font=font_small)[2],
    )
    for _, _, text in entries:
        max_text_w = max(max_text_w, probe.textbbox((0, 0), text, font=font_small)[2])
    width = max(min_width, pad * 2 + 16 + 6 + max_text_w + 4)

    needed_h = pad + 52 + len(entries) * row_h + pad
    height = max(frame_h, needed_h)
    panel = Image.new('RGB', (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(panel)
    draw.text((pad, pad), title, fill=(255, 255, 255), font=font)
    draw.text((pad, pad + 18), subtitle, fill=(180, 180, 180), font=font_small)
    draw.text((pad, pad + 34), header, fill=(180, 180, 180), font=font_small)

    y = pad + 54
    for color, number, text in entries:
        draw.ellipse([pad, y + 1, pad + 14, y + 15], fill=color, outline=(255, 255, 255))
        n = str(number)
        nb = draw.textbbox((0, 0), n, font=font_small)
        nw, nh = nb[2] - nb[0], nb[3] - nb[1]
        draw.text(
            (pad + 7 - nw / 2 - nb[0], y + 8 - nh / 2 - nb[1]),
            n,
            fill=_contrast_text_color(color),
            font=font_small,
        )
        draw.text((pad + 20, y), text, fill=(230, 230, 230), font=font_small)
        y += row_h
        if y + row_h > height - pad:
            draw.text((pad, height - pad - 14), '…', fill=(200, 200, 200), font=font_small)
            break
    return panel


def _csv_cell(row: dict, key: str) -> Any:
    val = row.get(key, '')
    if val is None:
        return None
    if isinstance(val, str):
        val = val.strip()
        if val == '' or val.lower() == 'nan':
            return None
    return val


def load_bboxes_from_navigation_csv(csv_path: Path) -> dict[int, dict[str, tuple[int, int, int, int]]]:
    """Map timestep -> {obj_id: (cmin, rmin, cmax, rmax)}."""
    by_step: dict[int, dict[str, tuple[int, int, int, int]]] = {}
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = _csv_cell(row, 'obj-id')
            if not obj:
                continue
            step_raw = _csv_cell(row, 'timestep')
            if step_raw is None:
                continue
            step = int(float(step_raw))
            bbox = _valid_bbox(
                (
                    _csv_cell(row, 'cmin'),
                    _csv_cell(row, 'rmin'),
                    _csv_cell(row, 'cmax'),
                    _csv_cell(row, 'rmax'),
                )
            )
            if bbox is None:
                continue
            by_step.setdefault(step, {})[str(obj)] = bbox
    return by_step


def annotate_step_image(
    step: dict,
    *,
    show_relations: bool = True,
    show_local_xyz: bool = False,
    nav_bboxes: Optional[dict[str, tuple[int, int, int, int]]] = None,
    font: Optional[ImageFont.ImageFont] = None,
    point_radius: int = 9,
) -> Image.Image:
    """Return frame with numbered colored points + a right-side legend.

    Legend relations come from ``edges_egocentric`` (agent → object).
    """
    image_path = step.get('image_path')
    if not image_path:
        raise ValueError(f"step {step.get('step')} has no image_path")
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")

    img = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = font or _load_font(14)
    font_small = _load_font(11)
    font_marker = _load_font(10)
    edges = _edge_lookup(step)
    visible = step.get('visible_objects') or {}

    legend_entries: list[tuple[tuple[int, int, int], int, str]] = []
    number = 0
    for obj_id, obj in visible.items():
        xy = _marker_xy(obj, nav_bboxes, obj_id)
        if xy is None:
            continue
        number += 1
        color = _color_for_index(number - 1)
        _draw_numbered_point(
            draw, xy, color, number, font=font_marker, radius=point_radius
        )
        legend_entries.append(
            (
                color,
                number,
                _legend_line(
                    number,
                    obj_id,
                    obj,
                    edges.get(obj_id),
                    show_relations=show_relations,
                    show_local_xyz=show_local_xyz,
                ),
            )
        )

    header = f"step={step.get('step')}  visible={len(visible)}"
    panel = _draw_legend_panel(
        legend_entries,
        frame_h=img.height,
        font=font,
        font_small=font_small,
        header=header,
    )
    canvas_h = max(img.height, panel.height)
    out = Image.new('RGB', (img.width + panel.width, canvas_h), color=(20, 20, 20))
    out.paste(img, (0, 0))
    out.paste(panel, (img.width, 0))
    return out


def annotate_episode(
    episode: dict,
    output_dir: Path,
    *,
    steps: Optional[list[int]] = None,
    show_relations: bool = True,
    show_local_xyz: bool = False,
    navigation_csv: Optional[Path] = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nav_by_step: dict[int, dict[str, tuple[int, int, int, int]]] = {}
    if navigation_csv is not None:
        nav_by_step = load_bboxes_from_navigation_csv(navigation_csv)

    selected = set(steps) if steps is not None else None
    written: list[Path] = []
    font = _load_font(14)
    for step in episode.get('steps') or []:
        step_idx = int(step.get('step'))
        if selected is not None and step_idx not in selected:
            continue
        annotated = annotate_step_image(
            step,
            show_relations=show_relations,
            show_local_xyz=show_local_xyz,
            nav_bboxes=nav_by_step.get(step_idx),
            font=font,
        )
        out_path = output_dir / f"annotated_step_{step_idx:04d}.png"
        annotated.save(out_path)
        written.append(out_path)
    return written


def _parse_steps(args: argparse.Namespace) -> Optional[list[int]]:
    if args.step is not None:
        return [args.step]
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else start
        if end < start:
            raise SystemExit('--end must be >= --start')
        return list(range(start, end + 1))
    return None


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description='Annotate episode frames with colored points + a color legend.'
    )
    parser.add_argument(
        '--episode_json',
        required=True,
        type=Path,
        help='Path to episode GT JSON',
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        type=Path,
        help='Directory for annotated PNGs',
    )
    parser.add_argument('--step', type=int, default=None, help='Single step index')
    parser.add_argument('--start', type=int, default=None, help='First step (inclusive)')
    parser.add_argument('--end', type=int, default=None, help='Last step (inclusive)')
    parser.add_argument(
        '--navigation_csv',
        type=Path,
        default=None,
        help='Optional nav CSV to supply bboxes for older GT without bbox field',
    )
    parser.add_argument(
        '--no_relations',
        action='store_true',
        help='Omit egocentric angle/distance labels from the legend',
    )
    parser.add_argument(
        '--show_local_xyz',
        action='store_true',
        help='Also print local camera-frame xyz in the legend',
    )
    args = parser.parse_args(argv)

    with args.episode_json.open() as f:
        episode = json.load(f)

    written = annotate_episode(
        episode,
        args.output_dir,
        steps=_parse_steps(args),
        show_relations=not args.no_relations,
        show_local_xyz=args.show_local_xyz,
        navigation_csv=args.navigation_csv,
    )
    print(f'Wrote {len(written)} annotated frame(s) → {args.output_dir}')


if __name__ == '__main__':
    main()
