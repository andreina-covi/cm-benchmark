#!/usr/bin/env python3
"""Build Benchmark avance examples.pptx from the taxonomy template + draft items.

Uses raw episode images (what a VLM would see), not annotated GT frames.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt
from PIL import Image

# --- paths ---
REPO = Path(__file__).resolve().parents[1]
TEMPLATE = Path("/home/andreina/Documents/Programs/Benchmark - avance.pptx")
OUTPUT = Path("/home/andreina/Documents/Programs/Benchmark - avance examples.pptx")
DRAFT_JSON = REPO / "src/cm_benchmark/storage/ai2thor/items/draft_house_001030.json"

# --- palette (from template theme) ---
NAVY = RGBColor(0x00, 0x2F, 0x4A)
TEAL = RGBColor(0x00, 0x93, 0x84)
TERRACOTTA = RGBColor(0xB8, 0x57, 0x41)
CREAM = RGBColor(0xED, 0xE3, 0xDA)
DARK = RGBColor(0x31, 0x39, 0x4D)
GRAY = RGBColor(0x62, 0x6B, 0x73)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

CONSTRUCT_DEFS = {
    "egocentric_encoding": (
        "Egocentric encoding",
        "Relation of a visible object to the viewer's current pose.",
        False,
    ),
    "allocentric_encoding": (
        "Allocentric encoding",
        "Object-to-object relation (draft / thin — answers still mix viewer-frame options).",
        True,
    ),
    "spatial_working_memory": (
        "Spatial working memory",
        "Recall a previously seen location after the object leaves the final view.",
        False,
    ),
    "invisible_displacement": (
        "Invisible displacement",
        "Track an object's location after it is hidden and relocated out of view.",
        False,
    ),
    "spatial_updating": (
        "Spatial updating",
        "Update own pose after real movement; report bearing of a static object (draft / thin).",
        True,
    ),
    "route_knowledge": (
        "Route knowledge",
        "Source→goal planning on a short experienced segment (draft / thin).",
        True,
    ),
    "survey_knowledge": (
        "Survey knowledge",
        "Infer a layout connection / novel path (one unique GT scenario so far).",
        True,
    ),
}

# Locked item_ids (concise unless noted)
EXAMPLES: dict[str, list[str]] = {
    "egocentric_encoding": [
        "house_001030_000_egocentric_encoding_0_ObjaScooter_4_5_concise",
        "house_001030_001_egocentric_encoding_0_ObjaTable_4_3_concise",
    ],
    "allocentric_encoding": [
        "house_001030_003_allocentric_encoding_0_television-sofa_4_0_1_concise",
        "house_001030_004_allocentric_encoding_1_television-sofa_4_0_1_concise",
    ],
    "spatial_working_memory": [
        "house_001030_006_spatial_working_memory_1_chair-diningtable_4_1_1_concise",
        "house_001030_007_spatial_working_memory_1_HousePlant_4_39_concise",
    ],
    "invisible_displacement": [
        "house_001030_009_invisible_displacement_184_Pencil_4_38_concise",
        "house_001030_010_invisible_displacement_184_Candle_4_36_concise",
    ],
    "spatial_updating": [
        "house_001030_012_spatial_updating_1_chair-diningtable_4_1_1_concise",
        "house_001030_013_spatial_updating_1_HousePlant_4_39_concise",
    ],
    "route_knowledge": [
        "house_001030_016_route_knowledge_5_None_concise",
        "house_001030_017_route_knowledge_23_None_concise",
    ],
    "survey_knowledge": [
        "house_001030_019_survey_knowledge_157_None_concise",
        "house_001030_019_survey_knowledge_157_None_verbose",
    ],
}


def _set_run(run, text: str, *, size: int, bold: bool = False, color=DARK) -> None:
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"


def add_textbox(
    slide,
    left,
    top,
    width,
    height,
    *,
    size=14,
    bold=False,
    color=DARK,
    align=PP_ALIGN.LEFT,
    lines: list[tuple[str, dict]] | None = None,
    text: str | None = None,
):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    if lines is None:
        lines = [(text or "", {"size": size, "bold": bold, "color": color})]
    first = True
    for content, style in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = style.get("align", align)
        p.space_after = Pt(style.get("space_after", 4))
        run = p.add_run()
        _set_run(
            run,
            content,
            size=style.get("size", size),
            bold=style.get("bold", bold),
            color=style.get("color", color),
        )
    return box


def add_rect(slide, left, top, width, height, fill: RGBColor):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    return shape


def _picture_size(path: str, max_w: int, max_h: int) -> tuple[int, int]:
    """Return EMU width/height fitting inside max box, preserving aspect."""
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    return int(iw * scale), int(ih * scale)


def clear_slide(slide) -> None:
    for shape in list(slide.shapes):
        sp = shape._element
        sp.getparent().remove(sp)


def blank_layout(prs: Presentation):
    return prs.slide_layouts[10]  # BLANK


def load_items() -> dict[str, dict]:
    data = json.loads(DRAFT_JSON.read_text())
    return {it["item_id"]: it for it in data["items"]}


def refresh_progress_slide(slide) -> None:
    clear_slide(slide)
    W = Inches(10)
    add_rect(slide, Inches(0), Inches(0), W, Inches(0.7), NAVY)
    add_textbox(
        slide,
        Inches(0.4),
        Inches(0.15),
        Inches(9),
        Inches(0.45),
        text="Advances so far (generation pipeline)",
        size=22,
        bold=True,
        color=WHITE,
    )
    bullets = [
        "Taxonomy locked: 4 classes × 8 constructs (+ FoR axis)",
        "SPOC episode → Episode GT (poses, visibility, edges, displacement, layout)",
        "First-draft MC items: CODE-locked answers + answer_source from metadata",
        "Frame annotator for GT review (not model input)",
        "Still upcoming: GT Validator · vision-necessity · FREEZE · Model Evaluation",
    ]
    lines = [(f"•  {b}", {"size": 15, "bold": False, "color": DARK, "space_after": 10}) for b in bullets]
    add_textbox(slide, Inches(0.5), Inches(1.0), Inches(9), Inches(3.8), lines=lines)
    add_textbox(
        slide,
        Inches(0.5),
        Inches(5.05),
        Inches(9),
        Inches(0.35),
        text="Not a frozen eval set — draft candidates only.",
        size=12,
        bold=True,
        color=TERRACOTTA,
    )


def add_divider(prs: Presentation, title: str, subtitle: str) -> None:
    slide = prs.slides.add_slide(blank_layout(prs))
    add_rect(slide, Inches(0), Inches(0), Inches(10), Inches(5.625), NAVY)
    add_textbox(
        slide,
        Inches(0.6),
        Inches(1.8),
        Inches(8.8),
        Inches(0.8),
        text=title,
        size=32,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )
    add_textbox(
        slide,
        Inches(0.8),
        Inches(2.8),
        Inches(8.4),
        Inches(1.0),
        text=subtitle,
        size=16,
        color=CREAM,
        align=PP_ALIGN.CENTER,
    )


def add_construct_header(prs: Presentation, name: str, definition: str, thin: bool) -> None:
    slide = prs.slides.add_slide(blank_layout(prs))
    add_rect(slide, Inches(0), Inches(0), Inches(10), Inches(0.7), TEAL if not thin else TERRACOTTA)
    add_textbox(
        slide,
        Inches(0.4),
        Inches(0.15),
        Inches(9),
        Inches(0.45),
        text=name + ("  ·  first-draft / thin" if thin else ""),
        size=22,
        bold=True,
        color=WHITE,
    )
    add_textbox(
        slide,
        Inches(0.5),
        Inches(1.2),
        Inches(9),
        Inches(1.2),
        text=definition,
        size=18,
        color=DARK,
    )
    add_textbox(
        slide,
        Inches(0.5),
        Inches(2.6),
        Inches(9),
        Inches(1.5),
        lines=[
            ("Next two slides: real draft Q&A with the raw frames a VLM would see.", {"size": 14, "color": GRAY}),
            ("Answers are locked by code from episode GT — not invented by an LLM.", {"size": 14, "color": GRAY}),
        ],
    )


def _fit_images(slide, image_paths: list[str], left, top, max_w, max_h) -> None:
    n = len(image_paths)
    if n == 0:
        return
    gap = Inches(0.12)
    label_h = Inches(0.22)
    if n == 1:
        w, h = _picture_size(image_paths[0], int(max_w), int(max_h - label_h))
        slide.shapes.add_picture(image_paths[0], left, top, width=w, height=h)
        return
    each_w = Emu(int((max_w - gap) / 2))
    labels = ["earlier", "later"]
    for i, path in enumerate(image_paths[:2]):
        x = left + i * (each_w + gap)
        add_textbox(
            slide,
            x,
            top,
            each_w,
            label_h,
            text=labels[i],
            size=10,
            bold=True,
            color=TEAL,
            align=PP_ALIGN.CENTER,
        )
        w, h = _picture_size(path, int(each_w), int(max_h - label_h))
        slide.shapes.add_picture(path, x, top + label_h, width=w, height=h)


def add_example_slide(prs: Presentation, item: dict, construct_title: str, example_n: int, thin: bool) -> None:
    slide = prs.slides.add_slide(blank_layout(prs))
    bar = TERRACOTTA if thin else NAVY
    add_rect(slide, Inches(0), Inches(0), Inches(10), Inches(0.55), bar)
    add_textbox(
        slide,
        Inches(0.3),
        Inches(0.1),
        Inches(9.4),
        Inches(0.4),
        text=f"{construct_title}  ·  Example {example_n}",
        size=16,
        bold=True,
        color=WHITE,
    )

    paths = list(item.get("image_paths") or [])
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    # images left
    _fit_images(slide, paths, Inches(0.25), Inches(0.7), Inches(5.0), Inches(4.2))

    # Q&A right
    rx, rw = Inches(5.4), Inches(4.3)
    add_textbox(slide, rx, Inches(0.7), rw, Inches(0.3), text="Question", size=11, bold=True, color=TEAL)
    q = item.get("question") or ""
    # shorten verbose for slide readability
    if len(q) > 420:
        q = q[:400].rsplit(" ", 1)[0] + "…"
    add_textbox(slide, rx, Inches(0.95), rw, Inches(1.7), text=q, size=11, color=DARK)

    opts = item.get("options") or {}
    ans = item.get("answer")
    opt_lines = []
    for key in sorted(opts.keys()):
        label = f"{key}.  {opts[key]}"
        if key == ans:
            opt_lines.append((label, {"size": 12, "bold": True, "color": TEAL, "space_after": 6}))
        else:
            opt_lines.append((label, {"size": 12, "bold": False, "color": DARK, "space_after": 6}))
    add_textbox(slide, rx, Inches(2.7), rw, Inches(0.25), text="Options", size=11, bold=True, color=TEAL)
    add_textbox(slide, rx, Inches(2.95), rw, Inches(1.6), lines=opt_lines)

    ans_text = opts.get(ans, "")
    add_textbox(
        slide,
        rx,
        Inches(4.55),
        rw,
        Inches(0.4),
        text=f"Answer: {ans} — {ans_text}",
        size=13,
        bold=True,
        color=TEAL,
    )

    for_ = item.get("frame_of_reference", "")
    style = item.get("question_style", "")
    iid = item.get("item_id", "")
    short_id = iid.replace("house_001030_", "")[:55]
    add_textbox(
        slide,
        Inches(0.25),
        Inches(5.2),
        Inches(9.5),
        Inches(0.3),
        text=f"house_001030  ·  FoR={for_}  ·  style={style}  ·  {short_id}",
        size=9,
        color=GRAY,
    )


def add_perspective_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(blank_layout(prs))
    add_rect(slide, Inches(0), Inches(0), Inches(10), Inches(0.7), TERRACOTTA)
    add_textbox(
        slide,
        Inches(0.4),
        Inches(0.15),
        Inches(9),
        Inches(0.45),
        text="Perspective-taking  ·  not generated yet",
        size=22,
        bold=True,
        color=WHITE,
    )
    add_textbox(
        slide,
        Inches(0.5),
        Inches(1.1),
        Inches(9),
        Inches(3.5),
        lines=[
            (
                "Definition: compute a relation from an imagined viewpoint "
                "(another location or object's facing) — not the camera's frame.",
                {"size": 16, "color": DARK, "space_after": 14},
            ),
            (
                "Blocker: requires reference-entity facing / object-frame edges in episode GT.",
                {"size": 15, "bold": True, "color": TERRACOTTA, "space_after": 14},
            ),
            (
                "No draft Q&A examples yet — we do not invent spatial facts.",
                {"size": 15, "color": DARK, "space_after": 10},
            ),
            (
                "Example question shape (taxonomy only): "
                '"From where the chair faces, which object is on its left?"',
                {"size": 14, "color": GRAY},
            ),
        ],
    )


def add_final_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(blank_layout(prs))
    add_rect(slide, Inches(0), Inches(0), Inches(10), Inches(0.7), NAVY)
    add_textbox(
        slide,
        Inches(0.4),
        Inches(0.15),
        Inches(9),
        Inches(0.45),
        text="Takeaways",
        size=22,
        bold=True,
        color=WHITE,
    )
    add_textbox(
        slide,
        Inches(0.5),
        Inches(1.0),
        Inches(9),
        Inches(4.0),
        lines=[
            ("• Taxonomy is the capability axis; FoR is cross-cutting.", {"size": 15, "space_after": 10}),
            ("• Draft items exist for 7/8 constructs from one AI2-THOR episode.", {"size": 15, "space_after": 10}),
            ("• Answers are CODE-locked to GT; images shown = model input.", {"size": 15, "space_after": 10}),
            ("• Thin drafts (allocentric / updating / route / survey) need stronger filters.", {"size": 15, "space_after": 10}),
            ("• Perspective-taking waits on facing metadata.", {"size": 15, "space_after": 10}),
            ("• Next wall: validate → vision-necessity → FREEZE → evaluate VLMs.", {"size": 15, "bold": True, "color": TEAL}),
        ],
    )


def main() -> None:
    if not TEMPLATE.exists():
        raise SystemExit(f"Template missing: {TEMPLATE}")
    shutil.copy2(TEMPLATE, OUTPUT)

    prs = Presentation(str(OUTPUT))
    by_id = load_items()

    # Slide 8 (index 7): refresh progress
    refresh_progress_slide(prs.slides[7])

    add_divider(
        prs,
        "First-draft items — what the model sees",
        "Real MC candidates from house_001030  ·  raw frames only  ·  not a frozen set",
    )

    for construct, item_ids in EXAMPLES.items():
        title, definition, thin = CONSTRUCT_DEFS[construct]
        add_construct_header(prs, title, definition, thin)
        for n, iid in enumerate(item_ids, start=1):
            item = by_id.get(iid)
            if item is None:
                raise KeyError(iid)
            add_example_slide(prs, item, title, n, thin)
        if construct == "spatial_updating":
            add_perspective_slide(prs)

    add_final_slide(prs)
    prs.save(str(OUTPUT))
    print(f"Wrote {OUTPUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
