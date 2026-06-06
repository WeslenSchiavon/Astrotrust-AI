#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a publication-ready stage-aware AstroTrust-AI pipeline figure.

The script rebuilds the visual style of the supplied reference figure:
- 5-card core processing pipeline
- 3 interpretation-layer cards
- calibrated/reliability/priority broker-like decision support
- clean scientific visual style, no GUI screenshot look

Expected asset structure, as produced by the zip previously generated:

astrotrust_pipe/
  01_exact_crops_png/
  02_transparent_background_png/

Recommended project placement:
  <project_root>/graphics/astrotrust_pipe/01_exact_crops_png
  <project_root>/graphics/astrotrust_pipe/02_transparent_background_png

Default output:
  results/final_publication/figures/fig_astrotrust_ai_conceptual_pipeline_stage_aware.png
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Project paths (fixed, no CLI parameters)
# Keeps the same style requested by the user.
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

def _safe_parent(path: Path, n: int) -> Path | None:
    try:
        return path.parents[n]
    except IndexError:
        return None

def _guess_project_root() -> Path:
    candidates = [
        _safe_parent(SCRIPT_PATH, 1),  # e.g., <project>/experiments/script.py
        _safe_parent(SCRIPT_PATH, 2),  # e.g., <project>/graphics/astrotrust_pipe/script.py
        Path.cwd(),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if (candidate / "results").exists() or (candidate / "experiments").exists() or (candidate / "graphics").exists():
            return candidate
    return _safe_parent(SCRIPT_PATH, 1) or Path.cwd()

def _guess_asset_root(root_dir: Path) -> Path:
    candidates = [
        SCRIPT_PATH.parent,
        SCRIPT_PATH.parent / "astrotrust_pipe",
        root_dir / "graphics" / "astrotrust_pipe",
        root_dir / "astrotrust_pipe",
        Path.cwd() / "graphics" / "astrotrust_pipe",
        Path.cwd() / "astrotrust_pipe",
    ]
    for candidate in candidates:
        if (candidate / "02_transparent_background_png").exists() and (candidate / "01_exact_crops_png").exists():
            return candidate
    return root_dir / "graphics" / "astrotrust_pipe"

ROOT_DIR = _guess_project_root()
ASSET_ROOT = _guess_asset_root(ROOT_DIR)
BASE_DIR = ROOT_DIR
ICONS_DIR = ASSET_ROOT / "02_transparent_background_png"
OUTPUT_PNG = ROOT_DIR / Path("results/final_publication/figures/fig_astrotrust_ai_conceptual_pipeline_stage_aware.png")
OUTPUT_PDF = ROOT_DIR / Path("results/final_publication/figures/fig_astrotrust_ai_conceptual_pipeline_stage_aware.pdf")


# ============================================================
# Global canvas configuration
# ============================================================

W, H = 1672, 941
BG = "white"

# Palette calibrated to the reference figure
NAVY = "#152565"
BLUE = "#1c55d5"
MID_BLUE = "#2f66e0"
PURPLE = "#5f38c5"
ORANGE = "#dd620e"
TEAL = "#0b7e8b"
LIGHT_TEAL = "#4bb5cc"
LIGHT_PURPLE = "#8868d8"
BOX_TEXT = "#10143d"
LIGHT_BORDER = "#ccd2ec"
ORANGE_LIGHT = "#f18b42"


# ============================================================
# Asset and font discovery
# ============================================================



def _font_candidates(bold: bool = False, italic: bool = False) -> Iterable[str]:
    """Times-like serif font priority, close to the supplied reference image.

    The script does not bundle fonts. On Windows, Times New Roman is normally
    available. On Linux, Liberation Serif / FreeSerif are close fallbacks.
    """
    if bold and italic:
        return [
            "C:/Windows/Fonts/timesbi.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf",
            "/Library/Fonts/Times New Roman Bold Italic.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSerifBoldItalic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf",
        ]
    if bold:
        return [
            "C:/Windows/Fonts/timesbd.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            "/Library/Fonts/Times New Roman Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ]
    if italic:
        return [
            "C:/Windows/Fonts/timesi.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
            "/Library/Fonts/Times New Roman Italic.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
        ]
    return [
        "C:/Windows/Fonts/times.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]


def get_font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    for path in _font_candidates(bold=bold, italic=italic):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()

# Typography calibrated for 1672 x 941 px
TITLE_FONT = get_font(38, bold=True)
SUBTITLE_FONT = get_font(20)
SECTION_FONT = get_font(18, bold=True)
STEP_TITLE_FONT = get_font(20, bold=True)
BODY_FONT = get_font(20)
BODY_BIG_FONT = get_font(17)
LOWER_TITLE_FONT = get_font(25, bold=True)
LOWER_BODY_FONT = get_font(20)
BROKER_SMALL_FONT = get_font(14)
CAPTION_FONT = get_font(17, italic=True)
PLUS_FONT = get_font(24, bold=True)


# ============================================================
# Drawing helpers
# ============================================================

def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, spacing: int = 4) -> Tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return box[2] - box[0], box[3] - box[1]


def draw_multiline(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    spacing: int = 4,
    align: str = "left",
) -> None:
    draw.multiline_text(xy, text, font=font, fill=fill, spacing=spacing, align=align)


def center_text(
    draw: ImageDraw.ImageDraw,
    x_center: float,
    y_top: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    spacing: int = 4,
) -> int:
    width, height = text_size(draw, text, font, spacing=spacing)
    draw_multiline(draw, (x_center - width / 2, y_top), text, font, fill, spacing=spacing, align="center")
    return height


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int, int, int],
    outline: str,
    width: int = 2,
    radius: int = 16,
    fill: Optional[str] = None,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, outline=outline, width=width, fill=fill)


def dotted_hline(
    draw: ImageDraw.ImageDraw,
    x1: int,
    x2: int,
    y: int,
    color: str,
    width: int = 2,
    dash: int = 3,
    gap: int = 5,
) -> None:
    x = x1
    while x < x2:
        draw.line((x, y, min(x + dash, x2), y), fill=color, width=width)
        x += dash + gap


def dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: str,
    width: int = 2,
    dash: int = 4,
    gap: int = 4,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0
    while pos < dist:
        end = min(pos + dash, dist)
        draw.line(
            (x1 + ux * pos, y1 + uy * pos, x1 + ux * end, y1 + uy * end),
            fill=color,
            width=width,
        )
        pos += dash + gap


def arrow(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: str,
    width: int = 2,
    head_len: int = 12,
    head_w: int = 7,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    left = (
        x2 - head_len * math.cos(angle) + head_w * math.sin(angle),
        y2 - head_len * math.sin(angle) - head_w * math.cos(angle),
    )
    right = (
        x2 - head_len * math.cos(angle) - head_w * math.sin(angle),
        y2 - head_len * math.sin(angle) + head_w * math.cos(angle),
    )
    draw.polygon([p2, left, right], fill=color)


def paste_icon(
    canvas: Image.Image,
    asset_root: Path,
    icon_name: str,
    x: int,
    y: int,
    use_transparent: bool = True,
) -> None:
    folder = asset_root / ("02_transparent_background_png" if use_transparent else "01_exact_crops_png")
    suffix = "_transparent.png" if use_transparent else ".png"
    path = folder / f"{icon_name}{suffix}"

    # Fallback: sometimes using the non-transparent crop is more faithful for plot-like assets.
    if not path.exists():
        path = asset_root / "01_exact_crops_png" / f"{icon_name}.png"
    if not path.exists():
        raise FileNotFoundError(f"Missing icon asset: {icon_name} in {asset_root}")

    icon = Image.open(path).convert("RGBA")
    canvas.alpha_composite(icon, (x, y))




def make_light_curve_plot(width: int = 246, height: int = 136, dpi: int = 180) -> Image.Image:
    """Generate the card-1 light-curve plot using the same model/style
    adopted in plot_astrotrust_ai_conceptual_pipeline.py.

    This is a real Matplotlib chart generated at runtime, not a static icon.
    It uses deterministic synthetic multi-band observations for g/r filters,
    with an inverted magnitude axis, points, connecting lines, grid, labels,
    and a compact legend.
    """
    rng = np.random.default_rng(7)
    t = np.linspace(0, 80, 24)

    # Same synthetic transient model used in the conceptual-pipeline script.
    mag_g = 20.1 - 1.0 * np.exp(-((t - 32) ** 2) / 180) + 0.08 * rng.normal(size=t.size)
    mag_r = 20.4 - 0.8 * np.exp(-((t - 36) ** 2) / 220) + 0.08 * rng.normal(size=t.size)

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0.24, 0.24, 0.68, 0.66])

    
    # Banda g
    ax.plot(
        t,
        mag_g,
        color="#2563EB",
        linewidth=0.5,
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgecolor="#2563EB",
        markeredgewidth=0.85,
        label="g",
        zorder=3,
    )

    # Banda r
    ax.plot(
        t + 1.0,
        mag_r,
        color="#F97316",
        linewidth=0.5,
        marker="s",
        markersize=2.5,
        markerfacecolor="white",
        markeredgecolor="#F97316",
        markeredgewidth=0.85,
        label="r",
        zorder=3,
    )

    # mais marcas no eixo X
    ax.set_xticks(np.arange(0, 81, 20))   # 0, 10, 20, ..., 80

    # mais marcas no eixo Y
    ax.set_yticks(np.arange(19.0, 20.6, 0.5))
    ax.set_xlabel("Time (MJD)", fontsize=5, labelpad=1)
    ax.set_ylabel("Mag", fontsize=5, labelpad=1)
    ax.tick_params(axis="both", labelsize=6.4, length=2, width=0.55, pad=1.2)
    ax.invert_yaxis()
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(
        frameon=False,
        fontsize=5,
        loc="upper right",
        handlelength=1.6,
        markerscale=1.0,
        borderpad=0.1,
        labelspacing=0.35,
    )
   # Borda externa completa do gráfico
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#9CA3AF")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#7f8491")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", transparent=False)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf).convert("RGBA")


# ============================================================
# Figure geometry
# ============================================================

BOXES_TOP = {
    1: (23, 205, 333, 552),
    2: (368, 202, 676, 552),
    3: (703, 202, 995, 552),
    4: (1022, 202, 1298, 552),
    5: (1326, 202, 1623, 552),
}

BOXES_BOTTOM = {
    "left": (43, 634, 562, 847),
    "mid": (591, 634, 1102, 847),
    "right": (1119, 634, 1619, 847),
}


# ============================================================
# Main figure generation
# ============================================================

def build_figure(asset_root: Path, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # ---------------------- title block ----------------------
    center_text(
        draw,
        W // 2,
        10,
        "AstroTrust-AI: stage-aware broker-like decision support",
        TITLE_FONT,
        NAVY,
        spacing=4,
    )

    star_y = 82
    draw.line((611, star_y, 807, star_y), fill=LIGHT_TEAL, width=2)
    draw.line((867, star_y, 1062, star_y), fill=LIGHT_PURPLE, width=2)
    paste_icon(canvas, asset_root, "21_title_center_star", 821, 70, use_transparent=True)

    subtitle = (
        "Conceptual architecture showing how complete light curves and partial alert states are routed through\n"
        "calibrated, uncertainty-aware, novelty- and rarity-informed follow-up triage."
    )
    center_text(draw, W // 2, 105, subtitle, SUBTITLE_FONT, NAVY, spacing=5)

    # ---------------------- section headers ------------------
    draw.text((35, 158), "CORE PROCESSING PIPELINE", font=SECTION_FONT, fill=NAVY)
    draw.line((36, 187, 93, 187), fill=LIGHT_TEAL, width=3)

    draw.text((36, 594), "INTERPRETATION LAYERS", font=SECTION_FONT, fill=NAVY)
    draw.line((37, 623, 126, 623), fill=LIGHT_TEAL, width=3)

    # ---------------------- top cards ------------------------
    outline_colors = {1: LIGHT_TEAL, 2: LIGHT_TEAL, 3: MID_BLUE, 4: LIGHT_PURPLE, 5: ORANGE_LIGHT}
    for i, rect in BOXES_TOP.items():
        rounded_box(draw, rect, outline=outline_colors[i], width=2, radius=16)

    # 1. Input observations
    paste_icon(canvas, asset_root, "01_step_badge_1", 43, 224, use_transparent=True)
    draw.text((98, 229), "Input observations", font=STEP_TITLE_FONT, fill=TEAL)
    draw_multiline(
        draw,
        (37, 275),
        "• Complete or partial light curve\n• Alert-stage metadata\n• Survey-ready transient candidate",
        BODY_BIG_FONT,
        BOX_TEXT,
        spacing=8,
    )
    # Real generated plot, not a static icon/crop.
   # Card 1 geometry
    box_x1, box_y1, box_x2, box_y2 = BOXES_TOP[1]

    # Light-curve plot position inside card 1
    plot_x = box_x1 - 10
    plot_y = box_y1 + 150
    plot_w = 246 * 1.3
    plot_h = 136 * 1.4

    canvas.alpha_composite(
        make_light_curve_plot(width=plot_w, height=plot_h),
        (plot_x, plot_y)
    )
    rounded_box(draw, BOXES_TOP[1], outline=outline_colors[1], width=2, radius=16)

    # 2. Representation building
    paste_icon(canvas, asset_root, "02_step_badge_2", 382, 216, use_transparent=True)
    draw.text((437, 229), "Representation building", font=STEP_TITLE_FONT, fill=TEAL)
    draw_multiline(
        draw,
        (388, 275),
        "• Temporal sequence encoding\n• Tabular feature extraction\n• Feature-completeness checks",
        BODY_BIG_FONT,
        BOX_TEXT,
        spacing=8,
    )
    dotted_hline(draw, 388, 656, 375, LIGHT_TEAL, width=2)
    # Caixa 2
    box_x1, box_y1, box_x2, box_y2 = BOXES_TOP[2]

    # Centro horizontal da caixa
    text_center_x = (box_x1 + box_x2) / 2

    center_text(
        draw,
        text_center_x,
        393,
        "Temporal branch\n+ tabular branch\n→ hybrid representation",
        BODY_BIG_FONT,
        BOX_TEXT,
        spacing=5,
    )
    paste_icon(canvas, asset_root, "07_temporal_sequence_icon", 380, 461, use_transparent=True)
    paste_icon(canvas, asset_root, "08_tabular_features_icon", 502, 454, use_transparent=True)
    paste_icon(canvas, asset_root, "09_hybrid_network_icon", 587, 457, use_transparent=True)
    draw.text((479, 491), "+", font=PLUS_FONT, fill=BOX_TEXT)
    arrow(draw, (560, 496), (586, 496), color=BOX_TEXT, width=2, head_len=9, head_w=5)

    # 3. Stage-aware predictive inference
    paste_icon(canvas, asset_root, "03_step_badge_3", 722, 216, use_transparent=True)
    draw_multiline(draw, (780, 224), "Stage-aware\ninference", STEP_TITLE_FONT, BLUE, spacing=2)
    draw_multiline(
        draw,
        (729, 275),
        "• Complete route: final ensemble\n• Partial route: early-aware ensemble\n• Fine-class probability vector",
        get_font(15),
        BOX_TEXT,
        spacing=7,
    )
    dotted_hline(draw, 723, 973, 375, MID_BLUE, width=2)

    # Compact routing boxes inside the predictive card.
    rounded_box(draw, (724, 392, 847, 438), outline="#7da7ff", width=1, radius=9, fill="#F8FBFF")
    rounded_box(draw, (855, 392, 974, 438), outline="#f0a45f", width=1, radius=9, fill="#FFF7ED")
    center_text(draw, 785, 398, "Complete\nroute", get_font(13, bold=True), BLUE, spacing=1)
    center_text(draw, 914, 398, "Partial\nroute", get_font(13, bold=True), ORANGE, spacing=1)

    # Probability expression above the bar chart
    prob_font = get_font(15, italic=True)

    draw.text(
        (733, 447),
        "p(class | stage, light-curve, features)",
        font=prob_font,
        fill=BOX_TEXT,
    )

    paste_icon(canvas, asset_root, "10_probability_vector_icon", 731, 472, use_transparent=False)

    # 4. Calibration and reliability
    paste_icon(canvas, asset_root, "04_step_badge_4", 1040, 220, use_transparent=True)
    draw_multiline(draw, (1095, 224), "Calibration and\nreliability layer", STEP_TITLE_FONT, PURPLE, spacing=2)
    draw_multiline(
        draw,
        (1055, 275),
        "• Temperature scaling\n• Uncertainty estimation\n• Domain-reliability assessment",
        BODY_BIG_FONT,
        BOX_TEXT,
        spacing=8,
    )
    # Caixa 4
    box_x1, box_y1, box_x2, box_y2 = BOXES_TOP[4]

    # Centro horizontal da caixa
    box_center_x = (box_x1 + box_x2) / 2

    dotted_hline(draw, 1054, 1275, 375, LIGHT_PURPLE, width=2)
    paste_icon(canvas, asset_root, "11_calibration_shield_icon", 1115, 391, use_transparent=True)

    center_text(
        draw,
        box_center_x,
        496,
        "Calibrated confidence\n+ uncertainty",
        BODY_BIG_FONT,
        PURPLE,
        spacing=5,
    )

    # 5. Broker-like decision support
    paste_icon(canvas, asset_root, "05_step_badge_5", 1350, 225, use_transparent=True)
    draw_multiline(draw, (1400, 224), "Broker-like decision\nsupport", STEP_TITLE_FONT, ORANGE, spacing=2)
    draw_multiline(
        draw,
        (1360, 275),
        "• Top-k classes + calibrated confidence\n• Stage-aware novelty/rarity signals\n• Priority score + follow-up\n  recommendation",
        get_font(15),
        BOX_TEXT,
        spacing=7,
    )
    dotted_hline(draw, 1360, 1589, 375, ORANGE_LIGHT, width=2)
    paste_icon(canvas, asset_root, "12_broker_priority_bars_only", 1360 - 15, 392, use_transparent=False)

    icon_items = [
        ("13_topk_target_icon", 1362, 498, "top-k"),
        ("14_confidence_shield_icon", 1416, 498, "conf."),
        ("15_novelty_search_icon", 1468, 498, "nov."),
        ("16_rarity_diamond_icon", 1519, 498, "rar."),
        ("17_priority_star_icon", 1573, 498, "prio."),
    ]

    icon_w = 28   # ajuste se necessário
    label_y = 530
    x_offset = -5   # move tudo para a direita

    for icon_name, icon_x, icon_y, label in icon_items:
        new_x = icon_x + x_offset
        paste_icon(canvas, asset_root, icon_name, new_x, icon_y, use_transparent=True)

        icon_center_x = new_x + icon_w / 2
        center_text(draw, icon_center_x, label_y, label, BROKER_SMALL_FONT, BOX_TEXT)

    # redesenha a borda da caixa 5 por cima de tudo
    rounded_box(draw, BOXES_TOP[5], outline=ORANGE_LIGHT, width=2, radius=16)

    # Solid arrows between pipeline cards
    arrow(draw, (333, 379), (366, 379), color=NAVY, width=2)
    arrow(draw, (675, 379), (702, 379), color=NAVY, width=2)
    arrow(draw, (995, 379), (1020, 379), color=NAVY, width=2)
    arrow(draw, (1298, 379), (1322, 379), color=NAVY, width=2)

    # ---------------------- interpretation cards -------------
    for rect in BOXES_BOTTOM.values():
        rounded_box(draw, rect, outline=LIGHT_BORDER, width=2, radius=16)

    paste_icon(canvas, asset_root, "18_scientific_telescope_icon", 65, 653, use_transparent=True)
    draw.text((207, 658), "Scientific interpretation", font=LOWER_TITLE_FONT, fill=TEAL)
    draw_multiline(
        draw,
        (205, 693),
        "Class probabilities summarize the likely\n"
        "astrophysical type.\n"
        "Top-k alternatives preserve ambiguity for\n"
        "downstream analysis.\n"
        "Reliability is separated from nominal\n"
        "class confidence.",
        LOWER_BODY_FONT,
        BOX_TEXT,
        spacing=5,
    )

    paste_icon(canvas, asset_root, "19_operational_trust_shield_icon", 624, 651, use_transparent=True)
    center_text(draw, 847, 658, "Operational trust signals", LOWER_TITLE_FONT, BLUE)
    draw_multiline(
        draw,
        (741, 693),
        "Input quality and feature completeness\n"
        "flag fragile cases.\n"
        "Stage-aware routing identifies partial-\n"
        "alert cases.\n"
        "Calibration and domain checks support\n"
        "operational reliability.",
        LOWER_BODY_FONT,
        BOX_TEXT,
        spacing=5,
    )

    paste_icon(canvas, asset_root, "20_actionable_clipboard_icon", 1132, 651, use_transparent=True)
    draw.text((1295, 658), "Actionable output", font=LOWER_TITLE_FONT, fill=PURPLE)
    draw_multiline(
        draw,
        (1245, 693),
        "Complete and partial candidates are\n"
        "ranked using calibrated evidence.\n"
        "High-priority objects combine plausibility,\n"
        "novelty, rarity, and alert stage.\n"
        "The output supports broker-like triage,\n"
        "not raw classification only.",
        LOWER_BODY_FONT,
        BOX_TEXT,
        spacing=5,
    )

    # Dashed interpretation connectors
    dashed_line(draw, (158, 552), (416, 612), color="#1994c0", width=2)
    dashed_line(draw, (555, 552), (423, 612), color="#1994c0", width=2)
    dashed_line(draw, (416, 612), (416, 634), color="#1994c0", width=2)
    dashed_line(draw, (423, 612), (423, 634), color="#1994c0", width=2)
    dashed_line(draw, (849, 552), (849, 634), color=MID_BLUE, width=2)
    dashed_line(draw, (1159, 552), (995, 634), color=PURPLE, width=2)
    dashed_line(draw, (1475, 552), (1247, 634), color=ORANGE, width=2)

    # Caption
    caption = (
        "Figure concept: AstroTrust-AI routes complete and partial light-curve evidence through calibrated,\n"
        "uncertainty-aware, novelty- and rarity-informed broker-like prioritization for scientific follow-up."
    )
    center_text(draw, W // 2, 870, caption, CAPTION_FONT, NAVY, spacing=4)

    rgb = canvas.convert("RGB")
    rgb.save(output_png, quality=95, dpi=(300, 300))
    try:
        rgb.save(OUTPUT_PDF, "PDF", resolution=300.0)
    except Exception as exc:
        print(f"[WARN] Could not save PDF: {exc}")
    print(f"PNG saved to: {output_png}")
    print(f"PDF saved to: {OUTPUT_PDF}")

   


def main() -> None:
    asset_root = ASSET_ROOT
    if not asset_root.exists():
        raise FileNotFoundError(
            f"Could not find the icon assets directory: {asset_root}\n"
            "Expected structure: graphics/astrotrust_pipe/02_transparent_background_png, "
            "or run this script from the extracted astrotrust_pipe folder."
        )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using assets from: {asset_root}")
    build_figure(asset_root=asset_root, output_png=OUTPUT_PNG)


if __name__ == "__main__":
    main()
