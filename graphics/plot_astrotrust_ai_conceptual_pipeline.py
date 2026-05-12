from pathlib import Path
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT_DIR = Path(__file__).resolve().parents[1]


def wrap_items(items, max_chars, bullet=False):
    """Wrap each item independently, optionally using bullet-style items."""
    wrapped = []

    for item in items:
        if bullet:
            wrapped.extend(
                textwrap.wrap(
                    item,
                    width=max_chars,
                    initial_indent="• ",
                    subsequent_indent="  ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        else:
            wrapped.extend(
                textwrap.wrap(
                    item,
                    width=max_chars,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )

    return wrapped


def add_adaptive_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    lines,
    *,
    note=None,
    wrap_chars=None,
    title_align="left",
    bullet_items=False,
    facecolor="#F8FAFC",
    edgecolor="#5E6B7E",
    title_color="#18202C",
    text_color="#374151",
    lw=1.25,
    fontsize=8.9,
    title_fontsize=10.2,
    rounding=0.014,
    zorder=2,
):
    """
    Add a compact, publication-style rounded box.
    The text is wrapped according to the box width, avoiding overflow and
    allowing boxes to be sized according to their semantic content.
    """
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.007,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(box)

    pad_x = 0.012 * w + 0.004
    title_y = y + h - 0.022
    body_y = title_y - 0.048

    if title_align == "center":
        title_x = x + 0.5 * w
        title_ha = "center"
    else:
        title_x = x + pad_x
        title_ha = "left"

    ax.text(
        title_x,
        title_y,
        title,
        transform=ax.transAxes,
        ha=title_ha,
        va="top",
        fontsize=title_fontsize,
        fontweight="bold",
        color=title_color,
        zorder=zorder + 1,
    )

    max_chars = wrap_chars if wrap_chars is not None else max(28, int(w * 205))
    body_lines = wrap_items(lines, max_chars=max_chars, bullet=bullet_items)
    ax.text(
        x + pad_x,
        body_y,
        "\n".join(body_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=text_color,
        linespacing=1.22,
        zorder=zorder + 1,
    )

    if note:
        ax.text(
            x + 0.5 * w,
            y + 0.050,
            note,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=text_color,
            linespacing=1.15,
            zorder=zorder + 1,
        )

    return {
        "patch": box,
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
        "center": (x + w / 2, y + h / 2),
    }


def add_arrow(
    ax,
    start,
    end,
    color="#6B7280",
    lw=1.20,
    style="-|>",
    alpha=1.0,
    mutation_scale=12,
    zorder=5,
):
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow

def draw_light_curve(ax, x, y, w, h):
    inset = ax.inset_axes([x, y, w, h], transform=ax.transAxes)
    rng = np.random.default_rng(7)
    t = np.linspace(0, 80, 24)

    mag_g = 20.1 - 1.0 * np.exp(-((t - 32) ** 2) / 180) + 0.08 * rng.normal(size=t.size)
    mag_r = 20.4 - 0.8 * np.exp(-((t - 36) ** 2) / 220) + 0.08 * rng.normal(size=t.size)

    inset.scatter(
        t,
        mag_g,
        s=14,
        marker="o",
        edgecolor="#2563EB",
        facecolor="white",
        linewidth=0.85,
        label="g",
    )
    inset.scatter(
        t + 1.0,
        mag_r,
        s=14,
        marker="s",
        edgecolor="#B45309",
        facecolor="white",
        linewidth=0.85,
        label="r",
    )
    inset.plot(t, mag_g, color="#93C5FD", linewidth=0.95)
    inset.plot(t + 1.0, mag_r, color="#FCD34D", linewidth=0.95)

    inset.set_xlabel("Time (MJD)", fontsize=7.2, labelpad=1)
    inset.set_ylabel("Mag.", fontsize=7.2, labelpad=1)
    inset.tick_params(axis="both", labelsize=6.4, length=2)
    inset.invert_yaxis()
    inset.grid(alpha=0.18, linewidth=0.5)
    inset.legend(frameon=False, fontsize=6.4, loc="upper right", handlelength=1.1)
    inset.spines[["top", "right"]].set_visible(False)
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)

    return inset


def draw_small_bars(ax, x, y, w, h):
    inset = ax.inset_axes([x, y, w, h], transform=ax.transAxes)
    vals = [0.71, 0.68, 0.19, 0.12]
    labels = ["conf.", "1-u", "nov.", "rar."]

    inset.bar(
        np.arange(len(vals)),
        vals,
        width=0.56,
        color=["#2563EB", "#0F766E", "#7C3AED", "#B45309"],
        edgecolor="#374151",
        linewidth=0.45,
    )
    inset.set_ylim(0, 0.85)
    inset.set_xticks(np.arange(len(vals)), labels, fontsize=6.5)
    inset.set_yticks([])
    inset.spines[["top", "right", "left"]].set_visible(False)
    inset.spines["bottom"].set_linewidth(0.6)
    inset.grid(axis="y", alpha=0.15, linewidth=0.5)

    return inset


def build_figure(output_dir="results/final_publication/figures"):
    output_dir = ROOT_DIR / Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(16.2, 7.25), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    title = "AstroTrust-AI: from light-curve ingestion to broker-like decision support"
    subtitle = (
        "Conceptual architecture illustrating how transient light-curve evidence is converted into calibrated,\n"
        "uncertainty-aware, novelty- and rarity-informed prioritization for follow-up triage."
    )

    ax.text(
        0.5,
        0.958,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=18.5,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.914,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color="#4B5563",
        linespacing=1.25,
    )

    # ------------------------------------------------------------------
    # Core pipeline: boxes have content-aware dimensions.
    # ------------------------------------------------------------------
    ax.text(
        0.025,
        0.780,
        "Core processing pipeline",
        transform=ax.transAxes,
        fontsize=10.2,
        color="#6B7280",
        fontweight="bold",
    )

    center_y = 0.592
    left = 0.030
    gap = 0.024
    widths = [0.162, 0.174, 0.176, 0.176, 0.168]
    heights = [0.322, 0.258, 0.258, 0.258, 0.312]

    xs = [left]
    for i in range(1, len(widths)):
        xs.append(xs[-1] + widths[i - 1] + gap)
    ys = [center_y - h / 2 for h in heights]

    cards = []

    cards.append(
        add_adaptive_box(
            ax,
            xs[0],
            ys[0],
            widths[0],
            heights[0],
            "1. Input observations",
            [
                "Multi-band light curve",
                "Optional contextual metadata",
                "Survey-ready transient candidate",
            ],
            title_align="center",
            bullet_items=True,
            wrap_chars=42,
        )
    )
    draw_light_curve(ax, xs[0] + 0.025, ys[0] + 0.045, widths[0] - 0.050, 0.118)

    cards.append(
        add_adaptive_box(
            ax,
            xs[1],
            ys[1],
            widths[1],
            heights[1],
            "2. Representation building",
            [
                "Temporal sequence encoding",
                "Tabular feature extraction",
                "Feature-completeness checks",
            ],
            note="Temporal branch\n+ tabular branch\n→ hybrid representation",
            title_align="center",
            bullet_items=True,
        )
    )

    cards.append(
        add_adaptive_box(
            ax,
            xs[2],
            ys[2],
            widths[2],
            heights[2],
            "3. Predictive inference",
            [
                "Hybrid temporal--tabular CNN",
                "Ensemble aggregation",
                "Fine-class probability vector",
            ],
            note=r"$p(\mathrm{class}\mid\mathrm{light\ curve})$",
            title_align="center",
            bullet_items=True,
        )
    )

    cards.append(
        add_adaptive_box(
            ax,
            xs[3],
            ys[3],
            widths[3],
            heights[3],
            "4. Trust calibration layer",
            [
                "Temperature scaling",
                "Uncertainty estimation",
                "Domain-reliability assessment",
            ],
            note="Calibrated confidence\n+ uncertainty",
            title_align="center",
            bullet_items=True,
        )
    )

    cards.append(
        add_adaptive_box(
            ax,
            xs[4],
            ys[4],
            widths[4],
            heights[4],
            "5. Broker-like decision",
            [
                "Novelty + rarity scoring",
                "Priority-score ranking",
                "Follow-up recommendation",
            ],
            title_align="center",
            bullet_items=True,
        )
    )
    draw_small_bars(ax, xs[4] + 0.034, ys[4] + 0.036, widths[4] - 0.068, 0.096)

    for i in range(len(cards) - 1):
        add_arrow(ax, cards[i]["right"], cards[i + 1]["left"], color="#6B7280", lw=1.25)

    # ------------------------------------------------------------------
    # Interpretation layer: compact cards, reduced empty area.
    # ------------------------------------------------------------------
    ax.text(
        0.025,
        0.350,
        "Interpretation layers",
        transform=ax.transAxes,
        fontsize=10.2,
        color="#6B7280",
        fontweight="bold",
    )

    lower_y = 0.115
    lower_h = 0.15
    lower_gap = 0.026
    lower_widths = [0.25, 0.25, 0.25]
    lower_xs = [0.054]
    for i in range(1, len(lower_widths)):
        lower_xs.append(lower_xs[-1] + lower_widths[i - 1] + lower_gap)

    lower_cards = [
        add_adaptive_box(
            ax,
            lower_xs[0],
            lower_y,
            lower_widths[0],
            lower_h,
            "Scientific interpretation",
            [
                "Class probabilities summarize the likely astrophysical type.",
                "Top-k alternatives preserve ambiguity for downstream analysis.",
                "Reliability is separated from nominal class confidence.",
            ],
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            title_fontsize=10.3,
            fontsize=8.15,
            lw=1.05,
            title_align="center",
            wrap_chars=70,
        ),
        add_adaptive_box(
            ax,
            lower_xs[1],
            lower_y,
            lower_widths[1],
            lower_h,
            "Operational trust signals",
            [
                "Input quality and feature completeness flag fragile cases.",
                "Calibration improves probabilistic interpretability.",
                "Domain checks warn about out-of-distribution inputs.",
            ],
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            title_fontsize=10.3,
            fontsize=8.15,
            lw=1.05,
            title_align="center",
            wrap_chars=70,
        ),
        add_adaptive_box(
            ax,
            lower_xs[2],
            lower_y,
            lower_widths[2],
            lower_h,
            "Actionable output",
            [
                "Candidates are ranked using calibrated evidence.",
                "High-priority objects combine plausibility, novelty, and rarity.",
                "The output supports broker-like triage, not raw classification only.",
            ],
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            title_fontsize=10.3,
            fontsize=8.15,
            lw=1.05,
            title_align="center",
            wrap_chars=70,
        ),
    ]

    vgap = 0.012

    add_arrow(
        ax,
        (cards[2]["bottom"][0], cards[2]["bottom"][1] - vgap),
        (lower_cards[0]["top"][0], lower_cards[0]["top"][1] + vgap),
        color="#9CA3AF",
        lw=1.0,
        alpha=0.95,
        mutation_scale=11,
    )

    add_arrow(
        ax,
        (cards[3]["bottom"][0], cards[3]["bottom"][1] - vgap),
        (lower_cards[1]["top"][0], lower_cards[1]["top"][1] + vgap),
        color="#9CA3AF",
        lw=1.0,
        alpha=0.95,
        mutation_scale=11,
    )

    add_arrow(
        ax,
        (cards[4]["bottom"][0], cards[4]["bottom"][1] - vgap),
        (lower_cards[2]["top"][0], lower_cards[2]["top"][1] + vgap),
        color="#9CA3AF",
        lw=1.0,
        alpha=0.95,
        mutation_scale=11,
    )

    footer = (
        "Figure concept: AstroTrust-AI converts light-curve evidence into calibrated, uncertainty-aware,\n"
        "novelty- and rarity-informed broker-like prioritization for scientific follow-up."
    )
    ax.text(
        0.5,
        0.042,
        footer,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="#4B5563",
        linespacing=1.20,
    )

    png_path = output_dir / "fig_astrotrust_pipeline_concept.png"

    # Do not use bbox_inches="tight"; all content is deliberately kept inside
    # the canvas to avoid accidental cuts in titles, labels, and footers.
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)

    return png_path


if __name__ == "__main__":
    output = build_figure()
    print("Generated file:")
    print(output)
