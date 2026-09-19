#!/usr/bin/env python3
"""Generate method and storage figures for the DDC manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_FULL_WIDTH_IN = 7.1
PAPER_FONT_SIZE = 9.0
PAPER_SMALL_FONT_SIZE = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "figures",
    )
    parser.add_argument(
        "--figure",
        action="append",
        choices=("overview", "storage"),
        help=(
            "Figure to generate; repeat to select both. By default both figures are generated. "
            "The overview figure does not require experiment JSON files."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    text_color: str = "#17263b",
    fontsize: float = PAPER_FONT_SIZE,
) -> FancyBboxPatch:
    rgb = np.asarray(to_rgb(color))
    facecolor = tuple(0.91 + 0.09 * rgb)
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.028",
        linewidth=1.15,
        edgecolor=color,
        facecolor=facecolor,
        zorder=3,
    )
    axis.add_patch(box)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        linespacing=1.24,
        zorder=4,
    )
    return box


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#596a75",
    linestyle: str = "-",
) -> FancyArrowPatch:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10.5,
        linewidth=1.05,
        linestyle=linestyle,
        color=color,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=2,
    )
    axis.add_patch(arrow)
    return arrow


def add_routed_arrow(
    axis: plt.Axes,
    vertices: list[tuple[float, float]],
    *,
    color: str,
) -> None:
    path = MplPath(
        vertices,
        [MplPath.MOVETO, *([MplPath.LINETO] * (len(vertices) - 1))],
    )
    axis.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=10.5,
            linewidth=1.05,
            color=color,
            joinstyle="round",
            zorder=2,
        )
    )


def plot_method_overview(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(PAPER_FULL_WIDTH_IN, 3.05))
    figure.subplots_adjust(left=0.012, right=0.988, bottom=0.025, top=0.985)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    heading_color = "#4c5d68"
    axis.text(
        0.018,
        0.94,
        "ENCODING",
        fontsize=PAPER_SMALL_FONT_SIZE,
        fontweight="bold",
        color=heading_color,
        va="top",
    )

    top_y = 0.655
    top_height = 0.215
    top_boxes = (
        (0.035, 0.205, "Calculated dense states\n$x_0,\\ldots,x_K$", "#1976d2"),
        (0.285, 0.170, "Two-sided time\nprediction", "#7b1fa2"),
        (
            0.500,
            0.220,
            "Residual quantization\n+ unquantized exceptions",
            "#ef6c00",
        ),
        (
            0.765,
            0.200,
            "Original exact anchors\n+ indexed DDC chunks",
            "#2e7d32",
        ),
    )
    for left, width, label, color in top_boxes:
        add_box(axis, (left, top_y), width, top_height, label, color, fontsize=8.6)
    for (left, width, _, _), (next_left, _, _, _) in zip(
        top_boxes[:-1], top_boxes[1:], strict=True
    ):
        add_arrow(
            axis,
            (left + width + 0.006, top_y + top_height / 2),
            (next_left - 0.015, top_y + top_height / 2),
        )

    axis.plot(
        [0.02, 0.98],
        [0.52, 0.52],
        color="#d5dde2",
        linewidth=0.8,
        zorder=0,
    )
    axis.text(
        0.018,
        0.455,
        "READING",
        fontsize=PAPER_SMALL_FONT_SIZE,
        fontweight="bold",
        color=heading_color,
        va="top",
    )

    bottom_y = 0.155
    bottom_height = 0.205
    bottom_boxes = (
        (0.075, 0.225, "Time + variable\nrequest", "#6d4c41"),
        (0.385, 0.245, "Selective decode\nof one local chunk", "#00796b"),
        (0.720, 0.205, "Reconstructed\nstate", "#1976d2"),
    )
    for left, width, label, color in bottom_boxes:
        add_box(axis, (left, bottom_y), width, bottom_height, label, color)
    for (left, width, _, _), (next_left, _, _, _) in zip(
        bottom_boxes[:-1], bottom_boxes[1:], strict=True
    ):
        add_arrow(
            axis,
            (left + width + 0.006, bottom_y + bottom_height / 2),
            (next_left - 0.015, bottom_y + bottom_height / 2),
        )

    archive_center = top_boxes[-1][0] + top_boxes[-1][1] / 2
    decode_center = bottom_boxes[1][0] + bottom_boxes[1][1] / 2
    add_routed_arrow(
        axis,
        [
            (archive_center, top_y - 0.004),
            (archive_center, 0.485),
            (decode_center, 0.485),
            (decode_center, bottom_y + bottom_height + 0.012),
        ],
        color="#2e7d32",
    )

    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"ddc_method_overview.{suffix}",
            dpi=240,
            facecolor="white",
        )
    plt.close(figure)

def storage_progression() -> tuple[list[str], list[float]]:
    search = load_json(PROJECT_ROOT / "experiments" / "dense_dump_codec_adaptive_search.json")
    final = load_json(PROJECT_ROOT / "experiments" / "ddc_xz_zigzag_final_matrix_summary.json")
    temporal = load_json(PROJECT_ROOT / "experiments" / "ddc_temporal_keyframe_matrix_summary.json")
    independent_values = sorted(
        float(row["baseline_storage_over_raw_0p5"]) for row in final["rows"]
    )
    independent_median = float(np.median(independent_values))
    selected = search["selected_full_matrix_candidate"]
    adaptive = search["lossless_backend_screen"]["adaptive_temporal_order_extension"]
    return (
        [
            "Raw 0.5M",
            "Quantized residual\n+ Deflate",
            "Adaptive residual\n+ raw anchors",
            "Independent\nkeyframe repack",
            "Temporal bzip2\nkeyframes",
            "Final XZ+zigzag\nkeyframes",
        ],
        [
            1.0,
            float(selected["median_storage_over_raw_0p5M_with_deflate"]),
            float(adaptive["matrix_median_storage_over_raw_0p5M"]),
            independent_median,
            float(temporal["temporal_storage_over_raw_0p5"]["median"]),
            float(final["temporal_storage_over_raw_0p5"]["median"]),
        ],
    )


def plot_storage_results(output_dir: Path) -> None:
    _, values = storage_progression()
    crossconfig = load_json(
        PROJECT_ROOT / "experiments" / "dense_dump_codec_crossconfig_storage.json"
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(PAPER_FULL_WIDTH_IN, 3.25),
        gridspec_kw={"width_ratios": [1.0, 1.08]},
    )
    figure.subplots_adjust(left=0.18, right=0.985, bottom=0.18, top=0.89, wspace=0.38)

    stage_labels = (
        "Original",
        "Residual coding",
        "Temporal residuals",
        "Independent anchors",
        "Temporal anchors",
        "Final DDC",
    )
    colors = ["#b0bec5", "#90caf9", "#4db6ac", "#81c784", "#ffb74d", "#ef5350"]
    positions = np.arange(len(values))
    bars = axes[0].barh(positions, values, color=colors, edgecolor="white", linewidth=0.7)
    axes[0].axvline(1.0, color="#546e7a", linewidth=0.9, linestyle="--")
    axes[0].set_yticks(positions, stage_labels, fontsize=7.6)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Storage ratio", fontsize=PAPER_FONT_SIZE)
    axes[0].set_xlim(0, 1.10)
    axes[0].set_title("(a) Development matrix", loc="left", fontsize=9.5, fontweight="bold")
    axes[0].tick_params(axis="x", labelsize=PAPER_SMALL_FONT_SIZE)
    axes[0].grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(
            value + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=7.1,
        )

    case_labels = [row["label"] for row in crossconfig["cases"]]
    dense_gib = [row["truth_bytes"] / 2**30 for row in crossconfig["cases"]]
    original_gib = [row["raw_bytes"] / 2**30 for row in crossconfig["cases"]]
    codec_gib = [row["codec_bytes"] / 2**30 for row in crossconfig["cases"]]
    compatibility_gib = [
        row.get("compatibility_bytes", row["codec_bytes"]) / 2**30
        for row in crossconfig["cases"]
    ]
    x_values = np.arange(len(case_labels))
    width = 0.19
    axes[1].bar(x_values - 1.5 * width, dense_gib, width, label=r"Dense $0.1M$", color="#90caf9")
    axes[1].bar(x_values - 0.5 * width, original_gib, width, label=r"Original $0.5M$", color="#b0bec5")
    axes[1].bar(x_values + 0.5 * width, codec_gib, width, label="DDC", color="#ef5350")
    axes[1].bar(
        x_values + 1.5 * width,
        compatibility_gib,
        width,
        label="DDC + original anchors",
        color="#ffb74d",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(6.8, 145)
    axes[1].set_ylabel("Storage (GiB)", fontsize=PAPER_FONT_SIZE)
    axes[1].set_xticks(x_values, case_labels, fontsize=7.8)
    axes[1].tick_params(axis="y", labelsize=PAPER_SMALL_FONT_SIZE)
    axes[1].set_title("(b) 801-frame sequences", loc="left", fontsize=9.5, fontweight="bold")
    axes[1].grid(axis="y", which="both", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7.1, ncol=2, loc="upper center")

    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"ddc_storage_results.{suffix}",
            dpi=220,
            facecolor="white",
        )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.figure or ("overview", "storage"))
    if "overview" in selected:
        plot_method_overview(args.output_dir)
    if "storage" in selected:
        plot_storage_results(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
