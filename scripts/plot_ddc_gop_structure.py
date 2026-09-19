#!/usr/bin/env python3
"""Draw the sequence-level and package-level structure of a DDC archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


PAPER_FULL_WIDTH_IN = 7.1
PAPER_FONT_SIZE = 9.0
PAPER_SMALL_FONT_SIZE = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", action="append", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def add_box(
    axis: plt.Axes,
    left: float,
    bottom: float,
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    fontsize: float = PAPER_SMALL_FONT_SIZE,
    linewidth: float = 1.0,
    pad: float = 0.006,
    rounding_size: float = 0.012,
    whiten: float = 0.92,
) -> FancyBboxPatch:
    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    facecolor = tuple(whiten + (1.0 - whiten) * rgb)
    box = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle=f"round,pad={pad},rounding_size={rounding_size}",
        linewidth=linewidth,
        edgecolor=color,
        facecolor=facecolor,
        zorder=3,
    )
    axis.add_patch(box)
    axis.text(
        left + width / 2,
        bottom + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#17212b",
        linespacing=1.18,
        zorder=4,
    )
    return box


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#52616b",
    linestyle: str = "-",
    mutation_scale: float = 8.5,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=0.95,
            linestyle=linestyle,
            color=color,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=2,
        )
    )


def add_routed_arrow(
    axis: plt.Axes,
    vertices: list[tuple[float, float]],
    *,
    color: str,
    linestyle: str = "-",
) -> None:
    path = MplPath(
        vertices,
        [MplPath.MOVETO, *([MplPath.LINETO] * (len(vertices) - 1))],
    )
    axis.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=0.95,
            linestyle=linestyle,
            color=color,
            joinstyle="round",
            zorder=2,
        )
    )


def panel_heading(axis: plt.Axes, vertical: float, label: str) -> None:
    axis.text(
        0.026,
        vertical,
        label,
        fontsize=PAPER_FONT_SIZE,
        fontweight="bold",
        color="#263746",
        va="top",
        zorder=5,
    )


def plot_sequence_layout(axis: plt.Axes) -> None:
    blue = "#1976d2"
    green = "#2e7d32"
    orange = "#ef6c00"
    neutral = "#607d8b"

    panel_heading(axis, 0.985, "(a) Sequence layout")
    add_box(
        axis,
        0.025,
        0.805,
        0.175,
        0.09,
        "Sequence manifest\nphysical times\nlogical index",
        green,
        fontsize=7.0,
        linewidth=1.15,
    )

    anchor_centers = (0.285, 0.500, 0.715, 0.930)
    anchor_times = (r"$t_0$", r"$t_K$", r"$t_{2K}$", r"$t_{3K}$")
    timeline_y = 0.750
    anchor_bottom = 0.815
    anchor_width = 0.088
    anchor_height = 0.075

    axis.plot(
        [anchor_centers[0], anchor_centers[-1]],
        [timeline_y, timeline_y],
        color=neutral,
        linewidth=1.15,
        zorder=1,
    )

    for anchor_index, (anchor_center, anchor_time) in enumerate(
        zip(anchor_centers, anchor_times, strict=True)
    ):
        axis.plot(
            [anchor_center, anchor_center],
            [timeline_y, anchor_bottom],
            color=blue,
            linewidth=0.9,
            zorder=1,
        )
        axis.scatter(
            [anchor_center],
            [timeline_y],
            s=20,
            facecolor=blue,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        add_box(
            axis,
            anchor_center - anchor_width / 2,
            anchor_bottom,
            anchor_width,
            anchor_height,
            f"$A_{anchor_index}$\noriginal exact",
            blue,
            fontsize=6.8,
            linewidth=1.25,
        )
        axis.text(
            anchor_center,
            0.785,
            anchor_time,
            ha="center",
            va="center",
            fontsize=7.0,
            color="#455a64",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},
            zorder=4,
        )
        axis.plot(
            [anchor_center, anchor_center],
            [timeline_y - 0.004, 0.575],
            color=blue,
            linewidth=0.75,
            linestyle=(0, (2, 2)),
            zorder=1,
        )

    for interval_index, (left_anchor, right_anchor) in enumerate(
        zip(anchor_centers[:-1], anchor_centers[1:], strict=True)
    ):
        intermediate_positions = np.linspace(left_anchor, right_anchor, 7)[1:-1]
        axis.scatter(
            intermediate_positions,
            np.full_like(intermediate_positions, timeline_y),
            s=14,
            facecolor="#d9e1e5",
            edgecolor=neutral,
            linewidth=0.55,
            zorder=2,
        )
        add_box(
            axis,
            left_anchor + 0.007,
            0.605,
            right_anchor - left_anchor - 0.014,
            0.085,
            f"GOP {interval_index}\nindexed residual chunks",
            orange,
            fontsize=6.9,
            linewidth=1.2,
        )

    axis.text(
        (anchor_centers[0] + anchor_centers[1]) / 2,
        0.772,
        r"$x_1,\ldots,x_{K-1}$",
        ha="center",
        fontsize=6.9,
        color="#455a64",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
        zorder=4,
    )
    axis.text(
        anchor_centers[1],
        0.558,
        "shared exact boundary",
        ha="center",
        va="top",
        fontsize=6.8,
        color=blue,
        zorder=4,
    )
    add_arrow(axis, (0.200, 0.850), (anchor_centers[0] - anchor_width / 2, 0.850), color=green)
    add_routed_arrow(
        axis,
        [(0.200, 0.817), (0.225, 0.790), (anchor_centers[0] - 0.002, 0.665)],
        color=green,
    )


def plot_container_layout(axis: plt.Axes) -> None:
    blue = "#1976d2"
    green = "#2e7d32"
    orange = "#ef6c00"
    red = "#c62828"
    dark = "#37474f"

    def content_y(value: float) -> float:
        return value - 0.025

    panel_heading(axis, 0.535, "(b) GOP container contents")
    add_box(
        axis,
        0.050,
        content_y(0.445),
        0.185,
        0.060,
        r"Original exact anchor  $A_i$",
        blue,
        fontsize=7.0,
        linewidth=1.1,
    )
    add_box(
        axis,
        0.765,
        content_y(0.445),
        0.185,
        0.060,
        r"Original exact anchor  $A_{i+1}$",
        blue,
        fontsize=7.0,
        linewidth=1.1,
    )

    container = FancyBboxPatch(
        (0.020, content_y(0.050)),
        0.960,
        0.355,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        linewidth=1.15,
        edgecolor=dark,
        facecolor="#fbfcfd",
        zorder=0,
    )
    axis.add_patch(container)
    axis.text(
        0.050,
        content_y(0.372),
        r"Indexed DDC residual chunks for GOP $i$",
        fontsize=7.9,
        fontweight="bold",
        color=dark,
        zorder=4,
    )
    axis.scatter(
        [0.285, 0.715],
        [content_y(0.408), content_y(0.408)],
        s=18,
        facecolor=blue,
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
    add_arrow(
        axis,
        (0.142, content_y(0.445)),
        (0.285, content_y(0.408)),
        color=blue,
        linestyle="--",
    )
    add_arrow(
        axis,
        (0.858, content_y(0.445)),
        (0.715, content_y(0.408)),
        color=blue,
        linestyle="--",
    )

    add_box(
        axis,
        0.055,
        content_y(0.105),
        0.225,
        0.215,
        "Logical index\n\nphysical time\nvariable / component\nspatial block",
        green,
        fontsize=7.2,
        linewidth=1.1,
    )

    axis.text(
        0.735,
        content_y(0.335),
        r"Independent chunks for channel/component $c$",
        ha="center",
        fontsize=7.0,
        color=dark,
        zorder=4,
    )
    grid_left = 0.560
    cell_width = 0.116
    cell_gap = 0.012
    column_labels = ("time chunk 0", "time chunk 1", r"$\cdots$")
    row_labels = (
        ("quantized codes", orange),
        ("block scales", blue),
        ("exception indices", red),
        ("exception values", red),
    )
    for column_index, column_label in enumerate(column_labels):
        cell_left = grid_left + column_index * (cell_width + cell_gap)
        axis.text(
            cell_left + cell_width / 2,
            content_y(0.298),
            column_label,
            ha="center",
            fontsize=6.4,
            color="#455a64",
            zorder=4,
        )

    row_bottoms = tuple(content_y(value) for value in (0.245, 0.195, 0.145, 0.095))
    for row_bottom, (row_label, row_color) in zip(row_bottoms, row_labels, strict=True):
        axis.text(
            0.545,
            row_bottom + 0.019,
            row_label,
            ha="right",
            va="center",
            fontsize=6.6,
            color="#455a64",
            zorder=4,
        )
        for column_index in range(len(column_labels)):
            cell_left = grid_left + column_index * (cell_width + cell_gap)
            add_box(
                axis,
                cell_left,
                row_bottom,
                cell_width,
                0.038,
                "" if column_index < 2 else r"$\cdots$",
                row_color,
                fontsize=6.2,
                linewidth=0.9,
                pad=0.002,
                rounding_size=0.006,
                whiten=0.94,
            )

    add_routed_arrow(
        axis,
        [
            (0.280, content_y(0.285)),
            (0.390, content_y(0.330)),
            (0.545, content_y(0.330)),
        ],
        color=green,
    )


def plot_structure(outputs: list[Path], dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(PAPER_FULL_WIDTH_IN, 4.20))
    figure.subplots_adjust(left=0.010, right=0.990, bottom=0.018, top=0.992)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    plot_sequence_layout(axis)
    plot_container_layout(axis)
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, facecolor="white")
        print(f"figure: {output}")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    plot_structure(args.output, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
