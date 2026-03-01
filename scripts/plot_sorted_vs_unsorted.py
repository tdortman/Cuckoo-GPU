#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
"""
Plot sorted vs unsorted insertion benchmark results.

Compares three insertion methods across two hardware platforms:
- Unsorted: Direct insertion without sorting
- Sorted: Insertion with inline sorting (sort time included)
- Presorted: Insertion with pre-sorted keys (sort time excluded)
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot sorted vs unsorted insertion benchmark results")


def extract_benchmark_data(df: pd.DataFrame) -> dict[str, dict[int, float]]:
    """Extract throughput data from benchmark CSV."""
    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    benchmark_data: dict[str, dict[int, float]] = defaultdict(dict)

    for _, row in df.iterrows():
        name = row["name"]

        # Parse benchmark name: CF/InsertUnsorted/65536/...
        match = re.match(r"CF/(Insert\w+)/(\d+)/", name)
        if not match:
            continue

        bench_name = match.group(1)
        size = int(match.group(2))

        items_per_second = row.get("items_per_second")
        if pd.notna(items_per_second):
            throughput_beps = pu.to_billion_elems_per_sec(items_per_second)
            benchmark_data[bench_name][size] = throughput_beps

    return benchmark_data


def load_benchmark_data(csv_file: Path, hardware_label: str) -> dict[str, dict[int, float]]:
    """Load and extract benchmark data from one hardware CSV."""
    df = pu.load_csv(csv_file)

    required_columns = {"name", "items_per_second"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        typer.secho(
            f"CSV for {hardware_label} is missing required columns ({missing}): {csv_file}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    benchmark_data = extract_benchmark_data(df)
    if not benchmark_data:
        typer.secho(
            f"No benchmark data found in {hardware_label} CSV: {csv_file}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    return benchmark_data


# Display name mapping for prettier labels
DISPLAY_NAMES = {
    "InsertUnsorted": "Unsorted",
    "InsertSorted": "Sorted (incl. sort)",
    "InsertPresorted": "Presorted (excl. sort)",
}

# Color scheme for each method
COLORS = {
    "InsertUnsorted": "#E74C3C",  # Red
    "InsertSorted": "#3498DB",  # Blue
    "InsertPresorted": "#2ECC71",  # Green
}

HARDWARE_STYLES = {
    "HBM3": {"linestyle": "-", "alpha": 1.0, "hollow_marker": False},
    "GDDR7": {"linestyle": "--", "alpha": 0.9, "hollow_marker": True},
}


@app.command()
def main(
    hbm3_csv: Path = typer.Argument(
        ...,
        help="Path to HBM3 benchmark CSV file",
        exists=True,
    ),
    gddr7_csv: Path = typer.Argument(
        ...,
        help="Path to GDDR7 benchmark CSV file",
        exists=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plot (default: build/)",
    ),
):
    """
    Generate a throughput comparison plot for sorted vs unsorted insertion methods.

    The plot overlays both hardware platforms on one axis:
    - HBM3
    - GDDR7

    Each insertion method keeps one color across platforms:
    - Unsorted: Direct insertion
    - Sorted: Insertion with sorting (sort time included in measurement)
    - Presorted: Insertion with pre-sorted keys (sort time excluded)
    """
    hbm3_data = load_benchmark_data(hbm3_csv, "HBM3")
    gddr7_data = load_benchmark_data(gddr7_csv, "GDDR7")

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    # Create plot
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = {"HBM3": hbm3_data, "GDDR7": gddr7_data}

    # Define plotting order (unsorted first, then sorted variants).
    plot_order = ["InsertUnsorted", "InsertSorted", "InsertPresorted"]

    for bench_name in plot_order:
        display_name = DISPLAY_NAMES.get(bench_name, bench_name)
        color = COLORS.get(bench_name, None)

        for hardware_label in ["HBM3", "GDDR7"]:
            benchmark_data = datasets[hardware_label]
            if bench_name not in benchmark_data:
                continue

            style = HARDWARE_STYLES[hardware_label]
            sizes = sorted(benchmark_data[bench_name].keys())
            throughput = [benchmark_data[bench_name][size] for size in sizes]

            marker_face_color = color
            if style["hollow_marker"]:
                marker_face_color = "none"

            ax.plot(
                sizes,
                throughput,
                marker="o",
                linestyle=style["linestyle"],
                label=f"{display_name} ({hardware_label})",
                color=color,
                alpha=style["alpha"],
                linewidth=pu.LINE_WIDTH,
                markersize=pu.MARKER_SIZE,
                markerfacecolor=marker_face_color,
                markeredgewidth=1.5,
            )

    ax.set_xlabel(
        "Capacity (elements)", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax.set_ylabel(
        pu.THROUGHPUT_LABEL, fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax.set_xscale("log", base=2)

    _, labels = ax.get_legend_handles_labels()
    legend_columns = min(3, max(1, len(labels)))
    ax.legend(
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=legend_columns,
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)

    plt.tight_layout(rect=(0, 0, 1, 0.88))

    output_file = output_dir / "sorted_vs_unsorted.pdf"
    pu.save_figure(None, output_file, f"Plot saved to {output_file}")


if __name__ == "__main__":
    app()
