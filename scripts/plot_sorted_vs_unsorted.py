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

Compares three insertion methods:
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
            throughput_mops = items_per_second / 1_000_000  # Convert to M ops/s
            benchmark_data[bench_name][size] = throughput_mops

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


@app.command()
def main(
    csv_file: Path = typer.Argument(
        ...,
        help="Path to benchmark CSV file",
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

    The plot shows throughput (G ops/s) vs input size for:
    - Unsorted: Direct insertion
    - Sorted: Insertion with sorting (sort time included in measurement)
    - Presorted: Insertion with pre-sorted keys (sort time excluded)
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        typer.secho(f"Error reading CSV: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    benchmark_data = extract_benchmark_data(df)

    if not benchmark_data:
        typer.secho("No benchmark data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

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

    # Define plotting order (unsorted first, then sorted variants)
    plot_order = ["InsertUnsorted", "InsertSorted", "InsertPresorted"]

    for bench_name in plot_order:
        if bench_name not in benchmark_data:
            continue

        sizes = sorted(benchmark_data[bench_name].keys())
        throughput = [benchmark_data[bench_name][size] for size in sizes]

        display_name = DISPLAY_NAMES.get(bench_name, bench_name)
        color = COLORS.get(bench_name, None)

        ax.plot(
            sizes,
            throughput,
            "o-",
            label=display_name,
            color=color,
            linewidth=pu.LINE_WIDTH,
            markersize=pu.MARKER_SIZE,
        )

    ax.set_xlabel(
        "Capacity (elements)", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax.set_ylabel(
        "Throughput [M ops/s]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax.set_xscale("log", base=2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(labels),
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)

    plt.tight_layout(rect=(0, 0, 1, 0.92))

    output_file = output_dir / "sorted_vs_unsorted.pdf"
    pu.save_figure(None, output_file, f"Plot saved to {output_file}")


if __name__ == "__main__":
    app()
