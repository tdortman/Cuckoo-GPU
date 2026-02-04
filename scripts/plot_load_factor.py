#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot load factor benchmark results")


def extract_load_factor(name: str) -> Optional[float]:
    """Extract load factor from benchmark name like CF_5/Insert, BBF_95/Query, or CF_99_5/Insert"""
    # Match patterns like _95/ or _99_5/ (where underscore represents decimal point)
    match = re.search(r"_([\d_]+)/", name)
    if match:
        # Replace underscore with decimal point (e.g., "99_5" -> "99.5")
        value_str = match.group(1).replace("_", ".")
        return float(value_str) / 100.0
    return None


def extract_operation_type(name: str) -> Optional[str]:
    """Extract operation type from benchmark name"""
    if "/Insert/" in name:
        return "Insert"
    elif "/QueryNegative/" in name:
        return "Query"
    elif "/Query/" in name:
        return "Query"
    elif "/Delete/" in name:
        return "Delete"
    return None


def extract_lookup_type(name: str) -> Optional[str]:
    """Extract lookup type (Positive or Negative) for Query operations"""
    if "/QueryNegative/" in name:
        return "Negative"
    elif "/Query/" in name:
        return "Positive"
    return None


def load_csv_data(csv_path: Path) -> dict:
    """Load and parse benchmark data from a CSV file.

    Returns:
        Dictionary structure: operation -> filter_type -> {load_factor: throughput}
    """
    df = pu.load_csv(csv_path)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    # Dictionary structure: operation -> filter_type -> {load_factor: throughput}
    benchmark_data = defaultdict(lambda: defaultdict(dict))

    for _, row in df.iterrows():
        name = row["name"]

        # Extract filter type using standardized approach
        filter_key = pu.normalize_benchmark_name(name)
        filter_type = pu.get_filter_display_name(filter_key)

        if not filter_type:
            continue

        load_factor = extract_load_factor(name)
        operation_type = extract_operation_type(name)
        lookup_type = extract_lookup_type(name)

        if filter_type is None or load_factor is None or operation_type is None:
            continue

        # For Query operations, append lookup type to filter name
        if operation_type == "Query" and lookup_type:
            filter_key = f"{filter_type} ({lookup_type})"
        else:
            filter_key = filter_type

        items_per_second = row.get("items_per_second")
        if pd.notna(items_per_second):
            throughput_mops = items_per_second / 1_000_000
            benchmark_data[operation_type][filter_key][load_factor] = throughput_mops

    return benchmark_data


def get_filter_styles() -> dict:
    """Define colors and markers for each filter type, with positive/negative variants."""
    # Use the standardized filter styles from plot_utils as base
    base_styles = {
        "GPU Cuckoo": pu.FILTER_STYLES.get("gcf", {"color": "#2E86AB", "marker": "o"}),
        "CPU Cuckoo": pu.FILTER_STYLES.get("ccf", {"color": "#00B4D8", "marker": "o"}),
        "Blocked Bloom": pu.FILTER_STYLES.get(
            "bbf", {"color": "#A23B72", "marker": "s"}
        ),
        "Two-Choice": pu.FILTER_STYLES.get("tcf", {"color": "#C73E1D", "marker": "v"}),
        "GPU Quotient": pu.FILTER_STYLES.get(
            "gqf", {"color": "#F18F01", "marker": "^"}
        ),
        "Partitioned Cuckoo": pu.FILTER_STYLES.get(
            "pcf", {"color": "#6A994E", "marker": "D"}
        ),
    }

    # Generate styles for both positive and negative variants
    filter_styles = {}
    for filter_name, base_style in base_styles.items():
        # Base style (for non-query operations)
        filter_styles[filter_name] = {
            "color": base_style["color"],
            "marker": base_style["marker"],
            "linestyle": "-",
        }
        # Positive lookups: solid line
        filter_styles[f"{filter_name} (Positive)"] = {
            "color": base_style["color"],
            "marker": base_style["marker"],
            "linestyle": "-",
        }
        # Negative lookups: dashed line
        filter_styles[f"{filter_name} (Negative)"] = {
            "color": base_style["color"],
            "marker": base_style["marker"],
            "linestyle": "--",
        }

    return filter_styles


def plot_operation_on_axis(
    ax: plt.Axes,
    operation_type: str,
    benchmark_data: dict,
    filter_styles: dict,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
) -> tuple[list, list]:
    """Plot a single operation's data on the given axis.

    Returns:
        A tuple of (handles, labels) for use in creating a combined legend.
        Labels are simplified to base filter names (without Positive/Negative suffixes).
    """
    handles = []
    labels = []
    seen_base_filters = set()

    for filter_type in sorted(benchmark_data[operation_type].keys()):
        load_factors = sorted(benchmark_data[operation_type][filter_type].keys())
        throughputs = [
            benchmark_data[operation_type][filter_type][lf] for lf in load_factors
        ]

        style = filter_styles.get(filter_type, {"marker": "o", "linestyle": "-"})
        (line,) = ax.plot(
            load_factors,
            throughputs,
            label=filter_type,
            linewidth=pu.LINE_WIDTH,
            markersize=pu.MARKER_SIZE,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
        )

        # Extract base filter name (strip Positive/Negative suffix)
        base_filter = filter_type.replace(" (Positive)", "").replace(" (Negative)", "")

        # Only add to legend if we haven't seen this base filter yet
        if base_filter not in seen_base_filters:
            handles.append(line)
            labels.append(base_filter)
            seen_base_filters.add(base_filter)

    if show_xlabel:
        ax.set_xlabel(
            "Load Factor", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
        )
    if show_ylabel:
        ax.set_ylabel(
            "Throughput [M ops/s]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
        )
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
    ax.set_yscale("log")

    return handles, labels


@app.command()
def main(
    csv_top_left: Path = typer.Argument(
        ...,
        help="Path to CSV file for top-left plot",
    ),
    csv_top_right: Path = typer.Argument(
        ...,
        help="Path to CSV file for top-right plot",
    ),
    csv_bottom_left: Path = typer.Argument(
        ...,
        help="Path to CSV file for bottom-left plot",
    ),
    csv_bottom_right: Path = typer.Argument(
        ...,
        help="Path to CSV file for bottom-right plot",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Generate throughput vs load factor comparison plots from four benchmark CSV files.

    Creates a 2x2 grid of plots for each operation (insert, query, delete) with data
    from each CSV file in the corresponding position.

    Examples:
        plot_load_factor.py tl.csv tr.csv bl.csv br.csv
        plot_load_factor.py tl.csv tr.csv bl.csv br.csv -o custom/dir
    """
    # Load data from all four CSV files
    csv_files = [
        (csv_top_left, "top-left"),
        (csv_top_right, "top-right"),
        (csv_bottom_left, "bottom-left"),
        (csv_bottom_right, "bottom-right"),
    ]

    benchmark_data_list = []

    for csv_file, position in csv_files:
        data = load_csv_data(csv_file)
        if not data:
            typer.secho(
                f"No throughput data found in {csv_file} ({position})",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        benchmark_data_list.append(data)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    filter_styles = get_filter_styles()

    # Get all operations from all datasets
    all_operations = set[str]()
    for data in benchmark_data_list:
        all_operations.update(data.keys())

    # Grid positions: (row, col)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for operation in sorted(all_operations):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")

        # Collect handles and labels for combined legend
        all_handles = []
        all_labels = []

        for data, (row, col) in zip(benchmark_data_list, positions):
            ax = axes[row, col]
            has_data = operation in data and data[operation]

            # Show ylabel only on left column, xlabel only on bottom row
            show_ylabel = col == 0
            show_xlabel = row == 1

            if has_data:
                handles, plot_labels = plot_operation_on_axis(
                    ax,
                    operation,
                    data,
                    filter_styles,
                    show_ylabel=show_ylabel,
                    show_xlabel=show_xlabel,
                )
                # Only add handles/labels that aren't already in the combined list
                for handle, lbl in zip(handles, plot_labels):
                    if lbl not in all_labels:
                        all_handles.append(handle)
                        all_labels.append(lbl)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

        # Create combined legend above the plots
        if all_handles:
            fig.legend(
                all_handles,
                all_labels,
                fontsize=pu.LEGEND_FONT_SIZE,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=len(all_labels),
                framealpha=pu.LEGEND_FRAME_ALPHA,
            )

        plt.tight_layout(rect=(0, 0, 1, 0.95))

        output_file = (
            output_dir / f"load_factor_{operation.lower().replace(' ', '_')}.pdf"
        )
        pu.save_figure(
            fig,
            output_file,
            f"{operation} throughput comparison plot saved to {output_file}",
            close=True,
        )


if __name__ == "__main__":
    app()
