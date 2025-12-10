#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import typer

app = typer.Typer(help="Plot load factor benchmark results")


def extract_num_elements(name: str) -> Optional[int]:
    """Extract number of elements from benchmark name like CF_5/Insert/4194304/..."""
    # Match pattern like /Insert/4194304/ or /Query/4194304/
    match = re.search(r"/(?:Insert|Query|QueryNegative|Delete)/(\d+)/", name)
    if match:
        return int(match.group(1))
    return None


def extract_load_factor(name: str) -> Optional[float]:
    """Extract load factor from benchmark name like CF_5/Insert, BBF_95/Query, or CF_99_5/Insert"""
    # Match patterns like _95/ or _99_5/ (where underscore represents decimal point)
    match = re.search(r"_([\d_]+)/", name)
    if match:
        # Replace underscore with decimal point (e.g., "99_5" -> "99.5")
        value_str = match.group(1).replace("_", ".")
        return float(value_str) / 100.0
    return None


def extract_filter_type(name: str) -> Optional[str]:
    """Extract filter type from benchmark name"""
    # Format: CF_5/Insert or BBF_95/Query
    if name.startswith("CPUCF_"):
        return "CPU Cuckoo"
    elif name.startswith("CF_"):
        return "Cuckoo Filter"
    elif name.startswith("BBF_"):
        return "Blocked Bloom"
    elif name.startswith("QF_"):
        return "Quotient Filter"
    elif name.startswith("TCF_"):
        return "TCF"
    elif name.startswith("GQF_"):
        return "GQF"
    elif name.startswith("PCF_"):
        return "Partitioned Cuckoo"
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

    Returns a tuple of (benchmark_data, num_elements_per_operation)
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        typer.secho(f"Error parsing CSV {csv_path}: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    # Dictionary structure: operation -> filter_type -> {load_factor: throughput}
    benchmark_data = defaultdict(lambda: defaultdict(dict))
    # Track number of elements per operation
    num_elements_per_operation = {}

    for _, row in df.iterrows():
        name = row["name"]

        filter_type = extract_filter_type(name)
        load_factor = extract_load_factor(name)
        operation = extract_operation_type(name)
        lookup_type = extract_lookup_type(name)
        num_elements = extract_num_elements(name)

        if filter_type is None or load_factor is None or operation is None:
            continue

        # Store the number of elements for this operation (assumes all rows have same count)
        if operation and num_elements and operation not in num_elements_per_operation:
            num_elements_per_operation[operation] = num_elements

        # For Query operations, append lookup type to filter name
        if operation == "Query" and lookup_type:
            filter_key = f"{filter_type} ({lookup_type})"
        else:
            filter_key = filter_type

        items_per_second = row.get("items_per_second")
        if pd.notna(items_per_second):
            throughput_mops = items_per_second / 1_000_000
            benchmark_data[operation][filter_key][load_factor] = throughput_mops

    return benchmark_data, num_elements_per_operation


def get_filter_styles() -> dict:
    """Define colors and markers for each filter type."""
    base_styles = {
        "Cuckoo Filter": {"color": "#2E86AB", "marker": "o"},
        "CPU Cuckoo": {"color": "#00B4D8", "marker": "o"},
        "Blocked Bloom": {"color": "#A23B72", "marker": "s"},
        "TCF": {"color": "#C73E1D", "marker": "v"},
        "GQF": {"color": "#F18F01", "marker": "^"},
        "Partitioned Cuckoo": {"color": "#6A994E", "marker": "D"},
    }

    # Generate styles for both positive and negative variants
    filter_styles = {}
    for filter_name, base_style in base_styles.items():
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
        # Base filter (for non-query operations)
        filter_styles[filter_name] = {
            "color": base_style["color"],
            "marker": base_style["marker"],
            "linestyle": "-",
        }

    return filter_styles


def plot_operation_on_axis(
    ax: plt.Axes,
    operation: str,
    benchmark_data: dict,
    num_elements_per_operation: dict,
    filter_styles: dict,
    show_ylabel: bool = True,
    show_legend: bool = True,
    title_suffix: Optional[str] = None,
):
    """Plot a single operation's data on the given axis."""
    for filter_type in sorted(benchmark_data[operation].keys()):
        load_factors = sorted(benchmark_data[operation][filter_type].keys())
        throughputs = [
            benchmark_data[operation][filter_type][lf] for lf in load_factors
        ]

        style = filter_styles.get(filter_type, {"marker": "o", "linestyle": "-"})
        ax.plot(
            load_factors,
            throughputs,
            label=filter_type,
            linewidth=2.5,
            markersize=8,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
        )

    ax.set_xlabel("Load Factor", fontsize=14, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Throughput [M ops/s]", fontsize=14, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    if show_legend:
        ax.legend(fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5), framealpha=0)

    # Build title with element count and optional suffix
    title = f"{operation} Performance"
    if operation in num_elements_per_operation:
        n = num_elements_per_operation[operation]
        # Calculate power of 2
        power = int(math.log2(n))
        title += f" $\\left(n=2^{{{power}}}\\right)$"
    if title_suffix:
        title += f" ({title_suffix})"

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
    )

    ax.set_yscale("log")


@app.command()
def main(
    csv_file_left: Path = typer.Argument(
        ...,
        help="Path to first CSV file (left plot)",
    ),
    csv_file_right: Path = typer.Argument(
        ...,
        help="Path to second CSV file (right plot)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
    label_left: Optional[str] = typer.Option(
        None,
        "--label-left",
        "-ll",
        help="Label to append to left plot titles (e.g., 'HBM3')",
    ),
    label_right: Optional[str] = typer.Option(
        None,
        "--label-right",
        "-lr",
        help="Label to append to right plot titles (e.g., 'GDDR7')",
    ),
):
    """
    Generate throughput vs load factor comparison plots from two benchmark CSV files.

    Creates side-by-side plots for each operation (insert, query, delete) with data
    from the first CSV on the left and second CSV on the right.

    Examples:
        plot_load_factor.py results1.csv results2.csv
        plot_load_factor.py results1.csv results2.csv -o custom/dir
    """
    # Load data from both CSV files
    benchmark_data_left, num_elements_left = load_csv_data(csv_file_left)
    benchmark_data_right, num_elements_right = load_csv_data(csv_file_right)

    if not benchmark_data_left:
        typer.secho(
            f"No throughput data found in {csv_file_left}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    if not benchmark_data_right:
        typer.secho(
            f"No throughput data found in {csv_file_right}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)

    filter_styles = get_filter_styles()

    # Get all operations from both datasets
    all_operations = set(benchmark_data_left.keys()) | set(benchmark_data_right.keys())

    for operation in sorted(all_operations):
        has_left = operation in benchmark_data_left and benchmark_data_left[operation]
        has_right = (
            operation in benchmark_data_right and benchmark_data_right[operation]
        )

        if not has_left and not has_right:
            typer.secho(
                f"No data for {operation} operation in either file",
                fg=typer.colors.YELLOW,
                err=True,
            )
            continue

        fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

        # Left plot
        if has_left:
            plot_operation_on_axis(
                axes[0],
                operation,
                benchmark_data_left,
                num_elements_left,
                filter_styles,
                show_ylabel=True,
                show_legend=False,
                title_suffix=label_left,
            )
        else:
            axes[0].set_title(f"{operation} (No data)", fontsize=16, fontweight="bold")
            axes[0].text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )

        # Right plot
        if has_right:
            plot_operation_on_axis(
                axes[1],
                operation,
                benchmark_data_right,
                num_elements_right,
                filter_styles,
                show_ylabel=False,
                show_legend=True,
                title_suffix=label_right,
            )
        else:
            axes[1].set_title(f"{operation} (No data)", fontsize=16, fontweight="bold")
            axes[1].text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )

        plt.tight_layout()

        output_file = (
            output_dir / f"load_factor_{operation.lower().replace(' ', '_')}.png"
        )
        plt.savefig(output_file, dpi=150, bbox_inches="tight", transparent=True)
        typer.secho(
            f"{operation} throughput comparison plot saved to {output_file}",
            fg=typer.colors.GREEN,
        )
        plt.close()


if __name__ == "__main__":
    app()
