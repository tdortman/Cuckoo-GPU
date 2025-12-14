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

app = typer.Typer(help="Plot eviction benchmark results")


def extract_eviction_policy(name: str) -> Optional[str]:
    """Extract eviction policy from benchmark name like BFSFixture/Evictions/..."""
    if name.startswith("BFSFixture"):
        return "BFS"
    elif name.startswith("DFSFixture"):
        return "DFS"
    return None


def extract_capacity(name: str) -> Optional[int]:
    """Extract capacity from benchmark name like BFSFixture/Evictions/16777216/..."""
    match = re.search(r"/Evictions/(\d+)/", name)
    if match:
        return int(match.group(1))
    return None


def extract_load_factor(row: pd.Series) -> Optional[float]:
    """Extract load factor from benchmark counter."""
    lf = row.get("load_factor")
    if pd.notna(lf):
        return float(lf) / 100.0
    return None


def load_csv_data(csv_path: Path) -> tuple[dict, dict, dict, Optional[int]]:
    """Load and parse benchmark data from a CSV file.

    Returns a tuple of (eviction_data, total_evictions_data, throughput_data, capacity)
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        typer.secho(f"Error parsing CSV {csv_path}: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    # Dictionary: policy -> {load_factor: value}
    eviction_data = defaultdict(dict)
    total_evictions_data = defaultdict(dict)
    throughput_data = defaultdict(dict)
    capacity = None

    for _, row in df.iterrows():
        name = row["name"]

        policy = extract_eviction_policy(name)
        load_factor = extract_load_factor(row)
        row_capacity = extract_capacity(name)

        if policy is None or load_factor is None:
            continue

        # Track capacity (assumes all rows have the same capacity)
        if row_capacity is not None and capacity is None:
            capacity = row_capacity

        evictions_per_insert = row.get("evictions_per_insert")
        evictions = row.get("evictions")
        items_per_second = row.get("items_per_second")

        if pd.notna(evictions_per_insert):
            eviction_data[policy][load_factor] = evictions_per_insert
        if pd.notna(evictions):
            total_evictions_data[policy][load_factor] = evictions
        if pd.notna(items_per_second):
            throughput_data[policy][load_factor] = items_per_second

    return eviction_data, total_evictions_data, throughput_data, capacity


def format_capacity_title(base_title: str, capacity: Optional[int]) -> str:
    """Format title with capacity as power of 2."""
    if capacity is not None:
        power = int(math.log2(capacity))
        return f"{base_title} $\\left(n=2^{{{power}}}\\right)$"
    return base_title


@app.command()
def main(
    csv_file_top: Path = typer.Argument(
        ...,
        help="Path to CSV file for top plot",
    ),
    csv_file_middle: Path = typer.Argument(
        ...,
        help="Path to CSV file for middle plot",
    ),
    csv_file_bottom: Path = typer.Argument(
        ...,
        help="Path to CSV file for bottom plot",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
    label_top: Optional[str] = typer.Option(
        None,
        "--label-top",
        "-lt",
        help="Label to append to top plot title",
    ),
    label_middle: Optional[str] = typer.Option(
        None,
        "--label-middle",
        "-lm",
        help="Label to append to middle plot title",
    ),
    label_bottom: Optional[str] = typer.Option(
        None,
        "--label-bottom",
        "-lb",
        help="Label to append to bottom plot title",
    ),
):
    """
    Generate eviction throughput plots from three benchmark CSV files.

    Creates a single figure with three vertically stacked throughput plots.

    Examples:
        plot_eviction.py small.csv medium.csv large.csv
        plot_eviction.py small.csv medium.csv large.csv -o custom/dir
    """
    # Load data from all three CSV files
    csv_files = [
        (csv_file_top, "top", label_top),
        (csv_file_middle, "middle", label_middle),
        (csv_file_bottom, "bottom", label_bottom),
    ]

    data_list = []
    for csv_file, position, label in csv_files:
        eviction_data, total_evictions_data, throughput_data, capacity = load_csv_data(
            csv_file
        )
        if not throughput_data:
            typer.secho(
                f"No throughput data found in {csv_file} ({position})",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        data_list.append((throughput_data, capacity, label))

    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)

    policy_styles = {
        "BFS": {"color": "#2E86AB", "marker": "o", "linestyle": "-"},
        "DFS": {"color": "#A23B72", "marker": "s", "linestyle": "--"},
    }

    # Create figure with 3 vertically stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    all_handles = []
    all_labels = []

    for idx, (ax, (throughput_data, capacity, label)) in enumerate(
        zip(axes, data_list)
    ):
        for policy in sorted(throughput_data.keys()):
            load_factors = sorted(throughput_data[policy].keys())
            throughputs = [
                throughput_data[policy][lf] / 1e6 for lf in load_factors
            ]  # Convert to millions

            style = policy_styles.get(policy, {"marker": "o", "linestyle": "-"})
            (line,) = ax.plot(
                load_factors,
                throughputs,
                label=policy,
                linewidth=2.5,
                markersize=8,
                color=style.get("color"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
            )

            # Collect handles/labels from first plot only
            if idx == 0 and policy not in all_labels:
                all_handles.append(line)
                all_labels.append(policy)

        # Only show x-label on bottom plot
        if idx == 2:
            ax.set_xlabel("Load Factor", fontsize=14, fontweight="bold")

        ax.set_ylabel("Throughput [M ops/s]", fontsize=14, fontweight="bold")
        ax.grid(True, which="both", ls="--", alpha=0.3)

        # Build title
        title = format_capacity_title("Insert Throughput", capacity)
        if label:
            title += f" ({label})"
        ax.set_title(title, fontsize=16, fontweight="bold")

    plt.tight_layout()

    # Create combined legend outside on the right
    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            fontsize=10,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            framealpha=0,
        )

    output_file = output_dir / "eviction_throughput.pdf"
    plt.savefig(
        output_file,
        bbox_inches="tight",
        transparent=True,
        format="pdf",
        dpi=600,
    )
    typer.secho(f"Throughput plot saved to {output_file}", fg=typer.colors.GREEN)
    plt.close()


if __name__ == "__main__":
    app()
