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
import typer

app = typer.Typer(help="Plot load factor benchmark results")


def extract_load_factor(name: str) -> Optional[float]:
    """Extract load factor from benchmark name like CF_5/Insert or BBF_95/Query"""
    match = re.search(r"_(\d+)/", name)
    if match:
        return int(match.group(1)) / 100.0
    return None


def extract_filter_type(name: str) -> Optional[str]:
    """Extract filter type from benchmark name"""
    # Format: CF_5/Insert or BBF_95/Query
    if name.startswith("CF_"):
        return "Cuckoo Filter"
    elif name.startswith("BBF_"):
        return "Bloom Filter"
    elif name.startswith("QF_"):
        return "Quotient Filter"
    elif name.startswith("TCF_"):
        return "TCF"
    elif name.startswith("PCF_"):
        return "Partitioned Cuckoo"
    return None


def extract_operation_type(name: str) -> Optional[str]:
    """Extract operation type (Insert, Query, Delete) from benchmark name"""
    if "/Insert/" in name:
        return "Insert"
    elif "/Query/" in name:
        return "Query"
    elif "/Delete/" in name:
        return "Delete"
    return None


@app.command()
def main(
    csv_file: Path = typer.Argument(
        "-",
        help="Path to CSV file, or '-' to read from stdin (default: stdin)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Generate throughput vs load factor plots from benchmark CSV results.

    Creates three plots: insert, query, and delete performance across different
    filter fill fractions for various AMQ implementations.

    Examples:
        cat results.csv | plot_load_factor.py
        plot_load_factor.py < results.csv
        plot_load_factor.py results.csv
        plot_load_factor.py results.csv -o custom/dir
    """
    try:
        if str(csv_file) == "-":
            import sys

            df = pd.read_csv(sys.stdin)
        else:
            df = pd.read_csv(csv_file)
    except Exception as e:
        typer.secho(f"Error parsing CSV: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    # Dictionary structure: operation -> filter_type -> {load_factor: throughput}
    benchmark_data = defaultdict(lambda: defaultdict(dict))

    for _, row in df.iterrows():
        name = row["name"]

        filter_type = extract_filter_type(name)
        load_factor = extract_load_factor(name)
        operation = extract_operation_type(name)

        if filter_type is None or load_factor is None or operation is None:
            continue

        items_per_second = row.get("items_per_second")
        if pd.notna(items_per_second):
            throughput_mops = items_per_second / 1_000_000
            benchmark_data[operation][filter_type][load_factor] = throughput_mops

    if not benchmark_data:
        typer.secho("No throughput data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Define colors and markers for each filter type
    filter_styles = {
        "Cuckoo Filter": {"color": "#2E86AB", "marker": "o", "linestyle": "-"},
        "Bloom Filter": {"color": "#A23B72", "marker": "s", "linestyle": "-"},
        "Quotient Filter": {"color": "#F18F01", "marker": "^", "linestyle": "-"},
        "TCF": {"color": "#C73E1D", "marker": "v", "linestyle": "-"},
        "Partitioned Cuckoo": {"color": "#6A994E", "marker": "D", "linestyle": "-"},
    }

    for operation in ["Insert", "Query", "Delete"]:
        if operation not in benchmark_data or not benchmark_data[operation]:
            typer.secho(
                f"No data for {operation} operation",
                fg=typer.colors.YELLOW,
                err=True,
            )
            continue

        fig, ax = plt.subplots(figsize=(12, 8))

        for filter_type in sorted(benchmark_data[operation].keys()):
            load_factors = sorted(benchmark_data[operation][filter_type].keys())
            throughputs = [
                benchmark_data[operation][filter_type][lf] for lf in load_factors
            ]

            style = filter_styles.get(filter_type, {})
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

        ax.set_xlabel("Filter Fill Fraction", fontsize=14, fontweight="bold")
        ax.set_ylabel("Throughput [M ops/s]", fontsize=14, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=12, loc="best", framealpha=0.9)
        ax.set_title(
            f"{operation} Performance",
            fontsize=16,
            fontweight="bold",
        )

        ax.set_yscale("log")

        plt.tight_layout()

        output_file = output_dir / f"load_factor_{operation.lower()}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        typer.secho(
            f"{operation} throughput plot saved to {output_file}",
            fg=typer.colors.GREEN,
        )
        plt.close()


if __name__ == "__main__":
    app()
