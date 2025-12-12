#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "seaborn",
#   "typer",
# ]
# ///
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

app = typer.Typer(help="Plot bucket size benchmark results")


def normalize_benchmark_name(name: str) -> str:
    """Convert FixtureName/BenchmarkName/... to FixtureName_BenchmarkName/..."""
    parts = name.split("/")
    if len(parts) >= 2 and "Fixture" in parts[0]:
        # Convert "BucketSizeFixture/Insert<4>/..." to "BucketSize_Insert<4>/..."
        fixture_name = parts[0].replace("Fixture", "")
        bench_name = parts[1]
        parts[0] = f"{fixture_name}_{bench_name}"
        parts.pop(1)  # Remove the benchmark name since it's now in parts[0]
    return "/".join(parts)


def parse_benchmark_name(name: str) -> pd.Series:
    # Pattern: BS<BucketSize>_<Operation>/<InputSize>/min_time:<MinTime>/repeats:<Repetitions>_<stat>
    # Extract the operation and input size
    match = re.match(r"BS\d+_(\w+)/(\d+)", name)
    if match:
        operation = match.group(1)
        input_size = int(match.group(2))
        return pd.Series(
            {
                "operation": operation,
                "input_size": input_size,
                "exponent": int(np.log2(input_size)),
            }
        )
    return pd.Series({"operation": None, "input_size": None, "exponent": None})


def create_performance_heatmap(df: pd.DataFrame, operation: str, ax):
    subset = df[df["operation"] == operation].copy()

    subset["normalized_time"] = subset.groupby("exponent")["time_ms"].transform(
        lambda x: x / x.min()
    )

    pivot_table = subset.pivot(
        index="exponent",
        columns="bucket_size",
        values="normalized_time",
    )

    sns.heatmap(
        pivot_table,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        cbar_kws={"label": "Performance Ratio (1.0 = Optimal)"},
        vmin=1.0,
        vmax=pivot_table.max().max() if pivot_table.max().max() > 1.0 else 2.0,
    )

    ax.set_title(
        f"{operation} Performance vs. Bucket Size", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Bucket Size", fontsize=12)
    ax.set_ylabel("Input Size", fontsize=12)
    ax.set_yticklabels([f"$2^{{{int(exp)}}}$" for exp in pivot_table.index], rotation=0)


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
    Generate bucket size performance heatmap plots from benchmark CSV results.

    Shows normalized performance for Insert and Query operations across different
    bucket sizes and input sizes.

    Examples:
        cat results.csv | plot_bucket_size.py
        plot_bucket_size.py < results.csv
        plot_bucket_size.py results.csv
        plot_bucket_size.py results.csv -o custom/dir
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

    # Filter for median records only and normalize names
    df = df[df["name"].str.endswith("_median")]
    df["name"] = df["name"].apply(normalize_benchmark_name)

    parsed = df["name"].apply(parse_benchmark_name)
    df = pd.concat([df, parsed], axis=1)

    # bucket_size comes as float from json
    df["bucket_size"] = df["bucket_size"].astype(int)

    df_filtered = df[df["operation"].isin(["Insert", "Query"])].copy()

    df_filtered["time_ms"] = df_filtered["real_time"]
    df_filtered["throughput_mops"] = df_filtered["items_per_second"] / 1_000_000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    create_performance_heatmap(df_filtered, "Insert", ax1)
    create_performance_heatmap(df_filtered, "Query", ax2)

    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "benchmark_bucket_size.pdf"

    plt.tight_layout()
    plt.savefig(
        output_file,
        bbox_inches="tight",
        edgecolor="none",
        transparent=True,
        format="pdf",
        dpi=600,
    )

    typer.secho(
        f"Bucket size performance plot saved to {output_file}", fg=typer.colors.GREEN
    )


if __name__ == "__main__":
    app()
