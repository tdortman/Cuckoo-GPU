#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import typer

app = typer.Typer(help="Plot runtime benchmark results")


def normalize_benchmark_name(name: str) -> str:
    """Convert FixtureName/BenchmarkName/... to FixtureName_BenchmarkName/..."""
    parts = name.split("/")
    if len(parts) >= 2 and "Fixture" in parts[0]:
        # Convert "CFFixture/Insert/..." to "CF_Insert/..."
        fixture_name = parts[0].replace("Fixture", "")
        bench_name = parts[1]
        parts[0] = f"{fixture_name}_{bench_name}"
        parts.pop(1)  # Remove the benchmark name since it's now in parts[0]
    return "/".join(parts)


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
    Generate runtime comparison plots from benchmark CSV results.

    Shows execution time vs input size for various benchmarks.

    Examples:
        cat results.csv | plot_runtime.py
        plot_runtime.py < results.csv
        plot_runtime.py results.csv
        plot_runtime.py results.csv -o custom/dir
    """
    try:
        if str(csv_file) == "-":
            df = pd.read_csv(sys.stdin)
        else:
            df = pd.read_csv(csv_file)
    except Exception as e:
        typer.secho(f"Error parsing CSV: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    benchmark_data = defaultdict(dict)

    for _, row in df.iterrows():
        name = normalize_benchmark_name(row["name"])
        if "/" not in name:
            continue

        # Extract base_name and size from name
        parts = name.split("/")
        if len(parts) < 2:
            continue

        base_name = parts[0]
        size_str = parts[1]

        if "FPR" in base_name or "InsertQueryDelete" in base_name:
            continue

        try:
            size = int(size_str)
            real_time = row.get("real_time", 0)
            if pd.notna(real_time):
                benchmark_data[base_name][size] = real_time
        except (ValueError, KeyError):
            continue

    if not benchmark_data:
        typer.secho("No benchmark data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    def get_last_value(bench_name):
        sizes = sorted(benchmark_data[bench_name].keys())
        if sizes:
            return benchmark_data[bench_name][sizes[-1]]
        return 0

    benchmark_names = sorted(benchmark_data.keys(), key=get_last_value, reverse=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle("Runtime Comparison", fontsize=16, fontweight="bold")

    for bench_name in benchmark_names:
        sizes = sorted(benchmark_data[bench_name].keys())
        times = [benchmark_data[bench_name][size] for size in sizes]

        ax.plot(sizes, times, "o-", label=bench_name, linewidth=2.5, markersize=8)

    ax.set_xlabel("Input Size", fontsize=14, fontweight="bold")
    ax.set_ylabel("Runtime (ms)", fontsize=14, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=10, loc="best", framealpha=0)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()

    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "benchmark_runtime.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight", transparent=True)
    typer.secho(f"Plot saved to {output_file}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
