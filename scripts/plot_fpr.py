#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import typer

app = typer.Typer(help="Plot false positive rate benchmark results")


def normalize_benchmark_name(name: str) -> str:
    """Convert FixtureName/BenchmarkName/... to FixtureName_BenchmarkName/..."""
    parts = name.split("/")
    if len(parts) >= 2 and "Fixture" in parts[0]:
        # Convert "CFFixture/FPR/..." to "CF_FPR/..."
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
    Generate false positive rate plots from benchmark CSV results.

    Creates two plots: FPR percentage and total false positives count.

    Examples:
        cat results.csv | plot_fpr.py
        plot_fpr.py < results.csv
        plot_fpr.py results.csv
        plot_fpr.py results.csv -o custom/dir
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

    fpr_data = defaultdict(dict)
    false_positives_data = defaultdict(dict)

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

        if "FalsePositiveRate" not in base_name and "FPR" not in base_name:
            continue

        try:
            size = int(size_str)
            fpr_percentage = row.get("fpr_percentage")
            false_positives = row.get("false_positives")

            if pd.notna(fpr_percentage):
                fpr_data[base_name][size] = fpr_percentage
            if pd.notna(false_positives):
                false_positives_data[base_name][size] = false_positives

        except (ValueError, KeyError):
            continue

    if not fpr_data and not false_positives_data:
        typer.secho(
            "No false positive rate data found in csv", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(1)

    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    if fpr_data:

        def get_last_fpr_value(bench_name):
            sizes = sorted(fpr_data[bench_name].keys())
            if sizes:
                return fpr_data[bench_name][sizes[-1]]
            return 0

        benchmark_names = sorted(fpr_data.keys(), key=get_last_fpr_value, reverse=True)

        for bench_name in benchmark_names:
            sizes = sorted(fpr_data[bench_name].keys())
            fpr = [fpr_data[bench_name][size] for size in sizes]
            ax1.plot(sizes, fpr, "o-", label=bench_name, linewidth=2, markersize=6)

        ax1.set_xlabel("Input Size", fontsize=12)
        ax1.set_ylabel("False Positive Rate (%)", fontsize=12)
        ax1.set_xscale("log", base=2)
        ax1.legend(fontsize=10, loc="best")
        ax1.grid(True, which="both", ls="--", alpha=0.5)
        ax1.set_title("False Positive Rate Percentage", fontsize=14)

    if false_positives_data:

        def get_last_fp_value(bench_name):
            sizes = sorted(false_positives_data[bench_name].keys())
            if sizes:
                return false_positives_data[bench_name][sizes[-1]]
            return 0

        benchmark_names = sorted(
            false_positives_data.keys(), key=get_last_fp_value, reverse=True
        )

        for bench_name in benchmark_names:
            sizes = sorted(false_positives_data[bench_name].keys())
            fp = [false_positives_data[bench_name][size] for size in sizes]
            ax2.plot(sizes, fp, "o-", label=bench_name, linewidth=2, markersize=6)

        ax2.set_xlabel("Input Size", fontsize=12)
        ax2.set_ylabel("Total False Positives", fontsize=12)
        ax2.set_xscale("log", base=2)
        ax2.set_yscale("log")
        ax2.legend(fontsize=10, loc="best")
        ax2.grid(True, which="both", ls="--", alpha=0.5)
        ax2.set_title("Total False Positives Count", fontsize=14)

    plt.tight_layout()
    output_file = output_dir / "benchmark_false_positives.png"
    plt.savefig(output_file, dpi=150)
    typer.secho(f"False positive plot saved to {output_file}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
