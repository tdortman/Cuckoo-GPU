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

app = typer.Typer(help="Plot memory usage benchmark results")


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
    Generate memory usage plots from benchmark CSV results.

    Creates two plots: total memory usage and bits per item efficiency metric.

    Examples:
        cat results.csv | plot_memory.py
        plot_memory.py < results.csv
        plot_memory.py results.csv
        plot_memory.py results.csv -o custom/dir
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

    memory_data = defaultdict(dict)
    bits_per_item_data = defaultdict(dict)

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

        if (
            "Insert" not in base_name
            or "InsertAndQuery" in base_name
            or "InsertQueryDelete" in base_name
            or "FalsePositiveRate" in base_name
        ):
            continue

        try:
            size = int(size_str)

            memory_bytes = row.get("memory_bytes")
            bits_per_item = row.get("bits_per_item")

            if pd.notna(memory_bytes):
                memory_data[base_name][size] = memory_bytes
            if pd.notna(bits_per_item):
                bits_per_item_data[base_name][size] = bits_per_item

        except (ValueError, KeyError):
            continue

    if not memory_data and not bits_per_item_data:
        typer.secho("No memory data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    if memory_data:

        def get_last_memory_value(bench_name):
            sizes = sorted(memory_data[bench_name].keys())
            if sizes:
                return memory_data[bench_name][sizes[-1]]
            return 0

        benchmark_names = sorted(
            memory_data.keys(), key=get_last_memory_value, reverse=True
        )
        for bench_name in benchmark_names:
            sizes = sorted(memory_data[bench_name].keys())
            memory = [memory_data[bench_name][size] for size in sizes]

            ax1.plot(sizes, memory, "o-", label=bench_name, linewidth=2, markersize=6)

        ax1.set_xlabel("Input Size", fontsize=12)
        ax1.set_ylabel("Memory Usage (MiB)", fontsize=12)
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log")
        ax1.legend(fontsize=10, loc="best")
        ax1.grid(True, which="both", ls="--", alpha=0.5)
        ax1.set_title("Total Memory Usage", fontsize=14)

    if bits_per_item_data:

        def get_last_bpi_value(bench_name):
            sizes = sorted(bits_per_item_data[bench_name].keys())
            if sizes:
                return bits_per_item_data[bench_name][sizes[-1]]
            return 0

        benchmark_names = sorted(
            bits_per_item_data.keys(), key=get_last_bpi_value, reverse=True
        )
        for bench_name in benchmark_names:
            sizes = sorted(bits_per_item_data[bench_name].keys())
            bpi = [bits_per_item_data[bench_name][size] for size in sizes]

            ax2.plot(sizes, bpi, "o-", label=bench_name, linewidth=2, markersize=6)

        ax2.set_xlabel("Input Size", fontsize=12)
        ax2.set_ylabel("Bits Per Item", fontsize=12)
        ax2.set_xscale("log", base=2)
        ax2.legend(fontsize=10, loc="best")
        ax2.grid(True, which="both", ls="--", alpha=0.5)
        ax2.set_title("Memory Efficiency (Bits Per Item)", fontsize=14)

    plt.tight_layout()

    output_file = output_dir / "benchmark_memory.png"
    plt.savefig(output_file, dpi=150)
    typer.secho(f"Memory plot saved to {output_file}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
