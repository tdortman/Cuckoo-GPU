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

app = typer.Typer(help="Plot FPR benchmark results")


def extract_filter_type(name: str) -> Optional[str]:
    """Extract filter type from benchmark name"""
    if "GPUCF_FPR" in name:
        return "Cuckoo Filter"
    elif "CPUCF_FPR" in name:
        return "CPU Cuckoo"
    elif "Bloom_FPR" in name:
        return "Bloom Filter"
    elif "TCF_FPR" in name:
        return "TCF"
    elif "PartitionedCF_FPR" in name:
        return "Partitioned Cuckoo"
    return None


@app.command()
def main(
    csv_file: Path = typer.Argument(
        Path("./benchmark_results/fpr_results.csv"),
        help="Path to the CSV file containing benchmark results",
    ),
    output_dir: Path = typer.Option(
        Path("./build"),
        help="Directory to save output plots",
    ),
):
    """
    Parse FPR benchmark CSV results and generate plots.
    """
    if not csv_file.exists():
        typer.secho(f"CSV file not found: {csv_file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.secho(f"Reading CSV from: {csv_file}", fg=typer.colors.CYAN)

    df = pd.read_csv(csv_file)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    # Dictionary structure: filter_type -> {memory_size: metric_value}
    fpr_data = defaultdict(dict)
    bits_per_item_data = defaultdict(dict)

    for _, row in df.iterrows():
        name = row["name"]
        filter_type = extract_filter_type(name)

        if filter_type is None:
            continue

        memory_bytes = row.get("memory_bytes")
        fpr_percentage = row.get("fpr_percentage")
        bits_per_item = row.get("bits_per_item")

        if pd.notna(memory_bytes):
            if pd.notna(fpr_percentage):
                fpr_data[filter_type][memory_bytes] = fpr_percentage
            if pd.notna(bits_per_item):
                bits_per_item_data[filter_type][memory_bytes] = bits_per_item

    if not fpr_data:
        typer.secho("No FPR data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Define colors and markers for each filter type
    filter_styles = {
        "Cuckoo Filter": {"color": "#2E86AB", "marker": "o"},
        "CPU Cuckoo": {"color": "#00B4D8", "marker": "o"},
        "Bloom Filter": {"color": "#A23B72", "marker": "s"},
        "TCF": {"color": "#C73E1D", "marker": "v"},
        "Partitioned Cuckoo": {"color": "#6A994E", "marker": "D"},
    }

    # Plot 1: FPR vs Memory Size
    fig, ax = plt.subplots(figsize=(12, 8))

    for filter_type in sorted(fpr_data.keys()):
        memory_sizes = sorted(fpr_data[filter_type].keys())
        fpr_values = [fpr_data[filter_type][mem] for mem in memory_sizes]

        style = filter_styles.get(filter_type, {"marker": "o"})
        ax.plot(
            memory_sizes,
            fpr_values,
            label=filter_type,
            linewidth=2.5,
            markersize=8,
            color=style.get("color"),
            marker=style.get("marker", "o"),
        )

    ax.set_xlabel("Memory Size [bytes]", fontsize=14, fontweight="bold")
    ax.set_ylabel("False Positive Rate [%]", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5), framealpha=1)
    ax.set_title(
        "False Positive Rate vs Memory Size",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    output_file = output_dir / "fpr_vs_memory.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight", transparent=False)
    typer.secho(
        f"FPR vs memory plot saved to {output_file}",
        fg=typer.colors.GREEN,
    )
    plt.close()

    # Plot 2: Bits per Item vs Memory Size
    fig, ax = plt.subplots(figsize=(12, 8))

    for filter_type in sorted(bits_per_item_data.keys()):
        memory_sizes = sorted(bits_per_item_data[filter_type].keys())
        bits_values = [bits_per_item_data[filter_type][mem] for mem in memory_sizes]

        style = filter_styles.get(filter_type, {"marker": "o"})
        ax.plot(
            memory_sizes,
            bits_values,
            label=filter_type,
            linewidth=2.5,
            markersize=8,
            color=style.get("color"),
            marker=style.get("marker", "o"),
        )

    ax.set_xlabel("Memory Size [bytes]", fontsize=14, fontweight="bold")
    ax.set_ylabel("Bits per Item", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5), framealpha=1)
    ax.set_title(
        "Space Efficiency vs Memory Size",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    output_file = output_dir / "bits_per_item_vs_memory.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight", transparent=False)
    typer.secho(
        f"Bits per item plot saved to {output_file}",
        fg=typer.colors.GREEN,
    )
    plt.close()

    typer.secho("\nAll plots generated successfully!", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
