#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
"""Plot bucket policy benchmark results as clustered bar charts."""

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot bucket policy benchmark results")


def load_policy_data(csv_path: Path) -> dict[str, dict[str, float]]:
    """Load and parse policy benchmark data from CSV.

    Returns:
        Nested dict {policy: {operation: throughput_mops}}
    """
    df = pu.load_csv(csv_path)
    df = df[df["name"].str.endswith("_median")]

    data: dict[str, dict[str, float]] = {}

    for _, row in df.iterrows():
        name = row["name"]
        # Parse: "XorFixture/Insert/..." -> policy="Xor", operation="Insert"
        match = re.match(r"(\w+)Fixture/(\w+)/", name)
        if not match:
            continue

        policy = match.group(1)
        operation = match.group(2)

        items_per_second = row.get("items_per_second")
        if pd.notna(items_per_second):
            throughput_mops = int(items_per_second / 1_000_000)
            if policy not in data:
                data[policy] = {}
            data[policy][operation] = throughput_mops

    return data


@app.command()
def main(
    csv_file: Path = typer.Argument(..., help="Path to policy benchmark CSV file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for plots"
    ),
):
    """Plot bucket policy benchmark results."""
    data = load_policy_data(csv_file)

    if not data:
        typer.secho("No valid data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    policies = ["Xor", "AddSub", "Offset"]
    policies = [p for p in policies if p in data]
    operations = ["Insert", "Query", "QueryNegative", "Delete"]

    fig, ax = pu.setup_figure(figsize=(10, 6))

    pu.clustered_bar_chart(
        ax,  # ty:ignore[invalid-argument-type]
        categories=operations,
        groups=policies,
        data=data,
        colors=pu.POLICY_COLORS,
    )

    pu.format_axis(
        ax,  # ty:ignore[invalid-argument-type]
        "Operation",
        "Throughput [M ops/s]",
        title="Bucket Policy Comparison",
        xscale=None,
    )
    pu.create_legend(ax, loc="upper right")  # ty:ignore[invalid-argument-type]

    plt.tight_layout()

    output_file = output_dir / "policy_benchmark.pdf"
    pu.save_figure(fig, output_file, f"Policy benchmark plot saved to {output_file}")


if __name__ == "__main__":
    app()
