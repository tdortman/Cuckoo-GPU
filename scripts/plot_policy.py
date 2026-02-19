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
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot bucket policy benchmark results")


def load_policy_data(csv_path: Path) -> tuple[dict[str, dict[str, float]], int | None]:
    """Load and parse policy benchmark data from CSV.

    Returns:
        Tuple of (nested dict {policy: {operation: throughput_mops}}, capacity)
    """
    df = pu.load_csv(csv_path)
    df = df[df["name"].str.endswith("_median")]

    data: dict[str, dict[str, float]] = {}
    capacity: int | None = None
    policy_name_map = {
        "xor": "Xor",
        "addsub": "AddSub",
        "offset": "Offset",
    }

    for _, row in df.iterrows():
        name = row["name"]
        parsed = pu.parse_fixture_benchmark_name(name)
        if parsed is None:
            continue

        policy_key, operation, parsed_capacity = parsed
        policy = policy_name_map.get(policy_key, policy_key.capitalize())
        if capacity is None:
            capacity = parsed_capacity

        items_per_second = row.get("items_per_second")
        if pd.notna(items_per_second):
            throughput_mops = int(items_per_second / 1_000_000)
            if policy not in data:
                data[policy] = {}
            data[policy][operation] = throughput_mops

    return data, capacity


@app.command()
def main(
    csv_file: Path = typer.Argument(..., help="Path to policy benchmark CSV file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for plots"
    ),
):
    """Plot bucket policy benchmark results."""
    data, capacity = load_policy_data(csv_file)

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
        xscale=None,
    )

    # Get handles and labels from axis, then create figure legend above
    handles, labels = ax.get_legend_handles_labels()  # ty:ignore[unresolved-attribute]
    fig.legend(
        handles,
        labels,
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    output_file = output_dir / "policy_benchmark.pdf"
    pu.save_figure(fig, output_file, f"Policy benchmark plot saved to {output_file}")


if __name__ == "__main__":
    app()
