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

import math
from pathlib import Path
from typing import Optional

import pandas as pd
import plot_utils as pu
import typer
from matplotlib.patches import Patch

app = typer.Typer(help="Plot bucket policy benchmark results")

_OPERATION_ORDER = ["Insert", "Query", "QueryNegative", "Delete"]
_OPERATION_LABELS = {
    "Insert": "Insert",
    "Query": "Query (+)",
    "QueryNegative": "Query (\u2212)",
    "Delete": "Delete",
}
_POLICY_ORDER = ["Xor", "AddSub", "Offset"]
_POLICY_STRIDE = 0.24
_SIZE_OFFSET = 0.055
_BAR_WIDTH = 0.10
_SMALL_ALPHA = 0.35
_X_MARGIN = 0.06


def format_capacity_label(kind: str, capacity: Optional[int]) -> str:
    """Create a short legend label for dataset size."""
    if capacity is None or capacity <= 0:
        return kind

    if capacity & (capacity - 1) == 0:
        power = int(math.log2(capacity))
        return f"{kind} ($n=2^{{{power}}}$)"
    return f"{kind} (n={capacity})"


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
            throughput_mops = items_per_second / 1_000_000
            if policy not in data:
                data[policy] = {}
            data[policy][operation] = throughput_mops

    return data, capacity


@app.command()
def main(
    csv_file_a: Path = typer.Argument(..., help="Path to first policy benchmark CSV"),
    csv_file_b: Path = typer.Argument(..., help="Path to second policy benchmark CSV"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for plots"
    ),
):
    """Plot bucket policy benchmark results from two datasets.

    Renders side-by-side bars per policy/operation:
    - Large filter: solid color
    - Small filter: lower opacity
    """
    data_a, capacity_a = load_policy_data(csv_file_a)
    data_b, capacity_b = load_policy_data(csv_file_b)

    if not data_a or not data_b:
        typer.secho(
            "No valid data found in one or both CSV files",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Determine small/large dataset by parsed benchmark capacity (if available).
    if capacity_a is not None and capacity_b is not None and capacity_a != capacity_b:
        if capacity_a > capacity_b:
            large_data = data_a,
            small_data = data_b,
        else:
            large_data = data_b, 
            small_data = data_a, 
    else:
        # Fallback: preserve input order if capacities are equal/missing.
        small_data = data_a, capacity_a
        large_data = data_b, capacity_b
        typer.secho(
            "Warning: Could not infer unique small/large capacities from inputs; "
            "using first CSV as small and second CSV as large.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    all_policies = set(small_data.keys()) | set(large_data.keys())
    policies = [p for p in _POLICY_ORDER if p in all_policies]
    extra_policies = sorted(all_policies - set(policies))
    policies.extend(extra_policies)

    if not policies:
        typer.secho(
            "No policy data available after parsing inputs",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    operations = _OPERATION_ORDER
    operation_labels = [_OPERATION_LABELS.get(op, op) for op in operations]

    fig, ax = pu.setup_figure(figsize=(10, 6))

    n_policies = len(policies)
    for op_idx, operation in enumerate(operations):
        for policy_idx, policy in enumerate(policies):
            cluster_center = (
                op_idx + (policy_idx - (n_policies - 1) / 2) * _POLICY_STRIDE
            )
            large_tp = large_data.get(policy, {}).get(operation, 0.0)
            small_tp = small_data.get(policy, {}).get(operation, 0.0)
            color = pu.POLICY_COLORS.get(policy, "#333333")

            if large_tp > 0:
                ax.bar(  # type: ignore
                    cluster_center - _SIZE_OFFSET,
                    large_tp,
                    _BAR_WIDTH,
                    color=color,
                    edgecolor="black",
                    linewidth=pu.BAR_EDGE_WIDTH,
                    alpha=1.0,
                    zorder=3,
                )
            if small_tp > 0:
                ax.bar(  # type: ignore
                    cluster_center + _SIZE_OFFSET,
                    small_tp,
                    _BAR_WIDTH,
                    color=color,
                    edgecolor="#666666",
                    linewidth=pu.BAR_EDGE_WIDTH,
                    alpha=_SMALL_ALPHA,
                    zorder=2,
                )

    ax.set_xticks(list(range(len(operations))))  # type: ignore
    ax.set_xticklabels(operation_labels, fontsize=pu.DEFAULT_FONT_SIZE)  # type: ignore
    ax.set_xlabel("Operation", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold")  # type: ignore
    ax.set_ylabel(  # type: ignore
        "Throughput [M ops/s]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA, zorder=0)  # type: ignore

    half_span = (
        ((n_policies - 1) / 2) * _POLICY_STRIDE + _SIZE_OFFSET + (_BAR_WIDTH / 2)
    )
    ax.set_xlim(-half_span - _X_MARGIN, len(operations) - 1 + half_span + _X_MARGIN)  # type: ignore

    policy_handles = [
        Patch(
            facecolor=pu.POLICY_COLORS.get(policy, "#333333"),
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
            label=pu.get_policy_display_name(policy),
        )
        for policy in policies
    ]
    size_handles = [
        Patch(
            facecolor="gray",
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
            alpha=1.0,
            label="DRAM-resident",
        ),
        Patch(
            facecolor="gray",
            edgecolor="#666666",
            linewidth=pu.BAR_EDGE_WIDTH,
            alpha=_SMALL_ALPHA,
            label="L2-resident",
        ),
    ]

    legend_y_top = 0.99
    legend_handles = policy_handles + size_handles
    fig.legend(
        handles=legend_handles,
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y_top),
        ncol=len(legend_handles),
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.86)

    output_file = output_dir / "policy_benchmark.pdf"
    pu.save_figure(fig, output_file, f"Policy benchmark plot saved to {output_file}")


if __name__ == "__main__":
    app()
