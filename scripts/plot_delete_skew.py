#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
"""Plot delete-skew benchmark throughput and CAS counters."""

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

PAPER_AXIS_LABEL_FONT_SIZE = 7
PAPER_TICK_LABEL_FONT_SIZE = 5.5
PAPER_LEGEND_FONT_SIZE = 6
PAPER_LINE_WIDTH = 0.9
PAPER_MARKER_SIZE = 2.0

app = typer.Typer(help="Plot delete-skew benchmark CSV results")


def _median_rows(csv_file: Path) -> pd.DataFrame:
    df = pu.load_csv(csv_file)
    required = {
        "name",
        "items_per_second",
        "delete_fraction",
        "stddev_fraction",
        "delete_cas_attempts_per_key",
        "delete_cas_failure_rate",
        "successful_delete_fraction",
        "delete_cas_failures",
        "successful_deletes",
    }
    missing = required - set(df.columns)
    if missing:
        typer.secho(
            f"Missing columns in {csv_file}: {sorted(missing)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    df = df[df["name"].str.endswith("_median")].copy()
    if df.empty:
        typer.secho("No *_median benchmark rows found", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    parts = df["name"].str.extract(r"DeleteSkew/(\d+)/(\d+)/(\d+)")
    df["capacity"] = parts[0].astype(int)
    df["delete_fraction"] = df["delete_fraction"].astype(float)
    df["stddev_fraction"] = df["stddev_fraction"].astype(float)
    df["throughput_beps"] = (
        df["items_per_second"].astype(float).map(pu.to_billion_elems_per_sec)
    )
    df["successful_throughput_beps"] = (
        df["items_per_second"].astype(float)
        * (df["successful_delete_fraction"].astype(float) / 100.0)
    ).map(pu.to_billion_elems_per_sec)
    df["cas_failures_per_successful_delete"] = (
        df["delete_cas_failures"].astype(float)
        / df["successful_deletes"].astype(float).where(
            df["successful_deletes"].astype(float) > 0
        )
    )
    return df.sort_values(["capacity", "stddev_fraction", "delete_fraction"])


def _plot_lines(
    df: pd.DataFrame,
    y_column: str,
    y_label: str,
    output_file: Path,
    title: str,
    yscale: Optional[str] = None,
) -> None:
    capacities = sorted(df["capacity"].unique())
    fig, axes = plt.subplots(
        1,
        len(capacities),
        figsize=(5 * len(capacities), 4.5),
        sharey=True,
        squeeze=False,
    )

    for ax, capacity in zip(axes[0], capacities):
        subset = df[df["capacity"] == capacity]
        for stddev, group in subset.groupby("stddev_fraction"):
            ax.plot(
                group["delete_fraction"],
                group[y_column],
                marker="o",
                linewidth=pu.LINE_WIDTH,
                markersize=pu.MARKER_SIZE,
                label=_stddev_label(stddev),
            )

        ax.set_title(pu.format_power_of_two(int(capacity)), fontsize=pu.TITLE_FONT_SIZE)
        ax.set_xlabel("Delete fraction [%]", fontsize=pu.AXIS_LABEL_FONT_SIZE)
        ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
        ax.tick_params(axis="both", labelsize=pu.TICK_LABEL_FONT_SIZE)
        if yscale:
            ax.set_yscale(yscale)

    axes[0][0].set_ylabel(y_label, fontsize=pu.AXIS_LABEL_FONT_SIZE)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, len(labels)),
        framealpha=pu.LEGEND_FRAME_ALPHA_SOLID,
        fontsize=pu.LEGEND_FONT_SIZE,
    )
    fig.suptitle(title, fontsize=pu.TITLE_FONT_SIZE, fontweight="bold", y=1.12)
    fig.tight_layout()
    pu.save_figure(fig, output_file, f"Saved {output_file}")


def _capacity_exponents(df: pd.DataFrame) -> list[int]:
    return [int(math.log2(c)) for c in sorted(df["capacity"].unique())]


def _format_capacity_axis(ax: plt.Axes, exponents: list[int]) -> None:
    even_exponents = [exp for exp in exponents if exp % 2 == 0]
    ax.set_xticks(even_exponents)
    ax.set_xticklabels([rf"$2^{{{exp}}}$" for exp in even_exponents])
    ax.set_xlim(min(exponents) - 0.3, max(exponents) + 0.3)
    ax.set_xlabel(
        "Capacity", fontsize=PAPER_AXIS_LABEL_FONT_SIZE, fontweight="bold", labelpad=1
    )
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
    ax.tick_params(axis="both", labelsize=PAPER_TICK_LABEL_FONT_SIZE, pad=1)



def _stddev_label(stddev: float) -> str:
    if math.isinf(stddev):
        return "uniform"
    return rf"$\sigma$={stddev:g}%"

def _plot_metric_by_group(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_column: str,
    y_column: str,
    y_label: str,
    title: Optional[str],
    label_fn,
) -> None:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    for idx, (value, group) in enumerate(df.groupby(group_column)):
        group = group.sort_values("capacity").copy()
        x = group["capacity"].map(lambda c: int(math.log2(c)))
        ax.plot(
            x,
            group[y_column],
            marker=markers[idx % len(markers)],
            linewidth=PAPER_LINE_WIDTH,
            markersize=PAPER_MARKER_SIZE,
            label=label_fn(value),
        )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=3)
    ax.set_ylabel(y_label, fontsize=10, labelpad=2)
    _format_capacity_axis(ax, _capacity_exponents(df))


def _save_two_panel(fig: plt.Figure, axes, output_file: Path, legend_columns: int) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=legend_columns,
        fontsize=7,
        framealpha=pu.LEGEND_FRAME_ALPHA_SOLID,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.0)
    pu.save_figure(fig, output_file, f"Saved {output_file}")


def _plot_hotspot_trends(df: pd.DataFrame, output_file: Path) -> None:
    delete_fraction = df["delete_fraction"].max()
    stress = df[(df["delete_fraction"] == delete_fraction) & (df["stddev_fraction"] >= 0)]
    if stress.empty:
        typer.secho("No hotspot-stress rows found", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True)
    _plot_metric_by_group(
        axes[0],
        stress,
        "stddev_fraction",
        "successful_delete_fraction",
        "successful deletes [%]",
        f"(a) Hit rate, delete={delete_fraction:g}%",
        _stddev_label,
    )
    _plot_metric_by_group(
        axes[1],
        stress,
        "stddev_fraction",
        "delete_cas_failure_rate",
        "CAS failure rate",
        f"(b) CAS failures, delete={delete_fraction:g}%",
        _stddev_label,
    )
    _save_two_panel(fig, axes, output_file, legend_columns=min(4, stress["stddev_fraction"].nunique()))


@app.command()
def main(
    csv_file: Path = typer.Argument(..., help="delete-skew benchmark CSV"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Directory for generated plots"
    ),
) -> None:
    """Create throughput, CAS retry, and successful-delete plots."""
    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))
    df = _median_rows(csv_file)

    _plot_hotspot_trends(df, output_dir / "delete_skew_stdev_stress.pdf")

    _plot_lines(
        df,
        "throughput_beps",
        pu.THROUGHPUT_LABEL,
        output_dir / "delete_skew_throughput.pdf",
        "Delete-heavy throughput",
    )
    _plot_lines(
        df,
        "delete_cas_attempts_per_key",
        "Delete CAS attempts / request",
        output_dir / "delete_skew_cas_attempts.pdf",
        "Delete CAS attempts",
    )
    _plot_lines(
        df,
        "delete_cas_failure_rate",
        "Delete CAS failure rate",
        output_dir / "delete_skew_cas_failure_rate.pdf",
        "Delete CAS failure rate",
    )
    _plot_lines(
        df,
        "successful_delete_fraction",
        "Successful deletes [%]",
        output_dir / "delete_skew_success_rate.pdf",
        "Delete hit rate",
    )


if __name__ == "__main__":
    app()
