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

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot delete-skew benchmark CSV results")


def _median_rows(csv_file: Path) -> pd.DataFrame:
    df = pu.load_csv(csv_file)
    required = {
        "name",
        "items_per_second",
        "delete_fraction",
        "skew",
        "delete_cas_attempts_per_key",
        "delete_cas_failure_rate",
        "successful_delete_fraction",
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
    df["skew"] = df["skew"].astype(float)
    df["throughput_beps"] = (
        df["items_per_second"].astype(float).map(pu.to_billion_elems_per_sec)
    )
    return df.sort_values(["capacity", "skew", "delete_fraction"])


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
        for skew, group in subset.groupby("skew"):
            label = f"skew={skew / 100:g}"
            ax.plot(
                group["delete_fraction"],
                group[y_column],
                marker="o",
                linewidth=pu.LINE_WIDTH,
                markersize=pu.MARKER_SIZE,
                label=label,
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
