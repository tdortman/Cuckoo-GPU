#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
#   "numpy",
# ]
# ///


from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import plot_utils as pu
import typer

app = typer.Typer(help="Plot cache benchmark results")


@app.command()
def main(
    csv_file: Path = typer.Argument(
        ...,
        help="Path to CSV file with cache benchmark results",
        exists=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Generate cache hit rate vs. capacity plots from benchmark CSV results.

    Creates line plots showing how L1 and L2 cache hit rates change as the
    data size increases, helping identify where cache efficiency degrades.

    Examples:
        plot_cache_benchmark.py cache_results.csv
        plot_cache_benchmark.py cache_results.csv -o custom/dir
    """
    df = pu.load_csv(csv_file)

    # Validate required columns
    required_cols = ["filter", "operation", "capacity", "l1_hit_rate", "l2_hit_rate"]
    if not all(col in df.columns for col in required_cols):
        typer.secho(
            f"CSV missing required columns. Expected: {required_cols}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    operation_markers = {
        "insert": "o",
        "query": "s",
        "delete": "^",
    }

    # Create 2x2 grid plots (one subplot per filter), per cache level
    for cache_level, metric_col in [("L1", "l1_hit_rate"), ("L2", "l2_hit_rate")]:
        filters = ["gcf", "bbf", "tcf", "gqf"]
        available_filters = [f for f in filters if f in df["filter"].unique()]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, filter_type in enumerate(available_filters):
            ax = axes[idx]
            filter_df = df[df["filter"] == filter_type]

            for operation in sorted(filter_df["operation"].unique()):
                subset = filter_df[filter_df["operation"] == operation].sort_values(
                    "capacity"
                )
                if subset.empty:
                    continue

                ax.plot(
                    subset["capacity"].values,
                    subset[metric_col].values,
                    label=operation.capitalize(),
                    linewidth=pu.LINE_WIDTH,
                    markersize=pu.MARKER_SIZE,
                    color=pu.FILTER_STYLES.get(filter_type, {}).get("color"),
                    marker=operation_markers.get(operation, "o"),
                    linestyle="-",
                )

            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 105)
            ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
            ax.set_title(
                pu.get_filter_display_name(filter_type),
                fontsize=pu.AXIS_LABEL_FONT_SIZE,
                fontweight="bold",
            )
            ax.legend(
                fontsize=pu.LEGEND_FONT_SIZE,
                loc="best",
                framealpha=pu.LEGEND_FRAME_ALPHA,
            )

        # Hide unused subplots if fewer than 4 filters
        for idx in range(len(available_filters), 4):
            axes[idx].set_visible(False)

        # Common axis labels
        fig.supxlabel(
            "Filter Capacity (elements)",
            fontsize=pu.AXIS_LABEL_FONT_SIZE,
            fontweight="bold",
        )
        fig.supylabel(
            f"{cache_level} Cache Hit Rate (%)",
            fontsize=pu.AXIS_LABEL_FONT_SIZE,
            fontweight="bold",
        )

        plt.tight_layout()

        output_file = output_dir / f"cache_grid_{cache_level.lower()}.pdf"
        plt.savefig(
            output_file,
            bbox_inches="tight",
            transparent=True,
            format="pdf",
            dpi=600,
        )
        typer.secho(f"Saved {output_file}", fg=typer.colors.GREEN)
        plt.close()


if __name__ == "__main__":
    app()
