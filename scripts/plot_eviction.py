#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils as pu
import typer
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

app = typer.Typer(help="Plot eviction benchmark results")


class HandlerTwoSegmentLine(HandlerBase):
    """Draw a legend sample as two equal line segments with one equal-size gap."""

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        y = ydescent + height / 2.0
        chunk = width / 3.0

        line1 = Line2D(
            [xdescent, xdescent + chunk],
            [y, y],
            color=orig_handle.get_color(),
            linewidth=orig_handle.get_linewidth(),
            solid_capstyle="butt",
            transform=trans,
        )
        line2 = Line2D(
            [xdescent + 2.0 * chunk, xdescent + 3.0 * chunk],
            [y, y],
            color=orig_handle.get_color(),
            linewidth=orig_handle.get_linewidth(),
            solid_capstyle="butt",
            transform=trans,
        )
        return [line1, line2]


class HandlerSolidLine(HandlerBase):
    """Draw a legend sample as one full-length line with the same geometry as DFS proxy."""

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        y = ydescent + height / 2.0
        line = Line2D(
            [xdescent, xdescent + width],
            [y, y],
            color=orig_handle.get_color(),
            linewidth=orig_handle.get_linewidth(),
            solid_capstyle="butt",
            transform=trans,
        )
        return [line]


def extract_eviction_policy(name: str) -> Optional[str]:
    """Extract eviction policy from benchmark name like BFSFixture/Evictions/..."""
    if name.startswith("BFSFixture"):
        return "BFS"
    elif name.startswith("DFSFixture"):
        return "DFS"
    return None


def extract_capacity(name: str) -> Optional[int]:
    """Extract capacity from benchmark name like BFSFixture/Evictions/16777216/..."""
    match = re.search(r"/Evictions/(\d+)/", name)
    if match:
        return int(match.group(1))
    return None


def extract_load_factor(row: pd.Series) -> Optional[float]:
    """Extract load factor from benchmark counter."""
    lf = row.get("load_factor")
    if pd.notna(lf):
        return float(lf) / 100.0
    return None


def load_csv_data(csv_path: Path) -> tuple[dict, dict, dict, Optional[int]]:
    """Load and parse benchmark data from a CSV file.

    Returns a tuple of (eviction_data, total_evictions_data, throughput_data, capacity)
    """
    df = pu.load_csv(csv_path)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    eviction_data = defaultdict(dict)
    total_evictions_data = defaultdict(dict)
    throughput_data = defaultdict(dict)
    capacity = None

    for _, row in df.iterrows():
        name = row["name"]

        policy = extract_eviction_policy(name)
        load_factor = extract_load_factor(row)
        row_capacity = extract_capacity(name)

        if policy is None or load_factor is None:
            continue

        if row_capacity is not None and capacity is None:
            capacity = row_capacity

        evictions_per_insert = row.get("evictions_per_insert")
        evictions = row.get("evictions")
        items_per_second = row.get("items_per_second")

        if pd.notna(evictions_per_insert):
            eviction_data[policy][load_factor] = evictions_per_insert
        if pd.notna(evictions):
            total_evictions_data[policy][load_factor] = evictions
        if pd.notna(items_per_second):
            throughput_data[policy][load_factor] = items_per_second

    return eviction_data, total_evictions_data, throughput_data, capacity


def load_histogram_csv(histogram_csv: Path) -> pd.DataFrame:
    """Load histogram CSV."""
    df = pd.read_csv(histogram_csv)

    required = {
        "policy",
        "capacity",
        "load_factor",
        "bin_start",
        "bin_end",
        "count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required histogram columns in {histogram_csv}: {sorted(missing)}"
        )

    return df.copy()


def plot_metric_figure(
    datasets: list[dict],
    metric_key: str,
    y_label: str,
    output_file: Path,
    policy_styles: dict,
    value_scale: float = 1.0,
    yscale: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    all_handles = []
    all_labels = []

    for idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        metric_data = dataset[metric_key]

        if not metric_data:
            ax.set_title("No data", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold")
            ax.axis("off")
            continue

        for policy in sorted(metric_data.keys()):
            load_factors = sorted(metric_data[policy].keys())
            values = [metric_data[policy][lf] / value_scale for lf in load_factors]

            style = policy_styles.get(policy, {})
            (line,) = ax.plot(
                load_factors,
                values,
                label=policy,
                linewidth=pu.LINE_WIDTH,
                markersize=pu.MARKER_SIZE,
                **style,
            )

            if idx == 0 and policy not in all_labels:
                all_handles.append(line)
                all_labels.append(policy)

        if yscale is not None:
            ax.set_yscale(yscale)

        if idx == len(axes) - 1:
            ax.set_xlabel(
                "Load Factor", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
            )

        ax.set_ylabel(y_label, fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold")
        ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)

    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            fontsize=pu.LEGEND_FONT_SIZE,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(all_labels),
            framealpha=pu.LEGEND_FRAME_ALPHA,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    pu.save_figure(fig, output_file, f"Saved {output_file}")


def plot_metric_separate(
    datasets: list[dict],
    metric_key: str,
    y_label: str,
    output_stem: str,
    output_dir: Path,
    policy_styles: dict,
    value_scale: float = 1.0,
    yscale: Optional[str] = None,
) -> None:
    for dataset in datasets:
        metric_data = dataset[metric_key]
        if not metric_data:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))

        for policy in sorted(metric_data.keys()):
            load_factors = sorted(metric_data[policy].keys())
            values = [metric_data[policy][lf] / value_scale for lf in load_factors]

            style = policy_styles.get(policy, {})
            ax.plot(
                load_factors,
                values,
                label=policy,
                linewidth=pu.LINE_WIDTH,
                markersize=pu.MARKER_SIZE,
                **style,
            )

        if yscale is not None:
            ax.set_yscale(yscale)

        ax.set_xlabel(
            "Load Factor", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
        )
        ax.set_ylabel(y_label, fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold")
        ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)

        ax.legend(
            fontsize=pu.LEGEND_FONT_SIZE,
            loc="upper right",
            framealpha=pu.LEGEND_FRAME_ALPHA,
        )

        output_file = output_dir / f"{output_stem}_{dataset['position']}.pdf"
        pu.save_figure(fig, output_file, f"Saved {output_file}")


def weighted_percentile(bin_values: np.ndarray, counts: np.ndarray, q: float) -> float:
    total = counts.sum()
    if total <= 0:
        return float("nan")

    cumulative = np.cumsum(counts)
    threshold = q * total
    idx = int(np.searchsorted(cumulative, threshold, side="left"))
    idx = min(idx, len(bin_values) - 1)
    return float(bin_values[idx])


def plot_histogram_percentiles(
    histogram_csv: Path,
    output_dir: Path,
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> None:
    """Plot percentile curves (p90/p95/p99) over load factor from histogram bins."""
    try:
        df = load_histogram_csv(histogram_csv)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    if df.empty:
        typer.secho(
            f"No histogram data found in {histogram_csv} for percentile plot",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return

    rows = []
    grouped = df.groupby(["policy", "load_factor"], as_index=False)
    for (policy, load_factor), group in grouped:
        group = group.sort_values("bin_start")
        bin_values = group["bin_start"].to_numpy(dtype=float)
        counts = group["count"].to_numpy(dtype=float)

        for q in quantiles:
            rows.append(
                {
                    "policy": policy,
                    "load_factor": float(load_factor),
                    "quantile": q,
                    "value": weighted_percentile(bin_values, counts, q),
                }
            )

    percentile_df = pd.DataFrame(rows)
    if percentile_df.empty:
        typer.secho(
            f"No percentile data could be computed from {histogram_csv}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    quantile_styles = {
        0.90: {"color": "#4C78A8", "marker": "o", "linestyle": "-"},
        0.95: {"color": "#F58518", "marker": "s", "linestyle": "--"},
        0.99: {"color": "#54A24B", "marker": "D", "linestyle": "-."},
    }
    policy_styles = {
        "BFS": {"linestyle": "-", "linewidth": pu.LINE_WIDTH + 0.4},
        "DFS": {"linestyle": (0, (7, 3)), "linewidth": pu.LINE_WIDTH + 0.4},
    }

    plotted_any = False
    for q in quantiles:
        for policy in ["BFS", "DFS"]:
            line_df = percentile_df[
                (percentile_df["policy"] == policy) & (percentile_df["quantile"] == q)
            ].sort_values("load_factor")
            if line_df.empty:
                continue

            q_style = quantile_styles.get(q, {"color": "#333333", "marker": "o"})
            policy_style = policy_styles.get(
                policy, {"linestyle": "-", "linewidth": pu.LINE_WIDTH}
            )
            y_values = line_df["value"].to_numpy(dtype=float)
            ax.plot(
                line_df["load_factor"].to_list(),
                y_values.tolist(),
                color=q_style["color"],
                marker=q_style["marker"],
                linestyle=policy_style["linestyle"],
                linewidth=policy_style["linewidth"],
                markersize=pu.MARKER_SIZE,
            )
            plotted_any = True

    if not plotted_any:
        typer.secho(
            f"No percentile lines could be plotted from {histogram_csv}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        plt.close(fig)
        return

    ax.set_xlabel("Load Factor", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold")
    ax.set_ylabel("Evictions", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold")
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.0, base=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
    bfs_handle = Line2D(
        [0, 1],
        [0, 0],
        color="#222222",
        linestyle="-",
        linewidth=policy_styles["BFS"]["linewidth"],
        marker=None,
        label="BFS",
    )
    dfs_handle = Line2D(
        [],
        [],
        color="#222222",
        linewidth=policy_styles["DFS"]["linewidth"],
        marker=None,
        label="DFS",
    )
    policy_handles = [bfs_handle, dfs_handle]
    quantile_handles = [
        Line2D(
            [0],
            [0],
            color=quantile_styles[q]["color"],
            marker=quantile_styles[q]["marker"],
            linestyle="-",
            linewidth=pu.LINE_WIDTH,
            markersize=pu.MARKER_SIZE,
            label=f"p{int(q * 100)}",
        )
        for q in quantiles
    ]

    policy_legend = ax.legend(
        handles=policy_handles,
        title="Policy",
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0, 0.2, 0.0),
        mode="expand",
        handlelength=4.0,
        numpoints=2,
        framealpha=pu.LEGEND_FRAME_ALPHA_SOLID,
        facecolor="white",
        edgecolor="#cccccc",
        handler_map={
            bfs_handle: HandlerSolidLine(),
            dfs_handle: HandlerTwoSegmentLine(),
        },
    )
    policy_legend.set_zorder(10)
    ax.add_artist(policy_legend)
    percentile_legend = ax.legend(
        handles=quantile_handles,
        title="Percentile",
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.82, 0.2, 0.0),
        mode="expand",
        framealpha=pu.LEGEND_FRAME_ALPHA_SOLID,
        facecolor="white",
        edgecolor="#cccccc",
    )
    percentile_legend.set_zorder(10)

    output_file = output_dir / f"eviction_{histogram_csv.stem}_percentiles.pdf"
    pu.save_figure(fig, output_file, f"Saved {output_file}")

    compare_rows = []
    for q in quantiles:
        bfs_q = percentile_df[
            (percentile_df["policy"] == "BFS") & (percentile_df["quantile"] == q)
        ][["load_factor", "value"]].rename(columns={"value": "bfs"})
        dfs_q = percentile_df[
            (percentile_df["policy"] == "DFS") & (percentile_df["quantile"] == q)
        ][["load_factor", "value"]].rename(columns={"value": "dfs"})

        merged = pd.merge(bfs_q, dfs_q, on="load_factor", how="inner")
        if merged.empty:
            continue

        merged["quantile"] = q
        merged["delta"] = merged["dfs"] - merged["bfs"]
        merged["ratio"] = np.nan
        bfs_positive = merged["bfs"] > 0
        both_zero = (merged["bfs"] == 0) & (merged["dfs"] == 0)
        merged.loc[bfs_positive, "ratio"] = (
            merged.loc[bfs_positive, "dfs"] / merged.loc[bfs_positive, "bfs"]
        )
        merged.loc[both_zero, "ratio"] = 1.0
        compare_rows.append(merged)

    if not compare_rows:
        return

    compare_df = pd.concat(compare_rows, ignore_index=True)
    fig_cmp, (ax_delta, ax_ratio) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for q in quantiles:
        q_df = compare_df[compare_df["quantile"] == q].sort_values("load_factor")
        if q_df.empty:
            continue

        style = quantile_styles.get(
            q, {"color": "#333333", "marker": "o", "linestyle": "-"}
        )
        label = f"p{int(q * 100)}"
        ax_delta.plot(
            q_df["load_factor"].to_list(),
            q_df["delta"].to_list(),
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=pu.LINE_WIDTH,
            markersize=pu.MARKER_SIZE,
        )
        ax_ratio.plot(
            q_df["load_factor"].to_list(),
            q_df["ratio"].to_list(),
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=pu.LINE_WIDTH,
            markersize=pu.MARKER_SIZE,
        )

    ax_delta.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0, zorder=0)
    ax_ratio.axhline(1.0, color="#666666", linestyle=":", linewidth=1.0, zorder=0)

    ax_delta.set_ylabel(
        "DFS - BFS", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax_ratio.set_ylabel(
        "DFS / BFS", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax_ratio.set_xlabel(
        "Load Factor", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )

    for cmp_ax in (ax_delta, ax_ratio):
        cmp_ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
        cmp_ax.legend(
            fontsize=pu.LEGEND_FONT_SIZE,
            loc="upper left",
            framealpha=pu.LEGEND_FRAME_ALPHA,
        )

    fig_cmp.subplots_adjust(hspace=0.2)
    compare_output = (
        output_dir / f"eviction_{histogram_csv.stem}_percentiles_compare.pdf"
    )
    pu.save_figure(fig_cmp, compare_output, f"Saved {compare_output}")


@app.command()
def main(
    csv_file_top: Path = typer.Argument(
        ...,
        help="Path to CSV file for top plot",
    ),
    csv_file_middle: Path = typer.Argument(
        ...,
        help="Path to CSV file for middle plot",
    ),
    csv_file_bottom: Path = typer.Argument(
        ...,
        help="Path to CSV file for bottom plot",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
    histogram_csv_top: Optional[Path] = typer.Option(
        None,
        "--histogram-csv-top",
        help="Optional histogram CSV for top benchmark",
    ),
    histogram_csv_middle: Optional[Path] = typer.Option(
        None,
        "--histogram-csv-middle",
        help="Optional histogram CSV for middle benchmark",
    ),
    histogram_csv_bottom: Optional[Path] = typer.Option(
        None,
        "--histogram-csv-bottom",
        help="Optional histogram CSV for bottom benchmark",
    ),
    histogram_percentiles: bool = typer.Option(
        True,
        "--histogram-percentiles/--no-histogram-percentiles",
        help="Generate p90/p95/p99 curves from histogram data",
    ),
):
    """
    Generate eviction plots from three benchmark CSV files.

    Output:
    - Throughput vs load factor
    - Average evictions per insertion attempt vs load factor
    - Total evictions vs load factor
    - Optional histogram percentile plots
    """
    csv_files = [
        (csv_file_top, "top"),
        (csv_file_middle, "middle"),
        (csv_file_bottom, "bottom"),
    ]

    datasets = []
    for csv_file, position in csv_files:
        eviction_data, total_evictions_data, throughput_data, capacity = load_csv_data(
            csv_file
        )
        if not throughput_data and not eviction_data and not total_evictions_data:
            typer.secho(
                f"No eviction benchmark data found in {csv_file} ({position})",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

        datasets.append(
            {
                "position": position,
                "csv_file": csv_file,
                "capacity": capacity,
                "throughput_data": throughput_data,
                "eviction_data": eviction_data,
                "total_evictions_data": total_evictions_data,
            }
        )

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    policy_styles = {
        "BFS": {**pu.FILTER_STYLES["gcf"], "linestyle": "-"},
        "DFS": {**pu.FILTER_STYLES["bbf"], "linestyle": "--"},
    }

    plot_metric_separate(
        datasets,
        metric_key="throughput_data",
        y_label="Throughput [M ops/s]",
        output_stem="eviction_throughput",
        output_dir=output_dir,
        policy_styles=policy_styles,
        value_scale=1e6,
    )

    plot_metric_separate(
        datasets,
        metric_key="eviction_data",
        y_label="Evictions per insert",
        output_stem="eviction_per_insert",
        output_dir=output_dir,
        policy_styles=policy_styles,
        yscale="log",
    )

    plot_metric_separate(
        datasets,
        metric_key="total_evictions_data",
        y_label="Total evictions",
        output_stem="eviction_total_evictions",
        output_dir=output_dir,
        policy_styles=policy_styles,
        yscale="log",
    )

    histogram_files = [
        histogram_csv_top,
        histogram_csv_middle,
        histogram_csv_bottom,
    ]
    for histogram_csv in histogram_files:
        if histogram_csv is None:
            continue
        if histogram_percentiles:
            plot_histogram_percentiles(histogram_csv, output_dir)


if __name__ == "__main__":
    app()
