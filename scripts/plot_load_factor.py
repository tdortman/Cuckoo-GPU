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
import pandas as pd
import plot_utils as pu
import typer
from matplotlib.patches import Patch

app = typer.Typer(help="Plot load factor benchmark results")


def extract_load_factor(name: str) -> Optional[float]:
    """Extract load factor from benchmark name like CF_5/Insert, BBF_95/Query, or CF_99_5/Insert"""
    # Match patterns like _95/ or _99_5/ (where underscore represents decimal point)
    match = re.search(r"_([\d_]+)/", name)
    if match:
        # Replace underscore with decimal point (e.g., "99_5" -> "99.5")
        value_str = match.group(1).replace("_", ".")
        return float(value_str) / 100.0
    return None


def extract_operation_type(name: str) -> Optional[str]:
    """Extract operation type from benchmark name"""
    if "/Insert/" in name:
        return "Insert"
    elif "/QueryNegative/" in name:
        return "Query"
    elif "/Query/" in name:
        return "Query"
    elif "/Delete/" in name:
        return "Delete"
    return None


def extract_lookup_type(name: str) -> Optional[str]:
    """Extract lookup type (Positive or Negative) for Query operations"""
    if "/QueryNegative/" in name:
        return "Negative"
    elif "/Query/" in name:
        return "Positive"
    return None


def load_csv_data(csv_path: Path) -> dict:
    """Load and parse benchmark data from a CSV file.

    Returns:
        Dictionary structure: operation -> filter_type -> {load_factor: throughput}
    """
    df = pu.load_csv(csv_path)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    # Dictionary structure: operation -> filter_type -> {load_factor: throughput}
    benchmark_data = defaultdict(lambda: defaultdict(dict))

    for _, row in df.iterrows():
        name = row["name"]

        # Extract filter type using standardized approach
        filter_key = pu.normalize_benchmark_name(name)
        filter_type = pu.get_filter_display_name(filter_key)

        if not filter_type:
            continue

        load_factor = extract_load_factor(name)
        operation_type = extract_operation_type(name)
        lookup_type = extract_lookup_type(name)

        if filter_type is None or load_factor is None or operation_type is None:
            continue

        # For Query operations, append lookup type to filter name
        if operation_type == "Query" and lookup_type:
            filter_key = f"{filter_type} ({lookup_type})"
        else:
            filter_key = filter_type

        items_per_second = row.get("items_per_second")
        if pd.notna(items_per_second):
            throughput_mops = items_per_second / 1_000_000
            benchmark_data[operation_type][filter_key][load_factor] = throughput_mops

    return benchmark_data


def get_filter_styles() -> dict:
    """Define colors and markers for each filter type, with positive/negative variants."""
    # Use the standardized filter styles from plot_utils as base
    base_styles = {
        "GPU Cuckoo": pu.FILTER_STYLES.get("gcf", {"color": "#2E86AB", "marker": "o"}),
        "CPU Cuckoo": pu.FILTER_STYLES.get("ccf", {"color": "#00B4D8", "marker": "o"}),
        "Blocked Bloom": pu.FILTER_STYLES.get(
            "bbf", {"color": "#A23B72", "marker": "s"}
        ),
        "Two-Choice": pu.FILTER_STYLES.get("tcf", {"color": "#C73E1D", "marker": "v"}),
        "GPU Quotient": pu.FILTER_STYLES.get(
            "gqf", {"color": "#F18F01", "marker": "^"}
        ),
        "Partitioned Cuckoo": pu.FILTER_STYLES.get(
            "pcf", {"color": "#6A994E", "marker": "D"}
        ),
        "BCHT": pu.FILTER_STYLES.get("bcht", {"color": "#264653", "marker": "X"}),
    }

    # Generate styles for both positive and negative variants
    filter_styles = {}
    for filter_name, base_style in base_styles.items():
        # Base style (for non-query operations)
        filter_styles[filter_name] = {
            "color": base_style["color"],
            "marker": base_style["marker"],
            "linestyle": "-",
        }
        # Positive lookups: solid line
        filter_styles[f"{filter_name} (Positive)"] = {
            "color": base_style["color"],
            "marker": base_style["marker"],
            "linestyle": "-",
        }
        # Negative lookups: dashed line
        filter_styles[f"{filter_name} (Negative)"] = {
            "color": base_style["color"],
            "marker": base_style["marker"],
            "linestyle": "--",
        }

    return filter_styles


def plot_operation_on_axis(
    ax: plt.Axes,
    operation_type: str,
    benchmark_data: dict,
    filter_styles: dict,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
) -> tuple[list, list]:
    """Plot a single operation's data on the given axis.

    Returns:
        A tuple of (handles, labels) for use in creating a combined legend.
        Labels are simplified to base filter names (without Positive/Negative suffixes).
    """
    handles = []
    labels = []
    seen_base_filters = set()

    for filter_type in sorted(benchmark_data[operation_type].keys()):
        load_factors = sorted(benchmark_data[operation_type][filter_type].keys())
        throughputs = [
            benchmark_data[operation_type][filter_type][lf] for lf in load_factors
        ]

        style = filter_styles.get(filter_type, {"marker": "o", "linestyle": "-"})
        (line,) = ax.plot(
            load_factors,
            throughputs,
            label=filter_type,
            linewidth=pu.LINE_WIDTH,
            markersize=pu.MARKER_SIZE,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
        )

        # Extract base filter name (strip Positive/Negative suffix)
        base_filter = filter_type.replace(" (Positive)", "").replace(" (Negative)", "")

        # Only add to legend if we haven't seen this base filter yet
        if base_filter not in seen_base_filters:
            handles.append(line)
            labels.append(base_filter)
            seen_base_filters.add(base_filter)

    if show_xlabel:
        ax.set_xlabel(
            "Load Factor", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
        )
    if show_ylabel:
        ax.set_ylabel(
            "Throughput [M ops/s]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
        )
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
    ax.set_yscale("log")

    return handles, labels


@app.command("sweep")
def sweep(
    csv_top_left: Path = typer.Argument(
        ...,
        help="Path to CSV file for top-left plot",
    ),
    csv_top_right: Path = typer.Argument(
        ...,
        help="Path to CSV file for top-right plot",
    ),
    csv_bottom_left: Path = typer.Argument(
        ...,
        help="Path to CSV file for bottom-left plot",
    ),
    csv_bottom_right: Path = typer.Argument(
        ...,
        help="Path to CSV file for bottom-right plot",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Generate throughput vs load factor sweep plots from four benchmark CSV files.

    Creates a 2x2 grid of line plots for each operation (insert, query, delete) with
    data from each CSV file in the corresponding position.

    Examples:
        plot_load_factor.py sweep tl.csv tr.csv bl.csv br.csv
        plot_load_factor.py sweep tl.csv tr.csv bl.csv br.csv -o custom/dir
    """
    # Load data from all four CSV files
    csv_files = [
        (csv_top_left, "top-left"),
        (csv_top_right, "top-right"),
        (csv_bottom_left, "bottom-left"),
        (csv_bottom_right, "bottom-right"),
    ]

    benchmark_data_list = []

    for csv_file, position in csv_files:
        data = load_csv_data(csv_file)
        if not data:
            typer.secho(
                f"No throughput data found in {csv_file} ({position})",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        benchmark_data_list.append(data)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    filter_styles = get_filter_styles()

    # Get all operations from all datasets
    all_operations = set[str]()
    for data in benchmark_data_list:
        all_operations.update(data.keys())

    # Grid positions: (row, col)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for operation in sorted(all_operations):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")

        # Collect handles and labels for combined legend
        all_handles = []
        all_labels = []

        for data, (row, col) in zip(benchmark_data_list, positions):
            ax = axes[row, col]
            has_data = operation in data and data[operation]

            # Show ylabel only on left column, xlabel only on bottom row
            show_ylabel = col == 0
            show_xlabel = row == 1

            if has_data:
                handles, plot_labels = plot_operation_on_axis(
                    ax,
                    operation,
                    data,
                    filter_styles,
                    show_ylabel=show_ylabel,
                    show_xlabel=show_xlabel,
                )
                # Only add handles/labels that aren't already in the combined list
                for handle, lbl in zip(handles, plot_labels):
                    if lbl not in all_labels:
                        all_handles.append(handle)
                        all_labels.append(lbl)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

        # Create combined legend above the plots
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

        plt.tight_layout(rect=(0, 0, 1, 0.95))

        output_file = (
            output_dir / f"load_factor_{operation.lower().replace(' ', '_')}.pdf"
        )
        pu.save_figure(
            fig,
            output_file,
            f"{operation} throughput comparison plot saved to {output_file}",
            close=True,
        )


# (label, hatch_pattern, alpha)
BAR_OPERATIONS = [
    ("Insert", "//", 1.0),
    ("Query (+)", None, 1.0),
    ("Query (\u2212)", r"\\", 1.0),
    ("Delete", "--", 1.0),
]

_SINGLE_BAR_WIDTH = 0.24
_SINGLE_OP_STRIDE = _SINGLE_BAR_WIDTH
_PAIRED_BAR_WIDTH = 0.145
_PAIRED_OP_STRIDE = 0.30
_PAIRED_MEM_OFFSET = 0.075
_PAIRED_FILTER_SPACING = 1.36
_GDDR7_PAIRED_ALPHA_SCALE = 0.55
_GDDR7_PAIRED_ALPHA_FLOOR = 0.35
_X_AXIS_MARGIN_LEFT = _SINGLE_BAR_WIDTH
_X_AXIS_MARGIN_RIGHT = _SINGLE_BAR_WIDTH
_CPU_CUCKOO_FILTER = "CPU Cuckoo"
_PCF_FILTER = "Partitioned Cuckoo"


def extract_fixed_lf_data(
    benchmark_data: dict, load_factor: float
) -> dict[str, dict[str, float]]:
    """Extract throughput values at a single load factor.

    Args:
        benchmark_data: Output of :func:`load_csv_data` –
            ``{operation: {filter_key: {load_factor: throughput}}}``.
        load_factor: The load factor to extract (e.g. 0.95).

    Returns:
        ``{filter_name: {operation_label: throughput_mops}}`` where
        *operation_label* matches the labels in :data:`BAR_OPERATIONS`.
    """
    result: dict[str, dict[str, float]] = defaultdict(dict)

    for op_type, filter_data in benchmark_data.items():
        for filter_key, lf_data in filter_data.items():
            base_filter = filter_key.replace(" (Positive)", "").replace(
                " (Negative)", ""
            )
            tp = lf_data.get(load_factor)
            if tp is None:
                continue

            if op_type == "Query":
                if "(Positive)" in filter_key:
                    result[base_filter]["Query (+)"] = tp
                elif "(Negative)" in filter_key:
                    result[base_filter]["Query (\u2212)"] = tp
            else:
                result[base_filter][op_type] = tp

    return dict(result)


def drop_filters(
    lf_data: dict[str, dict[str, float]], excluded_filters: set[str]
) -> dict[str, dict[str, float]]:
    """Remove excluded filters from a fixed-load-factor data map."""
    return {
        filter_name: op_data
        for filter_name, op_data in lf_data.items()
        if filter_name not in excluded_filters
    }


def apply_single_filter_override(
    primary_data: dict[str, dict[str, float]],
    secondary_data: dict[str, dict[str, float]],
    override_data: dict[str, dict[str, float]],
    filter_name: str,
) -> bool:
    """Override one filter in *primary_data* and remove it from *secondary_data*.

    Returns:
        True if override data for ``filter_name`` existed and was applied.
    """
    override_values = override_data.get(filter_name)
    if not override_values:
        return False

    primary_data[filter_name] = override_values
    secondary_data.pop(filter_name, None)
    return True


def plot_bar_on_axis(
    ax: plt.Axes,
    data: dict[str, dict[str, float]],
    filter_order: list[str],
    title: Optional[str] = None,
    show_ylabel: bool = True,
    bg_data: Optional[dict[str, dict[str, float]]] = None,
    single_source_filters: Optional[set[str]] = None,
) -> list[Patch]:
    """Plot a clustered bar chart of filter throughput on *ax*.

    Each filter occupies a position on the x-axis with up to four
    sub-bars (one per operation), distinguished by hatch pattern.

    If `bg_data` is provided, HBM3 and GDDR7 are rendered as
    side-by-side paired bars.

    Returns:
        Legend patch elements (filter colours + operation hatches + memory systems).
    """
    n_ops = len(BAR_OPERATIONS)
    has_bg = bg_data is not None and len(bg_data) > 0
    single_source_filters = single_source_filters or set()

    if has_bg:
        # Side-by-side memory-system bars for direct comparison without occlusion.
        bg_data_asserted = bg_data or {}
        for filter_idx, filter_name in enumerate(filter_order):
            filter_center = filter_idx * _PAIRED_FILTER_SPACING
            hbm_data = data.get(filter_name, {})
            gddr_data = bg_data_asserted.get(filter_name, {})
            single_source = filter_name in single_source_filters

            for op_idx, (op_label, hatch, alpha) in enumerate(BAR_OPERATIONS):
                cluster_center = (
                    filter_center + (op_idx - (n_ops - 1) / 2) * _PAIRED_OP_STRIDE
                )
                hbm_tp = hbm_data.get(op_label, 0)
                gddr_tp = gddr_data.get(op_label, 0)

                if hbm_tp > 0:
                    hbm_x = (
                        cluster_center
                        if single_source
                        else cluster_center - _PAIRED_MEM_OFFSET
                    )
                    ax.bar(
                        hbm_x,
                        hbm_tp,
                        _PAIRED_BAR_WIDTH,
                        color=pu.FILTER_COLORS.get(filter_name, "#333333"),
                        edgecolor="black",
                        linewidth=pu.BAR_EDGE_WIDTH,
                        hatch=hatch,
                        alpha=alpha,
                        zorder=3,
                    )

                if gddr_tp > 0 and not single_source:
                    gddr_alpha = max(
                        _GDDR7_PAIRED_ALPHA_FLOOR, alpha * _GDDR7_PAIRED_ALPHA_SCALE
                    )
                    ax.bar(
                        cluster_center + _PAIRED_MEM_OFFSET,
                        gddr_tp,
                        _PAIRED_BAR_WIDTH,
                        color=pu.FILTER_COLORS.get(filter_name, "#333333"),
                        edgecolor="#666666",
                        linewidth=pu.BAR_EDGE_WIDTH,
                        hatch=hatch,
                        alpha=gddr_alpha,
                        zorder=2,
                    )
    else:
        # Single-system bars.
        for filter_idx, filter_name in enumerate(filter_order):
            filter_data = data.get(filter_name, {})
            for op_idx, (op_label, hatch, alpha) in enumerate(BAR_OPERATIONS):
                tp = filter_data.get(op_label, 0)
                if tp <= 0:
                    continue

                x = filter_idx + (op_idx - (n_ops - 1) / 2) * _SINGLE_OP_STRIDE
                ax.bar(
                    x,
                    tp,
                    _SINGLE_BAR_WIDTH,
                    color=pu.FILTER_COLORS.get(filter_name, "#333333"),
                    edgecolor="black",
                    linewidth=pu.BAR_EDGE_WIDTH,
                    hatch=hatch,
                    alpha=alpha,
                    zorder=3,
                )

    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA, zorder=0)
    ax.set_xticks([])
    if filter_order:
        if has_bg:
            pair_half_span = (
                ((n_ops - 1) / 2) * _PAIRED_OP_STRIDE
                + _PAIRED_MEM_OFFSET
                + (_PAIRED_BAR_WIDTH / 2)
            )
            x_min = -pair_half_span - _X_AXIS_MARGIN_LEFT
            x_max = (
                (len(filter_order) - 1) * _PAIRED_FILTER_SPACING
                + pair_half_span
                + _X_AXIS_MARGIN_RIGHT
            )
        else:
            single_half_span = ((n_ops - 1) / 2) * _SINGLE_OP_STRIDE + (
                _SINGLE_BAR_WIDTH / 2
            )
            x_min = -single_half_span - _X_AXIS_MARGIN_LEFT
            x_max = len(filter_order) - 1 + single_half_span + _X_AXIS_MARGIN_RIGHT
        ax.set_xlim(x_min, x_max)
    ax.set_xlabel("")

    if title:
        ax.set_title(title, fontsize=pu.TITLE_FONT_SIZE, fontweight="bold")

    if show_ylabel:
        ax.set_ylabel(
            "Throughput [M ops/s]",
            fontsize=pu.AXIS_LABEL_FONT_SIZE,
            fontweight="bold",
        )

    # Build legend elements: filter colours, then operation hatches, then memory systems
    legend_elements: list[Patch] = [
        Patch(
            facecolor=pu.FILTER_COLORS.get(name, "#333333"),
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
            label=name,
        )
        for name in filter_order
    ]
    for op_label, hatch, alpha in BAR_OPERATIONS:
        legend_elements.append(
            Patch(
                facecolor="gray",
                edgecolor="black",
                linewidth=pu.BAR_EDGE_WIDTH,
                hatch=hatch,
                alpha=alpha,
                label=op_label,
            )
        )

    # Add memory system legend entries in paired-memory mode.
    if has_bg:
        legend_elements.append(
            Patch(
                facecolor="gray",
                edgecolor="black",
                linewidth=pu.BAR_EDGE_WIDTH,
                alpha=1.0,
                label="HBM3",
            )
        )
        gddr_alpha = max(_GDDR7_PAIRED_ALPHA_FLOOR, _GDDR7_PAIRED_ALPHA_SCALE)
        legend_elements.append(
            Patch(
                facecolor="gray",
                edgecolor="#666666",
                linewidth=pu.BAR_EDGE_WIDTH,
                alpha=gddr_alpha,
                label="GDDR7",
            )
        )

    return legend_elements


@app.command("bar")
def bar(
    csv_hbm3_left: Path = typer.Argument(
        ...,
        help="Path to HBM3 CSV for left subplot (e.g., gh200_28.csv)",
    ),
    csv_hbm3_right: Path = typer.Argument(
        ...,
        help="Path to HBM3 CSV for right subplot (e.g., gh200_22.csv)",
    ),
    csv_gddr7_left: Optional[Path] = typer.Option(
        None,
        "--bg-left",
        help="Path to GDDR7 CSV for left subplot (e.g., mon02_28.csv)",
    ),
    csv_gddr7_right: Optional[Path] = typer.Option(
        None,
        "--bg-right",
        help="Path to GDDR7 CSV for right subplot (e.g., mon02_22.csv)",
    ),
    csv_pcf_left: Optional[Path] = typer.Option(
        None,
        "--pcf-left",
        help="Path to PCF-only CSV for left subplot (single-series override).",
    ),
    csv_pcf_right: Optional[Path] = typer.Option(
        None,
        "--pcf-right",
        help="Path to PCF-only CSV for right subplot (single-series override).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
    load_factor_pct: float = typer.Option(
        95,
        "--load-factor",
        "-l",
        help="Load factor as a percentage (e.g. 95 for 95%)",
    ),
):
    """
    Generate a bar chart comparing filter throughput at a fixed load factor.

    Creates a 1x2 figure with one subplot per CSV file.  Within each subplot,
    filters are on the x-axis and operations (Insert, Query+, Query-, Delete)
    are shown as clustered bars with distinct hatch patterns.

    If --bg-left and/or --bg-right are provided, those CSVs are treated as
    GDDR7 data and compared against HBM3 using side-by-side paired bars.
    If --pcf-left and/or --pcf-right are provided, the Partitioned Cuckoo
    results for that subplot are taken from those CSVs and plotted as a
    single system (no HBM3/GDDR7 pair).

    Examples:
        plot_load_factor.py bar gh200_28.csv gh200_22.csv
        plot_load_factor.py bar gh200_28.csv gh200_22.csv --bg-left mon02_28.csv --bg-right mon02_22.csv
        plot_load_factor.py bar gh200_22.csv gh200_28.csv --bg-left mon02_22.csv --bg-right mon02_28.csv --pcf-left mon03_22_95.csv --pcf-right mon03_28_95.csv
    """
    load_factor = load_factor_pct / 100.0

    # Load foreground (HBM3) data
    data_hbm3_left = load_csv_data(csv_hbm3_left)
    data_hbm3_right = load_csv_data(csv_hbm3_right)

    if not data_hbm3_left or not data_hbm3_right:
        typer.secho(
            "No data found in one or both HBM3 CSV files",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Load background (GDDR7) data if provided
    data_gddr7_left = None
    data_gddr7_right = None

    if csv_gddr7_left:
        data_gddr7_left = load_csv_data(csv_gddr7_left)

    if csv_gddr7_right:
        data_gddr7_right = load_csv_data(csv_gddr7_right)

    # Optional PCF-only override inputs (e.g., different CPU system)
    data_pcf_left = load_csv_data(csv_pcf_left) if csv_pcf_left else None
    data_pcf_right = load_csv_data(csv_pcf_right) if csv_pcf_right else None

    output_dir_resolved = pu.resolve_output_dir(output_dir, Path(__file__))

    # Extract data at the fixed load factor
    lf_hbm3_left = extract_fixed_lf_data(data_hbm3_left, load_factor)
    lf_hbm3_right = extract_fixed_lf_data(data_hbm3_right, load_factor)

    lf_gddr7_left = (
        extract_fixed_lf_data(data_gddr7_left, load_factor) if data_gddr7_left else {}
    )
    lf_gddr7_right = (
        extract_fixed_lf_data(data_gddr7_right, load_factor) if data_gddr7_right else {}
    )
    lf_pcf_left = (
        extract_fixed_lf_data(data_pcf_left, load_factor) if data_pcf_left else {}
    )
    lf_pcf_right = (
        extract_fixed_lf_data(data_pcf_right, load_factor) if data_pcf_right else {}
    )

    # Always drop CPU Cuckoo from this chart.
    excluded_filters = {_CPU_CUCKOO_FILTER}
    lf_hbm3_left = drop_filters(lf_hbm3_left, excluded_filters)
    lf_hbm3_right = drop_filters(lf_hbm3_right, excluded_filters)
    lf_gddr7_left = drop_filters(lf_gddr7_left, excluded_filters)
    lf_gddr7_right = drop_filters(lf_gddr7_right, excluded_filters)
    lf_pcf_left = drop_filters(lf_pcf_left, excluded_filters)
    lf_pcf_right = drop_filters(lf_pcf_right, excluded_filters)

    # Per-subplot filters that should be single-system in paired mode.
    single_source_filters_left: set[str] = set()
    single_source_filters_right: set[str] = set()

    if csv_pcf_left:
        if apply_single_filter_override(
            lf_hbm3_left,
            lf_gddr7_left,
            lf_pcf_left,
            _PCF_FILTER,
        ):
            single_source_filters_left.add(_PCF_FILTER)
        else:
            typer.secho(
                f"No {_PCF_FILTER} data found at load factor {load_factor_pct}% in {csv_pcf_left}",
                fg=typer.colors.YELLOW,
                err=True,
            )

    if csv_pcf_right:
        if apply_single_filter_override(
            lf_hbm3_right,
            lf_gddr7_right,
            lf_pcf_right,
            _PCF_FILTER,
        ):
            single_source_filters_right.add(_PCF_FILTER)
        else:
            typer.secho(
                f"No {_PCF_FILTER} data found at load factor {load_factor_pct}% in {csv_pcf_right}",
                fg=typer.colors.YELLOW,
                err=True,
            )

    has_bg = bool(lf_gddr7_left) or bool(lf_gddr7_right)

    if not lf_hbm3_left and not lf_hbm3_right:
        typer.secho(
            f"No data found at load factor {load_factor_pct}%. "
            "Available load factors (in the left CSV): "
            + ", ".join(
                sorted(
                    {
                        str(lf)
                        for ops in data_hbm3_left.values()
                        for lfs in ops.values()
                        for lf in lfs
                    }
                )
            ),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Determine filter order (union of all filters across all datasets)
    all_filters = set(lf_hbm3_left.keys()) | set(lf_hbm3_right.keys())
    if has_bg:
        all_filters |= set(lf_gddr7_left.keys()) | set(lf_gddr7_right.keys())
    filter_order = [
        name
        for name in pu.FILTER_DISPLAY_NAMES.values()
        if name in all_filters and name != _CPU_CUCKOO_FILTER
    ]

    # Scale figure width with filter count so bars can use horizontal space even
    # when the top filter legend becomes very wide.
    fig_width = max(12.0, 2.1 * len(filter_order) + 1.5)
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(fig_width, 5), sharey=True, gridspec_kw={"wspace": 0.04}
    )

    legend_left = plot_bar_on_axis(
        ax_left,
        lf_hbm3_left,
        filter_order,
        title=None,
        show_ylabel=True,
        bg_data=lf_gddr7_left if lf_gddr7_left else None,
        single_source_filters=single_source_filters_left,
    )
    legend_right = plot_bar_on_axis(
        ax_right,
        lf_hbm3_right,
        filter_order,
        title=None,
        show_ylabel=False,
        bg_data=lf_gddr7_right if lf_gddr7_right else None,
        single_source_filters=single_source_filters_right,
    )

    # Use the legend with more entries
    legend_elements = (
        legend_left if len(legend_left) >= len(legend_right) else legend_right
    )

    # Split legend into rows: filters | operations | (optional: memory systems)
    n_filters = len(filter_order)
    n_ops = len(BAR_OPERATIONS)
    filter_handles = legend_elements[:n_filters]
    op_handles = legend_elements[n_filters : n_filters + n_ops]
    mem_handles = legend_elements[n_filters + n_ops :] if has_bg else []

    legend_y_top = 0.99
    legend_row_step = 0.075

    fig.legend(
        handles=filter_handles,
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y_top),
        ncol=len(filter_handles),
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )
    fig.legend(
        handles=op_handles,
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y_top - legend_row_step),
        ncol=len(op_handles),
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )
    if mem_handles:
        mem_legend_y = legend_y_top - (2 * legend_row_step)
        fig.legend(
            handles=mem_handles,
            fontsize=pu.LEGEND_FONT_SIZE,
            loc="upper center",
            bbox_to_anchor=(0.5, mem_legend_y),
            ncol=len(mem_handles),
            framealpha=pu.LEGEND_FRAME_ALPHA,
        )
        axes_top = mem_legend_y - 0.10
    else:
        axes_top = (legend_y_top - legend_row_step) - 0.10

    # Reserve explicit space for figure-level legends to avoid overlap with axes.
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.08, top=axes_top, wspace=0.04)

    lf_str = (
        str(int(load_factor_pct))
        if load_factor_pct == int(load_factor_pct)
        else str(load_factor_pct).replace(".", "_")
    )
    output_file = output_dir_resolved / f"load_factor_bar_{lf_str}.pdf"
    pu.save_figure(
        fig,
        output_file,
        f"Bar chart at {load_factor_pct}% load factor saved to {output_file}",
        close=True,
    )


if __name__ == "__main__":
    app()
