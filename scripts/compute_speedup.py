#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "typer",
# ]
# ///

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(help="Compute speedups of Cuckoo Filter vs other filters")

FILTER_NAMES = {
    "CF": "GPU Cuckoo",
    "BBF": "Blocked Bloom",
    "TCF": "TCF",
    "GQF": "GQF",
}


def parse_benchmark_data(
    df: pd.DataFrame,
    load_factor: int | None = None,
) -> dict[str, dict[str, dict[int, float]]]:
    """Parse CSV into {filter: {operation: {size: throughput}}} structure.

    Args:
        df: DataFrame with benchmark results
        load_factor: If specified, only include data for this load factor percentage
    """
    df = df[df["name"].str.endswith("_median")]

    data = defaultdict(lambda: defaultdict(dict))

    for _, row in df.iterrows():
        name = row["name"]
        if "/" not in name:
            continue

        parts = name.split("/")
        if len(parts) < 3:
            continue

        first_part = parts[0].strip('"')

        # Handle load factor format: "CF_50/Insert/268435456/..."
        # or Fixture format: "CFFixture/Insert/65536/..."
        if first_part.endswith("Fixture"):
            # Fixture format (no load factor)
            if load_factor is not None:
                continue  # Skip if user wants specific load factor
            filter_name = first_part[:-7]
            operation = parts[1]
            size_str = parts[2]
            lf = None
        elif "_" in first_part:
            # Load factor format: "CF_50/Insert/..." or old format "CF_Insert/..."
            prefix, suffix = first_part.rsplit("_", 1)

            # Check if suffix is a number (load factor) or operation name
            if suffix.isdigit():
                filter_name = prefix
                lf = int(suffix)
                operation = parts[1]
                size_str = parts[2] if len(parts) > 2 else parts[1]
            else:
                # Old format: CF_Insert
                filter_name = prefix
                operation = suffix
                size_str = parts[1]
                lf = None
        else:
            continue

        # Filter by load factor if specified
        if load_factor is not None and lf != load_factor:
            continue

        # Only consider Insert, Query (Positive & Negative), Delete operations
        if not re.fullmatch(r"(?:Query|QueryNegative|Insert|Delete)", operation):
            continue

        try:
            size = int(size_str)
            items_per_second = row.get("items_per_second")
            if pd.notna(items_per_second):
                data[filter_name][operation][size] = items_per_second
        except (ValueError, KeyError):
            continue

    return data


@app.command()
def main(
    csv_file: Path = typer.Argument(..., help="Path to benchmark CSV file"),
    baseline: str = typer.Option("CF", "--baseline", "-b", help="Baseline filter name"),
    load_factor: int = typer.Option(
        None,
        "--load-factor",
        "-l",
        help="Load factor percentage to filter by (e.g., 50)",
    ),
    size: int = typer.Option(
        None, "--size", "-s", help="Specific size to compare (default: largest)"
    ),
):
    """
    Compute relative speedups of the Cuckoo Filter compared to other filters.

    Outputs a table showing how many times faster/slower CF is vs each filter.

    Examples:
        ./compute_speedup.py benchmark_lf.csv -l 50
        ./compute_speedup.py benchmark_lf.csv -l 95 -s 67108864
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        typer.secho(f"Error reading CSV: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    data = parse_benchmark_data(df, load_factor)

    if not data:
        typer.secho(
            "No data found"
            + (f" for load factor {load_factor}%" if load_factor else ""),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    if baseline not in data:
        typer.secho(
            f"Baseline filter '{baseline}' not found. Available: {list(data.keys())}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    baseline_data = data[baseline]
    other_filters = [f for f in data.keys() if f != baseline]

    if not other_filters:
        typer.secho("No other filters found to compare", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Get operations present in baseline
    operations = list(baseline_data.keys())

    # Determine which size to use
    if size is None:
        # Use the largest size available
        all_sizes = set()
        for op_data in baseline_data.values():
            all_sizes.update(op_data.keys())
        size = max(all_sizes) if all_sizes else 0

    title = f"\nSpeedups for {FILTER_NAMES.get(baseline, baseline)} vs other filters"
    if load_factor is not None:
        title += f" at {load_factor}% load factor"
    title += f" (n={size:,})\n"

    typer.secho(title)
    col_width = max(len(op) for op in operations) + 2
    col_width = max(col_width, 11)
    table_width = 20 + 3 + len(operations) * (col_width + 3)

    # Print header
    header = f"{'Filter':<20} | " + " | ".join(
        f"{op:>{col_width}}" for op in operations
    )
    typer.echo(header)
    typer.secho("-" * table_width)

    for other_filter in sorted(other_filters):
        display_name = FILTER_NAMES.get(other_filter, other_filter)
        row = f"{display_name:<20} | "
        speedups = []

        for op in operations:
            baseline_throughput = baseline_data.get(op, {}).get(size)
            other_throughput = data[other_filter].get(op, {}).get(size)

            if baseline_throughput and other_throughput:
                speedup = baseline_throughput / other_throughput
                speedups.append(f"{speedup:>{col_width - 1}.2f}x")
            else:
                speedups.append(f"{'N/A':>{col_width}}")

        row += " | ".join(speedups)
        typer.echo(row)

    typer.secho("-" * table_width)


if __name__ == "__main__":
    app()
