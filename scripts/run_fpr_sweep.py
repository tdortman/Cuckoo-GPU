#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "typer",
# ]
# ///

from pathlib import Path

import benchmark_utils as bu
import typer

app = typer.Typer()


@app.command()
def main(
    output: Path = typer.Option(
        Path("build/fpr_sweep.csv"),
        "--output",
        "-o",
        help="Output CSV file path",
    ),
):
    """Run all FPR sweep benchmarks and combine results into a single CSV."""
    benchmark_dir = bu.get_benchmark_dir(Path(__file__))

    benchmark_names = [
        "fpr-sweep-gqf8",
        "fpr-sweep-gqf16",
        "fpr-sweep-gqf32",
    ]

    benchmarks = [(benchmark_dir / name, None, None) for name in benchmark_names]

    bu.run_benchmarks_and_merge(benchmarks, output)

    typer.secho(f"\nResults written to {output}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
