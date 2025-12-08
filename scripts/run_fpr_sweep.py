#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "typer",
# ]
# ///

import subprocess
import tempfile
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    output: Path = typer.Option(
        Path("build/fpr_sweep.csv"),
        "--output", "-o",
        help="Output CSV file path",
    ),
):
    """Run all FPR sweep benchmarks and combine results into a single CSV."""
    build_dir = Path(__file__).parent.parent / "build"

    benchmarks = [
        "benchmark-fpr-sweep-gqf8",
        "benchmark-fpr-sweep-gqf16",
        "benchmark-fpr-sweep-gqf32",
    ]

    all_lines = []
    header = None

    for bench in benchmarks:
        bench_path = build_dir / bench
        if not bench_path.exists():
            typer.echo(
                f"Error: {bench_path} not found. Did you run 'meson compile -C build'?",
                err=True,
            )
            raise typer.Exit(1)

        # Write CSV to a temp file so benchmark output goes to terminal
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
            csv_path = f.name

        result = subprocess.run(
            [str(bench_path), f"--benchmark_out={csv_path}", "--benchmark_out_format=csv", "--benchmark_format=csv"],
        )

        if result.returncode != 0:
            typer.echo(f"Error running {bench}", err=True)
            raise typer.Exit(1)

        # Read CSV from temp file
        with open(csv_path) as f:
            lines = [line.rstrip() for line in f if line.strip()]

        Path(csv_path).unlink()  # Clean up temp file

        if not lines:
            continue

        if header is None:
            header = lines[0]
            all_lines.append(header)
            all_lines.extend(lines[1:])
        else:
            if lines[0] == header:
                all_lines.extend(lines[1:])
            else:
                all_lines.extend(lines)

    # Write combined CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for line in all_lines:
            f.write(line + "\n")

    typer.echo(f"\nResults written to {output}")


if __name__ == "__main__":
    app()
