#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

BENCHMARK_EXE="$BUILD_DIR/benchmark/load-factor"
OUTPUT_CSV="$BUILD_DIR/load_factor_benchmark.csv"

if [ ! -f "$BENCHMARK_EXE" ]; then
    echo "Error: Benchmark executable not found at $BENCHMARK_EXE" >&2
    echo "Please build the project first with: meson compile -C build" >&2
    exit 1
fi

echo "Running load factor benchmarks..."
echo "Output will be saved to: $OUTPUT_CSV"

"$BENCHMARK_EXE" --benchmark_format=csv | tee "$OUTPUT_CSV"

echo ""
echo "Benchmark complete! Results saved to $OUTPUT_CSV"
echo ""
echo "Generating plots..."
"$SCRIPT_DIR/plot_load_factor.py" "$OUTPUT_CSV"
