#!/usr/bin/env -s uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "pandas"
# ]
# ///

import pandas as pd
import matplotlib.pyplot as plt
import sys
from io import StringIO


def main():
    # Read CSV data from stdin
    data = sys.stdin.read()

    # Parse the CSV data
    df = pd.read_csv(StringIO(data))

    implementation_names = {
        0: "HybridTable",
        1: "BucketsTableCpu",
        2: "BucketsTableGpu",
    }
    df["implementation_name"] = df["implementation"].map(implementation_names)

    df["filter_size"] = 2 ** df["exponent"]

    plt.figure(figsize=(12, 8))

    colors = ["blue", "green", "red"]
    for i, impl in enumerate(df["implementation"].unique()):
        impl_data = df[df["implementation"] == impl]
        impl_name = implementation_names[impl]

        plt.errorbar(
            impl_data["filter_size"],
            impl_data["mean"],
            yerr=impl_data["std"],
            label=impl_name,
            marker="o",
            capsize=5,
            linestyle="-",
            linewidth=2,
            color=colors[i],
        )

    plt.xscale("log", base=2)
    plt.xticks(
        df["filter_size"].unique(), [f"$2^{{{e}}}$" for e in df["exponent"].unique()]
    )
    plt.xlabel("Filter Size", fontsize=12)
    plt.ylabel("Execution Time (seconds)", fontsize=12)
    plt.title("Performance Comparison of Cuckoo Filter Implementations", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("cuckoo_filter_performance.png")
    print("Plot saved as 'cuckoo_filter_performance.png'")

    summary = df.pivot_table(
        index=["implementation_name", "exponent"],
        values=["mean", "std"],
        aggfunc="first",
    ).round(6)

    print("\nPerformance Summary:")
    print(summary)


if __name__ == "__main__":
    main()