#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "pandas",
#     "seaborn"
# ]
# ///
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns

df = pd.read_csv(sys.stdin)

plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)

fig, (ax1, ax2, ax3) = plt.subplots(
    1,
    3,
    figsize=(24, 8),
    gridspec_kw={"width_ratios": [1, 1, 1.2]},
)
# fig.suptitle("Cuckoo Hash Table Performance Analysis", fontsize=18, y=1.02)


def create_performance_heatmap(df, table_type, ax):
    subset = df[df["tableType"] == table_type].copy()

    subset["normalized_time"] = subset.groupby("exponent")["avgTotalTimeMs"].transform(
        lambda x: x / x.min()
    )

    pivot_table = subset.pivot(
        index="exponent",
        columns="bucketSize",
        values="normalized_time",
    )

    sns.heatmap(
        pivot_table,
        ax=ax,
        annot=False,
        cmap="rocket_r",
        cbar_kws={"label": "Performance Ratio (1.0 = Optimal)"},
        vmin=1.0,
        vmax=2.0,
    )

    ax.set_title(
        f"Performance vs. Bucket Size ({table_type.replace('BucketsTable', '')})"
    )
    ax.set_xlabel("Bucket Size")
    ax.set_ylabel("Input Size ")
    ax.set_yticklabels([f"$2^{{{exp}}}$" for exp in pivot_table.index], rotation=0)


create_performance_heatmap(df, "BucketsTableCpu", ax1)
create_performance_heatmap(df, "BucketsTableGpu", ax2)


gpu_data = df[df["tableType"] == "BucketsTableGpu"]

throughput_data = []
for exp in sorted(gpu_data["exponent"].unique()):
    subset = gpu_data[gpu_data["exponent"] == exp]
    avg_insert = subset["insertThroughputMops"].mean()
    avg_query = subset["queryThroughputMops"].mean()
    throughput_data.append(
        {
            "exponent": exp,
            "insert_throughput": avg_insert,
            "query_throughput": avg_query,
        }
    )

throughput_df = pd.DataFrame(throughput_data)

input_sizes_log = throughput_df["exponent"]
insert_throughput = throughput_df["insert_throughput"]
query_throughput = throughput_df["query_throughput"]

ax3.plot(
    input_sizes_log,
    insert_throughput,
    "o-",
    linewidth=2.5,
    markersize=8,
    label="Insert Throughput",
    color="#F18F01",
    markerfacecolor="white",
    markeredgewidth=2,
)
ax3.plot(
    input_sizes_log,
    query_throughput,
    "s-",
    linewidth=2.5,
    markersize=8,
    label="Query Throughput",
    color="#C73E1D",
    markerfacecolor="white",
    markeredgewidth=2,
)

ax3.set_xlabel("Input Size")
ax3.set_ylabel("Throughput (MOPS)")
ax3.set_title("GPU Throughput: Insert vs Query")
ax3.legend()
ax3.grid(True, alpha=0.3)

ax3.set_xticks(input_sizes_log)
ax3.set_xticklabels(
    [f"$2^{{{int(exp)}}}$" for exp in sorted(gpu_data["exponent"].unique())],
    ha="right",
)


plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    "performance_analysis_plot.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
