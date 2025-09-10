#!/usr/bin/env -S uv run --script
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


df = pd.read_csv(sys.stdin)

plt.figure(figsize=(12, 8))
colors = ["blue", "red", "green"]
for i, tableType in enumerate(df["tableType"].unique()):
    data = df[df["tableType"] == tableType]
    x_values = data["n"].apply(lambda x: int(x).bit_length() - 1)
    y_values = data["avgTimeMs"]

    plt.plot(
        x_values,
        y_values,
        marker="o",
        label=tableType,
        linewidth=2,
        color=colors[i % len(colors)],
    )

    slowest_idx = y_values.idxmax()
    slowest_x = x_values.loc[slowest_idx]
    slowest_y = y_values.loc[slowest_idx]

    plt.annotate(
        f"{int(slowest_y // 60000)}m {(slowest_y % 60000) / 1000:.1f}s",
        xy=(slowest_x, slowest_y),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor=colors[i % len(colors)], alpha=0.7
        ),
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=0", color=colors[i % len(colors)]
        ),
    )


plt.xlabel(r"Input Size ($2^x$)")
plt.xticks(range(int(df["n"].min()).bit_length() - 1, int(df["n"].max()).bit_length()))

plt.ylabel("Average Runtime (ms)")
plt.title("Performance Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("scaling_plot.png", dpi=300, bbox_inches="tight")
plt.close()
