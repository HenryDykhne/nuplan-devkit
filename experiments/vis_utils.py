import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import numpy as np  # Import numpy for numerical operations
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import scipy.stats as stats
import re
import os



pd.set_option("display.max_colwidth", 400)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)  # make huge


# we define the convention that if we are comparing two files with VISIBLE and OCCLUDED tracks, the VISIBLE file is first
def plot_diff(
    file_path_1: str,
    file_path_2: str,
    label_1: str,
    label_2: str,
    metrics: List[str],
    metric_op: List[str],
    good_metric: List[bool],
    primary: str = "scenario_name",
    group_by: str = None,
    k_differences: int = 5,
    colors: List[str] = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ],
    as_bool=False,
) -> None:
    # Read the Parquet files into pandas DataFrames
    df1 = pd.read_parquet(file_path_1)
    df2 = pd.read_parquet(file_path_2)

    # Convert all boolean columns to integers (0 or 1)
    boolean_cols_1 = df1.select_dtypes(include="bool").columns
    boolean_cols_2 = df2.select_dtypes(include="bool").columns

    df1[boolean_cols_1] = df1[boolean_cols_1].astype(int)
    df2[boolean_cols_2] = df2[boolean_cols_2].astype(int)

    if as_bool:  # convert int cols that are greater than 1 to 1
        num_to_bool_cols_1 = df1.select_dtypes(include="int").columns
        num_to_bool_cols_2 = df2.select_dtypes(include="int").columns

        df1[num_to_bool_cols_1] = df1[num_to_bool_cols_1].astype(bool).astype(int)
        df2[num_to_bool_cols_2] = df2[num_to_bool_cols_2].astype(bool).astype(int)

    # Create a figure and axis
    plt.figure(figsize=(14, 10))

    ax = plt.subplot(111)

    # If group_by is None, use one big group with all rows
    if group_by is None:
        df1["all"] = 1
        df2["all"] = 1
        group_by = "all"

    # Plotting each metric's values side by side
    width = 1 / len(metrics)
    groups = df1[group_by].unique()
    groups.sort()
    x_indices = np.arange(len(groups))
    for i, metric in enumerate(metrics):
        # Perform an inner merge based on 'scenario_name'
        merged = pd.merge(
            df1[[primary, group_by, metric]],
            df2[[primary, group_by, metric]],
            on=primary,
            suffixes=("_1", "_2"),
            how="inner",
        )
        # we remove all scenarios where either metric is null or inf, since we dont want to be comparing values that dont exist (also helps with checking if severity changes without being impacted by new occurances of lower severity)
        merged = merged.replace([np.inf, -np.inf], np.nan)
        merged = merged.dropna()

        # Calculate difference only for values existing in both files
        if good_metric[i]:
            merged["difference"] = merged[f"{metric}_1"] - merged[f"{metric}_2"]
        else:
            merged["difference"] = merged[f"{metric}_2"] - merged[f"{metric}_1"]

        # here, we get our scenarios that had the greatest differences in either direction
        if k_differences != 0:
            to_get_top_k = merged[[primary, "difference", f"{group_by}_1"]]
            print(metric + " (top " + str(k_differences) + " differences)")
            print("positive diff")
            print(
                to_get_top_k.nlargest(k_differences, "difference", "first")
            )  # we print the top k_differences largest differences
            print("negative diff")
            print(
                to_get_top_k.nsmallest(k_differences, "difference", "first")
            )  # we print the top k_differences smallest differences

        # Calculate confidence intervals (assuming normal distribution)
        std_err = merged.groupby(f"{group_by}_1")["difference"].sem()
        confidence_intervals = std_err * stats.norm.ppf(
            0.975
        )  # 95% confidence interval

        if metric_op[i] == "sum":
            grouped_diff = (
                merged.groupby(f"{group_by}_1")["difference"].sum().reset_index()
            )
            confidence_intervals = confidence_intervals * len(
                merged.groupby(f"{group_by}_1")
            )  # sum conf intervals need to be scaled by the size of each group
        elif metric_op[i] == "mean":
            grouped_diff = (
                merged.groupby(f"{group_by}_1")["difference"].mean().reset_index()
            )
        else:
            raise Exception

        # enforces a standard order on the groups and intervals for plotting
        grouped_diff[f"{group_by}_1"] = pd.Categorical(
            grouped_diff[f"{group_by}_1"], groups
        )
        grouped_diff = grouped_diff.sort_values(f"{group_by}_1").reset_index(drop=True)

        confidence_intervals = confidence_intervals.reset_index()
        confidence_intervals[f"{group_by}_1"] = pd.Categorical(
            confidence_intervals[f"{group_by}_1"], groups
        )
        confidence_intervals = confidence_intervals.sort_values(
            f"{group_by}_1"
        ).reset_index(drop=True)

        for (
            group
        ) in groups:  # some groups may be empty after the null culling. we fix this
            if group not in grouped_diff[f"{group_by}_1"].values:
                row = {f"{group_by}_1": group, "difference": 0}
                ci_row = {f"{group_by}_1": group, "difference": 0}

                grouped_diff = pd.concat(
                    [grouped_diff, pd.DataFrame([row])], ignore_index=True
                )
                confidence_intervals = pd.concat(
                    [confidence_intervals, pd.DataFrame([ci_row])], ignore_index=True
                )

                grouped_diff = grouped_diff.sort_values(f"{group_by}_1").reset_index(
                    drop=True
                )  # re-enforces a standard order on the groups
                confidence_intervals = confidence_intervals.sort_values(
                    f"{group_by}_1"
                ).reset_index(drop=True)  # re-enforces a standard order on the groups

        # colors below the bar should be twice as dark
        bar_colors = [
            (
                colors[i]
                if diff >= 0
                else mcolors.to_hex(np.array((mcolors.to_rgb(colors[i]))) / 2)
            )
            for diff in grouped_diff["difference"]
        ]
        bar_edge_colors = ["black" for diff in grouped_diff["difference"]]
        ax.bar(
            x_indices + i * width,
            grouped_diff["difference"],
            width=width,
            color=bar_colors,
            edgecolor=bar_edge_colors,
            linewidth=1,  # Adjust the width of the black outline
            label=metric
            + ", is_good: "
            + str(good_metric[i])
            + ", op: "
            + metric_op[i],
        )

        # Plotting error bars (confidence intervals)
        ax.errorbar(
            x=x_indices + i * width,
            y=grouped_diff["difference"],
            yerr=confidence_intervals["difference"],
            fmt="none",
            ecolor="black",
            capsize=5,
            capthick=1,
            linestyle="None",
        )

        # Annotating each bar with the count on top
        scaling_factor = min(
            width * 240 / len(groups), 30
        )  # Adjust this multiplier for optimal text size
        counts = merged[f"{group_by}_1"].value_counts()
        for j, group in enumerate(groups):
            if group not in counts:
                count = 0
                ax.text(
                    j + i * width,
                    0,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=scaling_factor,
                    color="red",
                )
                continue
            count = counts[group]
            ax.text(
                j + i * width,
                grouped_diff["difference"].iloc[j] / 2,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=scaling_factor,
                color="white",
                path_effects=[pe.withStroke(linewidth=4, foreground="black")]
            )

    # Set x-axis ticks and labels
    ax.set_xticks(x_indices + ((len(metrics) - 1) / 2) * width)
    ax.set_xticklabels(groups, rotation=45, ha="right")

    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Add vertical separation lines between groups
    for i in range(1, len(groups)):
        plt.axvline(x=i - (width / 2), color="black", linestyle="--", linewidth=1)

    # Add labels, title, legend, and show the plot
    plt.xlabel(group_by)
    plt.ylabel("Difference")
    plt.title(
        "Difference in Metrics between: \n("
        + label_1
        + ") and ("
        + label_2
        + ")\n (positive bars mean that the first option results in better metrics)"
    )
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    plt.show()
    
#source: https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python
def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', color='C0', fill=True, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches