from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def cluster_size_stats(labels: pd.Series) -> dict[str, float]:
    sizes = labels.value_counts(dropna=True)
    if len(sizes) == 0:
        return {
            "num_clusters": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }
    return {
        "num_clusters": float(len(sizes)),
        "mean": float(sizes.mean()),
        "min": float(sizes.min()),
        "max": float(sizes.max()),
        "median": float(sizes.median()),
    }


def plot_cluster_size_distribution(
    labels: pd.Series,
    *,
    title: str,
    max_bins: int = 80,
) -> go.Figure:
    """
    Plot histogram of cluster sizes (log-log).
    """
    sizes = labels.value_counts(dropna=True).astype(int).values
    if sizes.size == 0:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    max_size = int(sizes.max())
    max_bins = max(10, int(max_bins))
    # log bins from 1 to max_size+1
    bins = np.logspace(0, np.log10(max_size + 1), num=min(max_bins, max_size) + 1)
    counts, edges = np.histogram(sizes, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Compute key stats for annotation
    mean_sz = sizes.mean()
    median_sz = np.median(sizes)
    min_sz = sizes.min()
    total_clusters = len(sizes)

    subtitle = (
        f"Total: {total_clusters:,} | Max: {max_size:,} | Min: {min_sz:,} | "
        f"Mean: {mean_sz:.1f} | Median: {median_sz:.1f}"
    )
    full_title = f"{title}<br><sup style='color:gray;font-size:12px'>{subtitle}</sup>"

    fig = go.Figure(
        data=[
            go.Bar(
                x=centers,
                y=counts,
                marker=dict(color="#1f77b4"),
                hovertemplate="Size ≈ %{x:.0f}<br>Count: %{y:,}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=full_title, x=0, xanchor="left"),
        xaxis_title="Cluster Size (log scale)",
        yaxis_title="Count of Clusters",
        height=450,
        margin=dict(l=20, r=20, t=80, b=40),
        autosize=True,
        bargap=0.1,
    )
    fig.update_xaxes(type="log", tickformat=".0f", gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(type="log", gridcolor="rgba(0,0,0,0.05)")
    return fig

