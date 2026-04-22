from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_cumulative_variance(
    variance_table: pd.DataFrame,
    variance_threshold: float,
    output_path: Path,
) -> None:
    """Plot cumulative explained variance and mark selected threshold."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        variance_table["component"],
        variance_table["cumulative_explained_variance"],
        marker="o",
        linewidth=2,
    )
    ax.axhline(
        y=variance_threshold,
        color="red",
        linestyle="--",
        label=f"Threshold ({variance_threshold:.0%})",
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA cumulative explained variance")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_aic_bic(
    scores: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot AIC and BIC over candidate cluster counts."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scores["k"], scores["aic"], marker="o", linewidth=2, label="AIC")
    ax.plot(scores["k"], scores["bic"], marker="o", linewidth=2, label="BIC")
    ax.set_xlabel("Number of GMM components (k)")
    ax.set_ylabel("Score")
    ax.set_title("GMM model selection with AIC and BIC")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cluster_phase_heatmap(
    normalized_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot normalized cluster vs phase cross-tab heatmap."""

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        normalized_table,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        cbar_kws={"label": "Row-normalized proportion"},
        ax=ax,
    )
    ax.set_xlabel("Phase")
    ax.set_ylabel("Cluster")
    ax.set_title("Cluster to Phase alignment")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
