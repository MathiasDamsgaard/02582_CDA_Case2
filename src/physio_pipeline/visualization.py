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
    """Plot AIC and BIC over candidate GMM cluster counts."""

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


def plot_kmeans_elbow(
    scores: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot K-means inertia over candidate cluster counts."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scores["k"], scores["inertia"], marker="o", linewidth=2)
    ax.set_xlabel("Number of K-means clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-means elbow curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_kmeans_silhouette(
    scores: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot K-means silhouette scores over candidate cluster counts."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scores["k"], scores["silhouette"], marker="o", linewidth=2)
    ax.set_xlabel("Number of K-means clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("K-means model selection with silhouette score")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_gap_statistic(
    gap_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the Gap Statistic against candidate cluster counts."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gap_df["k"], gap_df["gap"], marker="o", linewidth=2)
    # ax.errorbar(
    #     gap_df["k"],
    #     gap_df["gap"],
    #     yerr=gap_df["s_k"],
    #     marker="o",
    #     linewidth=2,
    #     capsize=5,
    # )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Gap Statistic")
    ax.set_title("K-Means Model Selection using Gap Statistic")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cluster_phase_heatmap(
    normalized_table: pd.DataFrame,
    output_path: Path,
    title: str = "Cluster to Phase alignment",
) -> None:
    """Plot normalized cluster vs phase cross-tab heatmap."""

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        normalized_table,
        annot=False,
        cmap="YlOrBr",
        cbar_kws={"label": "Row-normalized proportion"},
        ax=ax,
    )

    for row_idx in range(normalized_table.shape[0]):
        for col_idx in range(normalized_table.shape[1]):
            value = normalized_table.iloc[row_idx, col_idx]

            text_color = "white" if value >= 0.45 else "black"

            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    ax.set_xlabel("Phase")
    ax.set_ylabel("Cluster")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)