from __future__ import annotations

import json

from .clustering import fit_final_gmm
from .config import PipelineConfig
from .data_io import ensure_output_directories, load_dataset
from .evaluation import evaluate_phase_alignment
from .model_selection import evaluate_gmm_candidates
from .pca_stage import run_pca
from .preprocessing import preprocess_dataset
from .visualization import plot_aic_bic, plot_cluster_phase_heatmap, plot_cumulative_variance


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, object]:
    """Execute the full unsupervised learning workflow and save artifacts."""

    cfg = config or PipelineConfig()
    figures_dir, results_dir = ensure_output_directories(cfg.output_dir)

    raw_df = load_dataset(
        csv_path=cfg.data_path,
        required_columns=cfg.required_columns,
    )

    preprocessing_result = preprocess_dataset(
        df=raw_df,
        prefixes=cfg.physiological_prefixes,
        group_column=cfg.group_column,
    )

    feature_matrix = preprocessing_result.processed_df[
        preprocessing_result.feature_columns
    ].to_numpy()

    pca_result = run_pca(
        feature_matrix=feature_matrix,
        variance_threshold=cfg.pca_variance_threshold,
        random_state=cfg.random_state,
    )

    selection_result = evaluate_gmm_candidates(
        feature_matrix=pca_result.transformed,
        k_min=cfg.gmm_k_min,
        k_max=cfg.gmm_k_max,
        random_state=cfg.random_state,
        n_init=cfg.gmm_n_init,
        covariance_type=cfg.gmm_covariance_type,
    )

    _, labels = fit_final_gmm(
        feature_matrix=pca_result.transformed,
        n_components=selection_result.optimal_k,
        random_state=cfg.random_state,
        n_init=cfg.gmm_n_init,
        covariance_type=cfg.gmm_covariance_type,
    )

    enriched_df = raw_df.copy()
    enriched_df[cfg.cluster_column] = labels

    evaluation_result = evaluate_phase_alignment(
        df=enriched_df,
        cluster_column=cfg.cluster_column,
        phase_column=cfg.phase_column,
    )

    preprocessing_result.missingness_report.to_csv(
        results_dir / "missingness_report.csv",
        index=False,
    )
    pca_result.variance_table.to_csv(results_dir / "pca_variance_table.csv", index=False)
    selection_result.scores.to_csv(results_dir / "gmm_aic_bic_scores.csv", index=False)
    evaluation_result.counts_table.to_csv(results_dir / "cluster_phase_counts.csv")
    evaluation_result.normalized_table.to_csv(results_dir / "cluster_phase_normalized.csv")
    enriched_df.to_csv(cfg.enriched_data_path, index=False)

    plot_cumulative_variance(
        variance_table=pca_result.variance_table,
        variance_threshold=cfg.pca_variance_threshold,
        output_path=figures_dir / "pca_cumulative_explained_variance.png",
    )
    plot_aic_bic(
        scores=selection_result.scores,
        output_path=figures_dir / "gmm_aic_bic_by_k.png",
    )
    plot_cluster_phase_heatmap(
        normalized_table=evaluation_result.normalized_table,
        output_path=figures_dir / "cluster_phase_heatmap.png",
    )

    selected_row = selection_result.scores[
        selection_result.scores["k"] == selection_result.optimal_k
    ].iloc[0]

    summary: dict[str, object] = {
        "selected_k": int(selection_result.optimal_k),
        "selected_k_aic": float(selected_row["aic"]),
        "selected_k_bic": float(selected_row["bic"]),
        "nmi": float(evaluation_result.nmi),
        "selected_pca_components": int(pca_result.selected_components),
        "pca_variance_threshold": float(cfg.pca_variance_threshold),
        "imputation_method": preprocessing_result.imputation_method,
        "global_missingness_ratio": float(preprocessing_result.global_missingness_ratio),
        "n_samples": int(len(enriched_df)),
        "n_physiological_features": int(len(preprocessing_result.feature_columns)),
        "aic_bic_by_k": selection_result.scores[["k", "aic", "bic"]].to_dict(
            orient="records"
        ),
    }

    with (results_dir / "metrics_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with (results_dir / "mapped_classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(evaluation_result.mapped_report)

    with (results_dir / "cluster_to_phase_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(evaluation_result.cluster_to_phase_map, file, indent=2)

    return summary
