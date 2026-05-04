from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
    
from .clustering import fit_final_gmm, fit_final_kmeans
from .config import PipelineConfig
from .data_io import ensure_output_directories, load_dataset
from .evaluation import EvaluationResult, evaluate_phase_alignment
from .model_selection import (
    ModelSelectionResult,
    evaluate_gmm_candidates,
    evaluate_kmeans_candidates,
    evaluate_kmeans_gap_statistic,
    select_optimal_k_gap,
)
from .pca_stage import PCAResult, run_pca
from .preprocessing import PreprocessingResult, preprocess_dataset
from .visualization import (
    plot_aic_bic,
    plot_cluster_phase_heatmap,
    plot_cumulative_variance,
    plot_gap_statistic,
    plot_kmeans_elbow,
    plot_kmeans_silhouette,
)


def _make_branch_dirs(results_root: Path, figures_root: Path, branch_name: str) -> tuple[Path, Path]:
    """Create result and figure directories for a model branch."""

    branch_results_dir = results_root / branch_name
    branch_figures_dir = figures_root / branch_name
    branch_results_dir.mkdir(parents=True, exist_ok=True)
    branch_figures_dir.mkdir(parents=True, exist_ok=True)
    return branch_results_dir, branch_figures_dir


def _write_alignment_outputs(
    evaluation_result: EvaluationResult,
    output_dir: Path,
) -> None:
    """Persist cluster-to-phase alignment tables and mapping artifacts."""

    evaluation_result.counts_table.to_csv(output_dir / "cluster_phase_counts.csv")
    evaluation_result.normalized_table.to_csv(output_dir / "cluster_phase_normalized.csv")

    with (output_dir / "mapped_classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(evaluation_result.mapped_report)

    with (output_dir / "cluster_to_phase_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(evaluation_result.cluster_to_phase_map, file, indent=2)


def _run_gmm_branch(
    cfg: PipelineConfig,
    raw_df: pd.DataFrame,
    pca_result: PCAResult,
    preprocessing_result: PreprocessingResult,
    results_root: Path,
    figures_root: Path,
) -> dict[str, object]:
    """Run model selection, final fitting, evaluation, and saving for GMM."""

    gmm_results_dir, gmm_figures_dir = _make_branch_dirs(results_root, figures_root, "gmm")

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
    enriched_df[cfg.gmm_cluster_column] = labels

    evaluation_result = evaluate_phase_alignment(
        df=enriched_df,
        cluster_column=cfg.gmm_cluster_column,
        phase_column=cfg.phase_column,
    )

    selection_result.scores.to_csv(gmm_results_dir / "gmm_aic_bic_scores.csv", index=False)
    enriched_df.to_csv(cfg.gmm_enriched_data_path, index=False)
    _write_alignment_outputs(evaluation_result, gmm_results_dir)

    plot_aic_bic(
        scores=selection_result.scores,
        output_path=gmm_figures_dir / "gmm_aic_bic_by_k.png",
    )
    plot_cluster_phase_heatmap(
        normalized_table=evaluation_result.normalized_table,
        output_path=gmm_figures_dir / "cluster_phase_heatmap.png",
        title="GMM cluster to Phase alignment",
    )

    selected_row = selection_result.scores[
        selection_result.scores["k"] == selection_result.optimal_k
    ].iloc[0]

    summary: dict[str, object] = {
        "model": "gmm",
        "selected_k": int(selection_result.optimal_k),
        "selection_criterion": "lowest_bic",
        "selected_k_aic": float(selected_row["aic"]),
        "selected_k_bic": float(selected_row["bic"]),
        "nmi": float(evaluation_result.nmi),
        "selected_pca_components": int(pca_result.selected_components),
        "pca_variance_threshold": float(cfg.pca_variance_threshold),
        "imputation_method": preprocessing_result.imputation_method,
        "global_missingness_ratio": float(preprocessing_result.global_missingness_ratio),
        "n_samples": int(len(enriched_df)),
        "n_physiological_features": int(len(preprocessing_result.feature_columns)),
        "aic_bic_by_k": selection_result.scores[["k", "aic", "bic"]].to_dict(orient="records"),
    }

    with (gmm_results_dir / "metrics_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def _run_kmeans_branch(
    cfg: PipelineConfig,
    raw_df: pd.DataFrame,
    pca_result: PCAResult,
    preprocessing_result: PreprocessingResult,
    results_root: Path,
    figures_root: Path,
) -> dict[str, object]:
    """Run model selection, final fitting, evaluation, and saving for K-means."""

    kmeans_results_dir, kmeans_figures_dir = _make_branch_dirs(results_root, figures_root, "kmeans")

    selection_result: ModelSelectionResult = evaluate_kmeans_candidates(
        feature_matrix=pca_result.transformed,
        k_min=cfg.kmeans_k_min,
        k_max=cfg.kmeans_k_max,
        random_state=cfg.random_state,
        n_init=cfg.kmeans_n_init,
    )
    
    gap_df = evaluate_kmeans_gap_statistic(
        feature_matrix=pca_result.transformed,
        k_min=cfg.kmeans_k_min,
        k_max=cfg.kmeans_k_max,
        random_state=cfg.random_state,
        n_init=cfg.kmeans_n_init,
        n_reference_datasets=10,
    )
    gap_optimal_k = select_optimal_k_gap(gap_df)
    
    selected_k = int(cfg.kmeans_final_k or gap_optimal_k)

    _, labels = fit_final_kmeans(
        feature_matrix=pca_result.transformed,
        n_clusters=selected_k,
        random_state=cfg.random_state,
        n_init=cfg.kmeans_n_init,
    )

    enriched_df = raw_df.copy()
    enriched_df[cfg.kmeans_cluster_column] = labels

    evaluation_result = evaluate_phase_alignment(
        df=enriched_df,
        cluster_column=cfg.kmeans_cluster_column,
        phase_column=cfg.phase_column,
    )

    selection_result.scores.to_csv(
        kmeans_results_dir / "kmeans_model_selection_scores.csv",
        index=False,
    )
    gap_df.to_csv(
        kmeans_results_dir / "kmeans_gap_scores.csv",
        index=False,
    )
    enriched_df.to_csv(cfg.kmeans_enriched_data_path, index=False)
    _write_alignment_outputs(evaluation_result, kmeans_results_dir)

    plot_kmeans_elbow(
        scores=selection_result.scores,
        output_path=kmeans_figures_dir / "kmeans_elbow_inertia_by_k.png",
    )
    plot_kmeans_silhouette(
        scores=selection_result.scores,
        output_path=kmeans_figures_dir / "kmeans_silhouette_by_k.png",
    )
    plot_gap_statistic(
        gap_df=gap_df,
        output_path=kmeans_figures_dir / "kmeans_gap_statistic_by_k.png",
    )
    plot_cluster_phase_heatmap(
        normalized_table=evaluation_result.normalized_table,
        output_path=kmeans_figures_dir / "cluster_phase_heatmap.png",
        title="K-means cluster to Phase alignment",
    )

    selected_row = selection_result.scores[selection_result.scores["k"] == selected_k]
    selected_silhouette = None
    selected_inertia = None
    if not selected_row.empty:
        selected_silhouette = float(selected_row.iloc[0]["silhouette"])
        selected_inertia = float(selected_row.iloc[0]["inertia"])

    selected_gap_row = gap_df[gap_df["k"] == selected_k]
    selected_gap = None
    if not selected_gap_row.empty:
        selected_gap = float(selected_gap_row.iloc[0]["gap"])

    summary: dict[str, object] = {
        "model": "kmeans",
        "selected_k": selected_k,
        "selection_criterion": "gap_statistic" if cfg.kmeans_final_k is None else "manual_k",
        "selected_k_silhouette": selected_silhouette,
        "selected_k_inertia": selected_inertia,
        "selected_k_gap": selected_gap,
        "nmi": float(evaluation_result.nmi),
        "selected_pca_components": int(pca_result.selected_components),
        "pca_variance_threshold": float(cfg.pca_variance_threshold),
        "imputation_method": preprocessing_result.imputation_method,
        "global_missingness_ratio": float(preprocessing_result.global_missingness_ratio),
        "n_samples": int(len(enriched_df)),
        "n_physiological_features": int(len(preprocessing_result.feature_columns)),
        "kmeans_scores_by_k": selection_result.scores[["k", "inertia", "silhouette"]].to_dict(
            orient="records"
        ),
        "kmeans_gap_scores_by_k": gap_df.to_dict(orient="records"),
    }

    with (kmeans_results_dir / "metrics_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


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

    preprocessing_result.missingness_report.to_csv(
        results_dir / "missingness_report.csv",
        index=False,
    )
    pca_result.variance_table.to_csv(results_dir / "pca_variance_table.csv", index=False)

    plot_cumulative_variance(
        variance_table=pca_result.variance_table,
        variance_threshold=cfg.pca_variance_threshold,
        output_path=figures_dir / "pca_cumulative_explained_variance.png",
    )

    summary: dict[str, object] = {
        "selected_pca_components": int(pca_result.selected_components),
        "pca_variance_threshold": float(cfg.pca_variance_threshold),
        "imputation_method": preprocessing_result.imputation_method,
        "global_missingness_ratio": float(preprocessing_result.global_missingness_ratio),
        "n_samples": int(len(raw_df)),
        "n_physiological_features": int(len(preprocessing_result.feature_columns)),
        "models": {},
    }

    if cfg.run_gmm:
        summary["models"]["gmm"] = _run_gmm_branch(
            cfg=cfg,
            raw_df=raw_df,
            pca_result=pca_result,
            preprocessing_result=preprocessing_result,
            results_root=results_dir,
            figures_root=figures_dir,
        )

    if cfg.run_kmeans:
        summary["models"]["kmeans"] = _run_kmeans_branch(
            cfg=cfg,
            raw_df=raw_df,
            pca_result=pca_result,
            preprocessing_result=preprocessing_result,
            results_root=results_dir,
            figures_root=figures_dir,
        )

    with (results_dir / "metrics_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary
