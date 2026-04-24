from __future__ import annotations

import argparse
from pathlib import Path

from physio_pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the pipeline runner."""

    parser = argparse.ArgumentParser(description="Run physiological unsupervised pipeline")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("HR_data_2") / "HR_data_2.csv",
        help="Path to HR_data_2.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where artifacts will be written",
    )
    parser.add_argument(
        "--pca-variance-threshold",
        type=float,
        default=0.90,
        help="Cumulative explained variance target for PCA",
    )
    parser.add_argument(
        "--gmm-k-min",
        "--k-min",
        dest="gmm_k_min",
        type=int,
        default=1,
        help="Minimum number of GMM components",
    )
    parser.add_argument(
        "--gmm-k-max",
        "--k-max",
        dest="gmm_k_max",
        type=int,
        default=8,
        help="Maximum number of GMM components",
    )
    parser.add_argument(
        "--gmm-covariance-type",
        type=str,
        default="full",
        help="Covariance type for GMM - set to full or diag",
    )
    parser.add_argument(
        "--kmeans-k-min",
        type=int,
        default=2,
        help="Minimum number of K-means clusters",
    )
    parser.add_argument(
        "--kmeans-k-max",
        type=int,
        default=8,
        help="Maximum number of K-means clusters",
    )
    parser.add_argument(
        "--kmeans-final-k",
        type=int,
        default=None,
        help="Optional fixed K-means k. If omitted, the best silhouette score is used.",
    )
    parser.add_argument(
        "--skip-gmm",
        action="store_true",
        help="Skip the GMM branch.",
    )
    parser.add_argument(
        "--skip-kmeans",
        action="store_true",
        help="Skip the K-means branch.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    """Create config from arguments and execute the pipeline."""

    args = parse_args()

    config = PipelineConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        pca_variance_threshold=args.pca_variance_threshold,
        gmm_k_min=args.gmm_k_min,
        gmm_k_max=args.gmm_k_max,
        gmm_covariance_type=args.gmm_covariance_type,
        kmeans_k_min=args.kmeans_k_min,
        kmeans_k_max=args.kmeans_k_max,
        kmeans_final_k=args.kmeans_final_k,
        run_gmm=not args.skip_gmm,
        run_kmeans=not args.skip_kmeans,
        random_state=args.random_state,
    )

    summary = run_pipeline(config)

    print("Pipeline run completed.")
    print(f"Selected PCA components: {summary['selected_pca_components']}")

    model_summaries = summary.get("models", {})
    if "gmm" in model_summaries:
        gmm_summary = model_summaries["gmm"]
        print(f"Selected GMM k (BIC): {gmm_summary['selected_k']}")
        print(f"Selected GMM AIC: {gmm_summary['selected_k_aic']:.3f}")
        print(f"Selected GMM BIC: {gmm_summary['selected_k_bic']:.3f}")
        print(f"GMM NMI (Phase vs Cluster): {gmm_summary['nmi']:.4f}")

    if "kmeans" in model_summaries:
        kmeans_summary = model_summaries["kmeans"]
        print(f"Selected K-means k: {kmeans_summary['selected_k']}")
        if kmeans_summary["selected_k_silhouette"] is not None:
            print(f"Selected K-means silhouette: {kmeans_summary['selected_k_silhouette']:.4f}")
        print(f"K-means NMI (Phase vs Cluster): {kmeans_summary['nmi']:.4f}")


if __name__ == "__main__":
    main()
