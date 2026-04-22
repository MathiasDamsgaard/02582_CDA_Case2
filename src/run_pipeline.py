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
        "--k-min",
        type=int,
        default=1,
        help="Minimum number of GMM components",
    )
    parser.add_argument(
        "--k-max",
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
        gmm_k_min=args.k_min,
        gmm_k_max=args.k_max,
        gmm_covariance_type=args.gmm_covariance_type,
        random_state=args.random_state,
    )

    summary = run_pipeline(config)

    print("Pipeline run completed.")
    print(f"Selected PCA components: {summary['selected_pca_components']}")
    print(f"Selected GMM k (BIC): {summary['selected_k']}")
    print(f"Selected model AIC: {summary['selected_k_aic']:.3f}")
    print(f"Selected model BIC: {summary['selected_k_bic']:.3f}")
    print(f"NMI (Phase vs Cluster): {summary['nmi']:.4f}")


if __name__ == "__main__":
    main()
