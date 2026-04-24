from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the physiological clustering pipeline."""

    data_path: Path = Path("HR_data_2") / "HR_data_2.csv"
    output_dir: Path = Path("outputs")
    random_state: int = 42
    pca_variance_threshold: float = 0.90

    # Shared project/data settings
    missingness_iterative_threshold: float = 0.20
    group_column: str = "Individual"
    phase_column: str = "Phase"
    required_columns: tuple[str, ...] = ("Individual", "Phase", "Round")
    physiological_prefixes: tuple[str, ...] = ("HR_", "EDA_", "TEMP_")

    # GMM settings
    run_gmm: bool = True
    gmm_k_min: int = 1
    gmm_k_max: int = 8
    gmm_n_init: int = 10
    gmm_covariance_type: str = "full"
    gmm_cluster_column: str = "GMMCluster"

    # K-means settings
    run_kmeans: bool = True
    kmeans_k_min: int = 2
    kmeans_k_max: int = 8
    kmeans_n_init: int = 50
    kmeans_final_k: int | None = None
    kmeans_cluster_column: str = "KMeansCluster"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    @property
    def gmm_figures_dir(self) -> Path:
        return self.figures_dir / "gmm"

    @property
    def gmm_results_dir(self) -> Path:
        return self.results_dir / "gmm"

    @property
    def kmeans_figures_dir(self) -> Path:
        return self.figures_dir / "kmeans"

    @property
    def kmeans_results_dir(self) -> Path:
        return self.results_dir / "kmeans"

    @property
    def gmm_enriched_data_path(self) -> Path:
        return self.gmm_results_dir / "hr_data_with_gmm_clusters.csv"

    @property
    def kmeans_enriched_data_path(self) -> Path:
        return self.kmeans_results_dir / "hr_data_with_kmeans_clusters.csv"

    @property
    def enriched_data_path(self) -> Path:
        """Backward-compatible alias for older GMM-only scripts."""

        return self.gmm_enriched_data_path
