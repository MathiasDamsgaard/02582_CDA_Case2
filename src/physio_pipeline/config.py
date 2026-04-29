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
    gmm_k_min: int = 1
    gmm_k_max: int = 8
    gmm_n_init: int = 10
    gmm_covariance_type: str = "full"
    group_column: str = "Individual"
    phase_column: str = "Phase"
    cluster_column: str = "Cluster"
    required_columns: tuple[str, ...] = ("Individual", "Phase", "Round")
    physiological_prefixes: tuple[str, ...] = ("HR_", "EDA_", "TEMP_")

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    @property
    def enriched_data_path(self) -> Path:
        return self.results_dir / "hr_data_with_clusters.csv"
