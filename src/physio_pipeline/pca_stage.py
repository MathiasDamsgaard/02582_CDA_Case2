from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PCAResult:
    """Outputs from PCA fitting and transformation."""

    transformed: np.ndarray
    variance_table: pd.DataFrame
    selected_components: int


def run_pca(
    feature_matrix: np.ndarray,
    variance_threshold: float,
    random_state: int,
) -> PCAResult:
    """Fit PCA and choose the smallest number of components reaching the threshold."""

    pca_full = PCA(random_state=random_state)
    pca_full.fit(feature_matrix)

    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    selected_components = int(np.searchsorted(cumulative, variance_threshold) + 1)

    pca_selected = PCA(n_components=selected_components, random_state=random_state)
    transformed = pca_selected.fit_transform(feature_matrix)

    variance_table = pd.DataFrame(
        {
            "component": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": cumulative,
        }
    )

    return PCAResult(
        transformed=transformed,
        variance_table=variance_table,
        selected_components=selected_components,
    )
