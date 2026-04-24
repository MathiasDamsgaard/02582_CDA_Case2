from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class ModelSelectionResult:
    """Model-selection scores and chosen cluster count."""

    scores: pd.DataFrame
    optimal_k: int


def evaluate_gmm_candidates(
    feature_matrix: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    n_init: int,
    covariance_type: str,
) -> ModelSelectionResult:
    """Fit GMM models over a cluster range and collect AIC/BIC."""

    rows: list[dict[str, float]] = []

    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            random_state=random_state,
            n_init=n_init,
            covariance_type=covariance_type,
        )
        gmm.fit(feature_matrix)
        rows.append(
            {
                "k": float(k),
                "aic": float(gmm.aic(feature_matrix)),
                "bic": float(gmm.bic(feature_matrix)),
                "log_likelihood": float(gmm.score(feature_matrix)),
            }
        )

    scores = pd.DataFrame(rows)
    scores = scores.sort_values("k").reset_index(drop=True)
    scores["k"] = scores["k"].astype(int)

    optimal_k = int(scores.sort_values(["bic", "k"], ascending=[True, True]).iloc[0]["k"])

    return ModelSelectionResult(scores=scores, optimal_k=optimal_k)


def evaluate_kmeans_candidates(
    feature_matrix: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    n_init: int,
) -> ModelSelectionResult:
    """
    Fit K-means models over a cluster range and collect inertia/silhouette.

    The selected k is the candidate with the highest silhouette score. Ties are
    resolved by the smaller k to prefer the simpler clustering.
    """

    if k_min < 2:
        raise ValueError("K-means model selection requires k_min >= 2 for silhouette scoring.")
    if k_max < k_min:
        raise ValueError("k_max must be greater than or equal to k_min.")
    if feature_matrix.shape[0] <= k_min:
        raise ValueError("Not enough samples to evaluate the requested K-means cluster range.")

    rows: list[dict[str, float]] = []

    for k in range(k_min, k_max + 1):
        if k >= feature_matrix.shape[0]:
            break

        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
        )
        labels = kmeans.fit_predict(feature_matrix)
        rows.append(
            {
                "k": float(k),
                "inertia": float(kmeans.inertia_),
                "silhouette": float(silhouette_score(feature_matrix, labels)),
            }
        )

    if not rows:
        raise ValueError("No valid K-means candidates were evaluated.")

    scores = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    scores["k"] = scores["k"].astype(int)

    optimal_k = int(
        scores.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"]
    )

    return ModelSelectionResult(scores=scores, optimal_k=optimal_k)
