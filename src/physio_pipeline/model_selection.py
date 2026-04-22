from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class ModelSelectionResult:
    """AIC/BIC tracking and chosen cluster count."""

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
