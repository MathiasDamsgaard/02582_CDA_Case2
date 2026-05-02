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

def evaluate_kmeans_gap_statistic(
    feature_matrix: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    n_init: int,
    n_reference_datasets: int = 10,
) -> pd.DataFrame:
    """
    Evaluate K-Means candidates using the Gap Statistic method.
    
    Fits K-Means to the real dataset and generated reference datasets to compute the Gap Statistic, 
    which compares the change in within-cluster dispersion to that expected under a reference null distribution.
    """
    if k_min < 1:
        raise ValueError("K-means model selection requires k_min >= 1.")
    if k_max < k_min:
        raise ValueError("k_max must be greater than or equal to k_min.")

    rng = np.random.default_rng(random_state)
    
    rows: list[dict[str, float]] = []
    
    for k in range(k_min, k_max + 1):
        if k >= feature_matrix.shape[0]:
            break
            
        # Fit to real data
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        kmeans.fit(feature_matrix)
        log_W_k = np.log(kmeans.inertia_)
        
        # Fit to reference datasets
        reference_log_W_k: list[float] = []
        for _ in range(n_reference_datasets):
            reference_data = rng.uniform(
                low=np.min(feature_matrix, axis=0),
                high=np.max(feature_matrix, axis=0),
                size=feature_matrix.shape
            )
            kmeans_ref = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            kmeans_ref.fit(reference_data)
            reference_log_W_k.append(np.log(kmeans_ref.inertia_))
            
        expected_log_W_k = np.mean(reference_log_W_k)
        sd_k = np.std(reference_log_W_k, ddof=0)
        gap = expected_log_W_k - log_W_k
        s_k = sd_k * np.sqrt(1 + 1 / n_reference_datasets)
        
        rows.append({
            "k": k,
            "log_W_k": float(log_W_k),
            "expected_log_W_k": float(expected_log_W_k),
            "gap": float(gap),
            "s_k": float(s_k),
        })

    scores = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    scores["k"] = scores["k"].astype(int)
    return scores

def select_optimal_k_gap(gap_df: pd.DataFrame) -> int:
    """
    Select the optimal k using the standard Tibshirani selection rule for the Gap Statistic.
    
    Rule: Choose smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}.
    If the condition is never met, falls back to the k with the maximum gap.
    """
    if gap_df.empty:
        raise ValueError("The provided Gap DataFrame is empty.")
        
    for i in range(len(gap_df) - 1):
        gap_k = gap_df.loc[i, "gap"]
        gap_k1 = gap_df.loc[i + 1, "gap"]
        s_k1 = gap_df.loc[i + 1, "s_k"]
        if gap_k >= gap_k1 - s_k1:
            return int(gap_df.loc[i, "k"])
            
    # Fallback to max gap
    optimal_idx = gap_df["gap"].idxmax()
    return int(gap_df.loc[optimal_idx, "k"])
