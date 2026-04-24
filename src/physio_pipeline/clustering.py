from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def fit_final_gmm(
    feature_matrix: np.ndarray,
    n_components: int,
    random_state: int,
    n_init: int,
    covariance_type: str,
) -> tuple[GaussianMixture, np.ndarray]:
    """Fit the final GMM and predict cluster labels."""

    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        n_init=n_init,
        covariance_type=covariance_type,
    )
    labels = gmm.fit_predict(feature_matrix)
    return gmm, labels


def fit_final_kmeans(
    feature_matrix: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> tuple[KMeans, np.ndarray]:
    """Fit the final K-means model and predict cluster labels."""

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = kmeans.fit_predict(feature_matrix)
    return kmeans, labels
