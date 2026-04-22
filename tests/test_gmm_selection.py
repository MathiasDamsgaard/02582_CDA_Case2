from __future__ import annotations

import numpy as np

from physio_pipeline.model_selection import evaluate_gmm_candidates


def test_evaluate_gmm_candidates_selects_two_clusters() -> None:
    rng = np.random.default_rng(21)
    cluster_a = rng.normal(loc=-2.5, scale=0.3, size=(250, 2))
    cluster_b = rng.normal(loc=2.5, scale=0.3, size=(250, 2))
    x = np.vstack([cluster_a, cluster_b])

    result = evaluate_gmm_candidates(
        feature_matrix=x,
        k_min=1,
        k_max=4,
        random_state=42,
        n_init=5,
        covariance_type="full",
    )

    assert result.optimal_k == 2
    assert set(result.scores.columns) >= {"k", "aic", "bic"}
    assert len(result.scores) == 4
