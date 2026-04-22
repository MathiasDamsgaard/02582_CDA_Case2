from __future__ import annotations

import numpy as np

from physio_pipeline.pca_stage import run_pca


def test_run_pca_reaches_variance_threshold() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(400, 1))
    noise = rng.normal(scale=0.05, size=(400, 3))
    x = np.hstack([base + noise[:, [0]], base + noise[:, [1]], base + noise[:, [2]], rng.normal(size=(400, 1))])

    result = run_pca(feature_matrix=x, variance_threshold=0.90, random_state=42)

    cumulative_at_selection = result.variance_table.loc[
        result.selected_components - 1, "cumulative_explained_variance"
    ]

    assert cumulative_at_selection >= 0.90
    assert result.transformed.shape[1] == result.selected_components
