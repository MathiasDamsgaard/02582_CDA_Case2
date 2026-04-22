from __future__ import annotations

import numpy as np
import pandas as pd

from physio_pipeline.preprocessing import scale_within_group


def test_scale_within_group_centers_each_individual() -> None:
    df = pd.DataFrame(
        {
            "Individual": [1, 1, 2, 2],
            "HR_TD_Mean": [70.0, 80.0, 60.0, 100.0],
            "TEMP_TD_Mean": [31.0, 32.0, 29.0, 33.0],
        }
    )

    feature_columns = ["HR_TD_Mean", "TEMP_TD_Mean"]
    scaled = scale_within_group(df, feature_columns, "Individual")

    for individual, group in scaled.groupby("Individual"):
        _ = individual
        means = group[feature_columns].mean().to_numpy()
        stds = group[feature_columns].std(ddof=0).to_numpy()

        assert np.allclose(means, 0.0)
        assert np.allclose(stds, 1.0)
