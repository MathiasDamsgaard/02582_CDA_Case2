from __future__ import annotations

import pandas as pd

from physio_pipeline.preprocessing import select_physiological_features


def test_select_physiological_features_only_returns_hr_eda_temp_numeric() -> None:
    df = pd.DataFrame(
        {
            "HR_TD_Mean": [72.1, 73.4],
            "EDA_TD_P_Mean": [0.11, 0.15],
            "TEMP_TD_Mean": [31.2, 31.0],
            "Phase": ["phase1", "phase2"],
            "Frustrated": [1, 2],
            "EDA_note": ["a", "b"],
        }
    )

    features = select_physiological_features(df, prefixes=("HR_", "EDA_", "TEMP_"))

    assert features == ["HR_TD_Mean", "EDA_TD_P_Mean", "TEMP_TD_Mean"]
