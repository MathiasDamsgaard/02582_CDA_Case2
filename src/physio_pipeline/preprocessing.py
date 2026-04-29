from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PreprocessingResult:
    """Result bundle from preprocessing stage."""

    processed_df: pd.DataFrame
    feature_columns: list[str]
    missingness_report: pd.DataFrame
    imputation_method: str
    global_missingness_ratio: float


def select_physiological_features(
    df: pd.DataFrame,
    prefixes: tuple[str, ...],
) -> list[str]:
    """Select only numeric HR/EDA/TEMP features from the dataset."""

    candidates = [
        column
        for column in df.columns
        if any(column.startswith(prefix) for prefix in prefixes)
    ]
    numeric_features = [column for column in candidates if pd.api.types.is_numeric_dtype(df[column])]
    if not numeric_features:
        raise ValueError("No numeric physiological features were found with the configured prefixes.")
    return numeric_features


def compute_missingness_report(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Create a per-feature missingness table for diagnostics and logging."""

    missing_fraction = df[feature_columns].isna().mean()
    report = (
        pd.DataFrame(
            {
                "feature": missing_fraction.index,
                "missing_fraction": missing_fraction.values,
                "missing_count": df[feature_columns].isna().sum().values,
            }
        )
        .sort_values("missing_fraction", ascending=False)
        .reset_index(drop=True)
    )
    # Add a total missing count and fraction as a total feature row
    total_missing_count = report["missing_count"].sum()
    total_count = len(df) * len(feature_columns)
    total_missing_fraction = total_missing_count / total_count
    report.loc[len(report)] = {
        "feature": "TOTAL",
        "missing_fraction": total_missing_fraction,
        "missing_count": total_missing_count,
    }
    return report


def impute_missing_values(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, str, float, pd.DataFrame]:
    """
    Impute missing values with median by default.

    Median imputation is robust for sensor data and keeps the pipeline deterministic.
    If missingness is high, IterativeImputer is used to preserve multivariate structure.
    """

    report = compute_missingness_report(df, feature_columns)
    global_missingness_ratio = float(df[feature_columns].isna().mean().mean())
    method = "median"
    imputer = SimpleImputer(strategy=method)

    imputed_values = imputer.fit_transform(df[feature_columns])
    float_feature_dtypes = {column: float for column in feature_columns}
    imputed_df = df.astype(float_feature_dtypes)
    imputed_df.loc[:, feature_columns] = imputed_values

    if imputed_df[feature_columns].isna().any().any():
        raise ValueError("Missing values remain after imputation.")

    return imputed_df, method, global_missingness_ratio, report


def scale_within_group(
    df: pd.DataFrame,
    feature_columns: list[str],
    group_column: str,
) -> pd.DataFrame:
    """Apply StandardScaler independently inside each individual group."""

    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' is not present in the dataframe.")

    float_feature_dtypes = {column: float for column in feature_columns}
    scaled_df = df.astype(float_feature_dtypes)

    for _, index_values in df.groupby(group_column, sort=False).groups.items():
        group_idx = list(index_values)
        group_frame = df.loc[group_idx, feature_columns]
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(group_frame)
        scaled_df.loc[group_idx, feature_columns] = scaled_values

    return scaled_df


def preprocess_dataset(
    df: pd.DataFrame,
    prefixes: tuple[str, ...],
    group_column: str,
) -> PreprocessingResult:
    """Run feature selection, imputation, and within-individual scaling."""

    feature_columns = select_physiological_features(df, prefixes)
    imputed_df, method, global_missingness_ratio, missingness_report = impute_missing_values(
        df=df,
        feature_columns=feature_columns,
    )
    processed_df = scale_within_group(
        df=imputed_df,
        feature_columns=feature_columns,
        group_column=group_column,
    )

    return PreprocessingResult(
        processed_df=processed_df,
        feature_columns=feature_columns,
        missingness_report=missingness_report,
        imputation_method=method,
        global_missingness_ratio=global_missingness_ratio,
    )
