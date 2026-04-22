from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import classification_report, normalized_mutual_info_score


@dataclass(frozen=True)
class EvaluationResult:
    """Phase-alignment evaluation outputs."""

    counts_table: pd.DataFrame
    normalized_table: pd.DataFrame
    nmi: float
    mapped_report: str
    cluster_to_phase_map: dict[str, str]


def build_cluster_phase_tables(
    df: pd.DataFrame,
    cluster_column: str,
    phase_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build raw and row-normalized contingency tables."""

    counts = pd.crosstab(df[cluster_column], df[phase_column], dropna=False)
    normalized = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    return counts, normalized


def build_mapped_classification_report(
    phases: pd.Series,
    clusters: pd.Series,
) -> tuple[str, dict[str, str]]:
    """
    Build a post-hoc mapped classification report.

    The mapping is computed with Hungarian matching and falls back to per-cluster
    majority mapping for unmatched clusters.
    """

    phases_str = phases.astype(str)
    clusters_str = clusters.astype(str)
    contingency = pd.crosstab(clusters_str, phases_str, dropna=False)

    if contingency.empty:
        return "No data available for mapped classification report.", {}

    cost_matrix = contingency.to_numpy().max() - contingency.to_numpy()
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    mapping: dict[str, str] = {
        str(contingency.index[row]): str(contingency.columns[col])
        for row, col in zip(row_idx, col_idx)
    }

    for cluster in contingency.index.astype(str):
        if cluster not in mapping:
            mapping[cluster] = str(contingency.loc[cluster].idxmax())

    mapped_predictions = clusters_str.map(mapping)
    report = classification_report(phases_str, mapped_predictions, zero_division=0)
    return report, mapping


def evaluate_phase_alignment(
    df: pd.DataFrame,
    cluster_column: str,
    phase_column: str,
) -> EvaluationResult:
    """Compute contingency tables, NMI, and optional mapped-label report."""

    counts, normalized = build_cluster_phase_tables(
        df=df,
        cluster_column=cluster_column,
        phase_column=phase_column,
    )

    nmi = float(
        normalized_mutual_info_score(
            df[phase_column].astype(str),
            df[cluster_column].astype(str),
        )
    )

    mapped_report, cluster_to_phase_map = build_mapped_classification_report(
        phases=df[phase_column],
        clusters=df[cluster_column],
    )

    return EvaluationResult(
        counts_table=counts,
        normalized_table=normalized,
        nmi=nmi,
        mapped_report=mapped_report,
        cluster_to_phase_map=cluster_to_phase_map,
    )
