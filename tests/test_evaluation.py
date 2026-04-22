from __future__ import annotations

from physio_pipeline.evaluation import evaluate_phase_alignment


def test_evaluate_phase_alignment_outputs_tables_and_nmi() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "Phase": ["phase1", "phase1", "phase2", "phase2", "phase3", "phase3"],
            "Cluster": [0, 0, 1, 1, 2, 2],
        }
    )

    result = evaluate_phase_alignment(df, cluster_column="Cluster", phase_column="Phase")

    assert result.counts_table.shape == (3, 3)
    assert result.normalized_table.shape == (3, 3)
    assert 0.0 <= result.nmi <= 1.0
    assert "precision" in result.mapped_report
