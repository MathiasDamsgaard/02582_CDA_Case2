from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_output_directories(output_root: Path) -> tuple[Path, Path]:
    """Create output folders for figures and tabular results."""

    figures_dir = output_root / "figures"
    results_dir = output_root / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, results_dir


def validate_required_columns(df: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    """Validate that all required columns exist before processing."""

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def load_dataset(csv_path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    """Load the dataset and drop unnamed index-like columns if present."""

    df = pd.read_csv(csv_path)
    drop_cols = [col for col in df.columns if col.startswith("Unnamed:") or col == ""]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    validate_required_columns(df, required_columns)
    return df
