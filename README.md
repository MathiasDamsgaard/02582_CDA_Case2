# 02582 Case II - EmoPairCompete

This repository contains an unsupervised machine learning pipeline for discovering latent physiological states from wearable data and evaluating how well those states align with experimental phases.

## Data
The expected input is HR_data_2/HR_data_2.csv.

The dataset is license-restricted and therefore not stored in this repository.

## Pipeline Overview
The implementation executes the following workflow:

1. Load HR_data_2.csv with schema checks.
2. Select only physiological features with HR_, EDA_, and TEMP_ prefixes.
3. Impute missing values:
	- Default: median imputation.
	- Fallback: IterativeImputer if global missingness is above threshold.
4. Scale physiological features within each Individual group using StandardScaler.
5. Run PCA and choose the minimum number of components that explain at least 90% variance.
6. Evaluate GaussianMixture models for k = 1..8 and report both AIC and BIC.
7. Select final k using the minimum BIC.
8. Fit final GMM, attach cluster labels to the original dataframe.
9. Evaluate phase alignment using cross-tab, heatmap, and NMI.
10. Save all tables, figures, and summary metrics to outputs.

## Project Structure
- src/physio_pipeline/config.py: configuration dataclass.
- src/physio_pipeline/data_io.py: loading, schema validation, output directory management.
- src/physio_pipeline/preprocessing.py: feature selection, missing-value handling, within-individual scaling.
- src/physio_pipeline/pca_stage.py: PCA fitting and component selection.
- src/physio_pipeline/model_selection.py: GMM candidate fitting with AIC/BIC tracking.
- src/physio_pipeline/clustering.py: final GMM fit and predictions.
- src/physio_pipeline/evaluation.py: cross-tab, NMI, and mapped classification report.
- src/physio_pipeline/visualization.py: PCA, AIC/BIC, and heatmap plotting.
- src/physio_pipeline/pipeline.py: end-to-end orchestration.
- src/run_pipeline.py: command-line entrypoint.
- tests/: unit tests for key logic.

## Environment Setup (uv + pyproject.toml)
Install uv (if needed):
https://docs.astral.sh/uv/getting-started/installation/

Then sync dependencies from pyproject.toml:

uv sync

Note: dependency management in this workflow is pyproject.toml-first. requirements.txt is intentionally not part of the active dependency update path.

## Run the Pipeline
From repository root:

uv run python src/run_pipeline.py

Optional arguments:
- --data-path
- --output-dir
- --pca-variance-threshold
- --k-min
- --k-max
- --random-state

Example:

uv run python src/run_pipeline.py --data-path HR_data_2/HR_data_2.csv --pca-variance-threshold 0.90 --k-min 1 --k-max 8

## Run Tests
uv run pytest -q

## Generated Artifacts
Tables and metrics are written to outputs/results:
- missingness_report.csv
- pca_variance_table.csv
- gmm_aic_bic_scores.csv
- cluster_phase_counts.csv
- cluster_phase_normalized.csv
- hr_data_with_clusters.csv
- metrics_summary.json
- mapped_classification_report.txt
- cluster_to_phase_mapping.json

Figures are written to outputs/figures:
- pca_cumulative_explained_variance.png
- gmm_aic_bic_by_k.png
- cluster_phase_heatmap.png