# 02582 Case II - EmoPairCompete

This repository contains an unsupervised machine learning pipeline and a suite of exploratory notebooks for discovering latent physiological states from wearable data and evaluating how well those states align with experimental phases. 

It provides an end-to-end framework, exploring different algorithms (GMM vs. K-Means), determining the optimal number of states (AIC/BIC), and rigorously evaluating the generalizability of these states using Leave-One-Subject-Out Cross-Validation (LOSO CV).

## Data
The expected input is `HR_data_2/HR_data_2.csv` (or `data/HR_data_2.csv` depending on your local setup).

The dataset is license-restricted and therefore not stored in this repository.

## Environment Setup (uv + pyproject.toml)
Install `uv` (if needed):
https://docs.astral.sh/uv/getting-started/installation/

Then sync dependencies from `pyproject.toml`:
```bash
uv sync
```
*Note: Dependency management in this workflow uses the `pyproject.toml` file.*

To run the exploratory notebooks using the correct environment, you can launch Jupyter Lab directly through `uv`:
```bash
uv run jupyter lab
```

## Workflows

The repository is designed to be used in two phases: **Exploration & Experiments** (via Jupyter Notebooks) and the **Automated Pipeline** (via the CLI).

### Phase 1: Exploration & Experiments (Notebooks)
We recommend exploring the data and experimental setups in the `notebooks/` directory before running the final automated pipeline.

*   `exploratory_data_analysis.ipynb`: Performs initial data loading, missing value checks, and generates feature correlation heatmaps, box plots, and questionnaire response distributions.
*   `PCA_exploration.ipynb`: Deep dives into the dimensionality reduction step. It evaluates PCA loadings, cumulative explained variance thresholds (80% vs 90%), and generates scree plots (Kaiser criterion).
*   `exploratory_clustering_k.ipynb`: Mirrors the main pipeline but includes an exploratory switch (`USE_FIXED_K`) allowing you to force a fixed number of clusters (e.g., k=3) to override the BIC-selected number and visualize phase alignment heatmaps.
*   `loso_cv_individuals.ipynb`: The core generalization experiment. Implements Leave-One-Subject-Out (LOSO) Cross-Validation to evaluate if the unsupervised clustering states generalize to unseen individuals, measuring performance via Normalized Mutual Information (NMI) and Adjusted Rand Index (ARI).

### Phase 2: Automated Pipeline Overview (CLI)
The main pipeline (`src/run_pipeline.py`) automates the following steps for the full dataset:

1.  Load `HR_data_2.csv` with schema checks.
2.  Select only physiological features with `HR_`, `EDA_`, and `TEMP_` prefixes.
3.  Impute missing values (Default: median imputation; Fallback: IterativeImputer).
4.  Scale physiological features within each Individual group using StandardScaler.
5.  Run PCA and choose the minimum number of components that explain a defined variance threshold (default: 80% or 90%).
6.  Evaluate **GaussianMixture (GMM)** models for candidate k values (1 to 8) and report both AIC and BIC.
7.  Select final k using the minimum BIC.
8.  Fit final GMM and a comparative **K-Means** model using the selected k, attaching cluster labels to the original dataframe.
9.  Evaluate phase alignment for both models using cross-tabulation, heatmaps, and NMI.
10. Save all tables, figures, and summary metrics.

## Run the Automated Pipeline
From the repository root:
```bash
uv run python src/run_pipeline.py
```

Optional arguments:
*   `--data-path`
*   `--output-dir`
*   `--pca-variance-threshold`
*   `--k-min`
*   `--k-max`
*   `--random-state`

Example:
```bash
uv run python src/run_pipeline.py --data-path HR_data_2/HR_data_2.csv --pca-variance-threshold 0.90 --k-min 1 --k-max 8
```

## Project Structure
*   `src/physio_pipeline/`: Core modules for the automated pipeline (`config.py`, `data_io.py`, `preprocessing.py`, `pca_stage.py`, `model_selection.py`, `clustering.py`, `evaluation.py`, `visualization.py`, `pipeline.py`).
*   `src/run_pipeline.py`: Command-line entrypoint.
*   `notebooks/`: Jupyter notebooks used for EDA, PCA exploration, fixed-k clustering, and LOSO cross-validation.

## Generated Artifacts

Because the repository encompasses both an automated pipeline and deep-dive experimental notebooks, artifacts are distributed into specific output directories based on the workflow you execute.

### 1. Automated Pipeline Outputs (`outputs/`)
**General Metrics (`outputs/results/`):**
*   `missingness_report.csv`
*   `pca_variance_table.csv`

**Model-Specific Metrics (`outputs/results/[gmm|kmeans]/`):**
*   `gmm_aic_bic_scores.csv` *(GMM only)*
*   `kmeans_model_selection_scores.csv` *(K-Means only)*
*   `cluster_phase_counts.csv` & `cluster_phase_normalized.csv`
*   `hr_data_with_[gmm|kmeans]_clusters.csv`
*   `metrics_summary.json`
*   `mapped_classification_report.txt`
*   `cluster_to_phase_mapping.json`

**Figures (`outputs/figures/`):**
*   `pca_cumulative_explained_variance.png`
*   `gmm_aic_bic_by_k.png`
*   `[gmm|kmeans]/cluster_phase_heatmap.png`

### 2. Exploratory Data Analysis Outputs (`outputs/eda/`)
*Generated by `exploratory_data_analysis.ipynb`*
*   `mean physiological features.png` & `box plot of physiological features.png`
*   `correlation heatmap physiological features.png`
*   `sanity_check_of_Questionnaire_features.png` & `frequency_distribution_of_Questionnaire_features.png`

### 3. PCA Exploration Outputs (`outputs_jawhara_exploratory_gmm/`)
*Generated by `PCA_exploration.ipynb`*
*   `pca_scree_plot.png` & `pca_scores_by_phase.png` 
*   `pca_scores_by_cluster.png` & `pca_scores_by_individual.png`
*   Top loadings bar charts (`top_loadings_pc1.png`, `top_loadings_pc2.png`)

### 4. LOSO Cross-Validation Outputs (`outputs_loso_<threshold>/`)
*Generated by `loso_cv_individuals.ipynb` (subdirectories structured by `auto` or `fixed_k` runs)*
*   `results/loso_fold_results.csv`: Fold-by-fold NMI, ARI, and selected k.
*   `results/loso_gmm_candidate_scores.csv`: AIC/BIC scores per fold.
*   `results/loso_test_predictions.csv`: Out-of-sample predictions for every subject.
*   `results/loso_summary.json`: Aggregated global CV metrics.
*   `figures/loso_selected_k_counts.png`, `loso_test_nmi_by_fold.png`, `loso_mean_aic_bic_by_k.png`
