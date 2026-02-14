# Explanation Multiplicity in SHAP: Characterization and Assessment

**Authors:** Hyunseung Hwang, Seungeun Lee, Lucas Rosenblatt, Steven Euijong Whang, and Julia Stoyanovich

**Paper:** [Available on arXiv]([https://arxiv.org/abs/XXXX.XXXXX](https://arxiv.org/abs/2601.12654))

---

## Overview

We conceptualize and name the phenomenon of **explanation multiplicity**: the existence of multiple, internally valid but substantively different explanations for the same individual prediction, even when the dataset, prediction task, model class, and trained model are held fixed.

This repository provides the full experimental pipeline for characterizing and assessing explanation multiplicity in SHAP-based feature attribution methods.

### Key Contributions

1. **Explanation multiplicity as a first-class concept**: We frame the phenomenon as a distinct object of study, separating it from model multiplicity and explanation robustness.

2. **Comprehensive methodology**: A dual-seed protocol that disentangles model-induced and explainer-induced sources of multiplicity.

3. **Metrics and baselines**: A hierarchy of disagreement metrics (ℓ₂, Top-k Jaccard, RBO) with randomized null models for calibrated interpretation.

4. **Empirical evidence**: Extensive evaluation across three datasets, six model classes, and multiple confidence regimes.

---

## Datasets

| Dataset | N | d | Task |
|---------|---|---|------|
| **German Credit** | 988 | 16 | Credit risk classification |
| **Diabetes** | 392 | 8 | Diabetes onset prediction |
| **ACS Income** | 45,960 | 8 | Income > $50K prediction |

---

## Repository Structure

```
├── environment.yaml        # Conda environment specification
├── experiment.py           # SHAP experiment pipeline
├── models.py               # Model definitions (DT, RF, XGB, FTT, MLP, TabPFN)
├── utils.py                # Utility functions (seeding, data loading)
├── run.sh                  # Bash script to run all experiments
├── baseline.py             # Randomized baseline computation
├── plot_violin.py          # Analysis and visualization module
├── Plots_Violin.ipynb      # Notebook for generating all figures
└── dataset/
    ├── acs_X.pkl, acs_Y.pkl
    ├── german_X.pkl, german_Y.pkl
    └── diabetes_X.pkl, diabetes_Y.pkl
```

---

## Getting Started

### 1. Environment Setup

```bash
conda env create -f environment.yaml
conda activate shap
```

### 2. Run Experiments

You can either download pre-computed results or run experiments from scratch.

**Option A: Download pre-computed results (recommended)**

Download the `results/` folder from [Google Drive](https://drive.google.com/drive/folders/1KCE6P0potd4uME4hXwMy7RKvVrZXRI73?usp=drive_link) and place it in the repository root:

```
├── results/
│   ├── acs_dt_0_0_0_sv.pkl
│   ├── acs_dt_0_0_1_sv.pkl
│   └── ...
├── experiment.py
├── ...
```

**Option B: Run experiments from scratch**

This runs all dataset × model × seed combinations (3 datasets × 6 models × 5 split seeds × 5 model seeds × 5 explainer seeds × 15 chunks = 33,750 jobs). This requires a GPU and may take a significant amount of time.

```bash
bash run.sh
```

Completed experiments are automatically skipped on re-run, so the script is safe to interrupt and resume.

### 3. Generate Figures

Once the `results/` folder is ready (via download or experiment), open and run the notebook:

```bash
jupyter notebook Plots_Violin.ipynb
```

Inside the notebook, set the dataset in Cell 1:

```python
DATASET = 'german'  # Options: 'acs', 'diabetes', 'german'
```

Then run all cells sequentially. The notebook will:
- Load SHAP results from `results/`
- Compute pairwise instability metrics (ℓ₂, Jaccard, RBO)
- Generate all figures into the `figures/` directory

Repeat for each dataset by changing the `DATASET` variable.

---

## Citation

```bibtex
@article{hwang2025multiplicity,
  title   = {Explanation Multiplicity in SHAP: Characterization and Assessment},
  author  = {Hwang, Hyunseung and Lee, Seungeun and Rosenblatt, Lucas and Whang, Steven Euijong and Stoyanovich, Julia},
  year    = {2025}
}
```
