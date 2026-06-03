# Lipophilicity (logD) Prediction — Machine Learning for Drug Design

**Viktória Kubišová, Alexandra Biddiscombe, Louie Daans · SUPSI (Exchange) · 2026**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green?style=flat)
![XGBoost](https://img.shields.io/badge/XGBoost-189B2D?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Drug Discovery](https://img.shields.io/badge/Domain-Drug%20Discovery-blueviolet?style=flat)

---

## Overview

This project was completed as part of the **Machine Learning for Drug Design** course during my exchange semester at SUPSI (University of Applied Sciences and Arts of Southern Switzerland). Working in a group of three, we investigated the prediction of lipophilicity — measured as the octanol/water distribution coefficient logD at pH 7.4 — using the AstraZeneca dataset from the Therapeutics Data Commons (TDC) benchmark suite, applying and comparing a range of classical ML and graph-based approaches across multiple molecular representations and splitting strategies.

Lipophilicity is one of the most critical physicochemical properties in drug design, directly influencing a drug's absorption, distribution, metabolism, and excretion (ADME) profile.

**My contributions:** I built the full classical ML pipeline (data cleaning, EDA, molecular representations, feature selection, model training, evaluation, applicability domain, and explainability) and wrote the methodology and results sections of the scientific report. My teammate Alexandra led the literature research and scientific writing, and Louie implemented the Graph Neural Network models.

---

## Research Question

> *"Which molecular representations and machine learning models best predict lipophilicity (logD at pH 7.4) for drug-like molecules, and how does model performance and reliability vary with structural similarity to the training set?"*

---

## Dataset

**AstraZeneca Lipophilicity dataset** via [Therapeutics Data Commons](https://tdcommons.ai/single_pred_tasks/adme/#lipophilicity-astrazeneca)

| Property | Detail |
|----------|--------|
| Molecules | 4,200 drug-like compounds |
| Target | logD at pH 7.4 (continuous regression) |
| Target range | −1.5 to 4.5 |
| Split ratio | 70% train / 10% validation / 20% test |
| Splitting strategies | Random (optimistic baseline) + Scaffold (primary evaluation) |

---

## Approach

| Step | Description |
|------|-------------|
| 01 Data cleaning | SMILES validity check, canonicalization, deduplication |
| 02 EDA | logD distribution, Lipinski Rule of Five descriptors, logP vs logD correlation (Spearman ρ = 0.394) |
| 03 Similarity analysis | Nearest-neighbour Tanimoto similarity (Morgan FP) per split to quantify evaluation difficulty |
| 04 Chemical space | UMAP visualization of full dataset colored by logD and split assignment |
| 05 Molecular representations | MACCS Keys (166 bits), Morgan/ECFP4 (1024 bits), RDKit FP (2048 bits), extended physicochemical descriptors (~100 features) |
| 06 Baseline | Linear Regression on all features — revealed RDKit FP failure (R² = −0.860) due to high feature density |
| 07 Feature selection | Clustered Forward Selection (CFS) with Spearman correlation threshold 0.7; applied separately per model on training set only |
| 08 Full model comparison | 24 combinations: 3 models × 4 encodings × 2 splits; hyperparameter tuning on validation set |
| 09 Learning curves | MAE vs training set size for best model; confirmed data-limited regime |
| 10 Applicability domain | Empirically determined threshold (nn_sim = 0.5) from performance-vs-similarity analysis |
| 11 Explainability | Permutation Feature Importance (PFI) on best model |
| 12 Bonus — KRR | Kernel Ridge Regression with Tanimoto and Gaussian kernels vs best classical model |
| 13 GNN models *(Louie)* | GCN from scratch, PyG GCN, PyG GIN — both random and scaffold splits |

**Pipeline:**

```
SMILES → Cleaning & EDA → Similarity analysis → UMAP
→ 4 molecular representations → Baseline LR → CFS feature selection
→ 24 model combinations (LR / RF / XGB × MACCS / Morgan / RDKit / Physchem × Random / Scaffold)
→ Learning curves → Applicability domain → PFI explainability → KRR bonus
```

---

## Key Results

**Classical ML — Best model: XGBoost with CFS-selected physicochemical descriptors**

| Split | MAE | RMSE | R² |
|-------|-----|------|----|
| Random | 0.520 | 0.732 | 0.637 |
| Scaffold | 0.588 | 0.765 | 0.584 |

The RMSE of 0.732 on the random split is competitive with the MoleculeNet benchmark (~0.65) reported for graph convolution models, using only hand-crafted descriptors.

**Model ranking** (consistent across all encodings and splits): XGBoost > Random Forest > Linear Regression

**Encoding ranking** (consistent across most combinations): Physchem > MACCS > Morgan > RDKit FP

**GNN models *(Louie's contribution)* — Best model: PyG GIN, scaffold split**

| Split | Model | RMSE | MAE | R² |
|-------|-------|------|-----|----|
| Scaffold | PyG GIN | 0.500 | 0.379 | 0.823 |
| Random | PyG GIN | 0.667 | 0.508 | 0.699 |

---

## Feature Selection Findings

CFS identified 49 correlated clusters from ~100 descriptors and selected 33 / 26 / 37 features for LR / RF / XGBoost respectively, achieving cross-validated R² of 0.440 / 0.601 / 0.607.

**logp_rdkit** appears in all three selected sets — the single most universally important descriptor for logD prediction. **fr_COO** (carboxylic acid count) is the second most important, chemically interpretable as carboxylic acids reducing lipophilicity through ionization at pH 7.4.

---

## Applicability Domain

Scaffold test molecules were grouped into four Tanimoto similarity bins. MAE and RMSE increase monotonically as similarity to the training set decreases:

| Similarity bin | n | MAE | RMSE |
|---------------|---|-----|------|
| > 0.7 (very similar) | 207 | 0.470 | 0.587 |
| 0.5–0.7 (similar) | 301 | 0.572 | 0.723 |
| 0.4–0.5 (dissimilar) | 158 | 0.660 | 0.858 |
| < 0.4 (very dissimilar) | 174 | 0.707 | 0.921 |

**AD threshold: nn_sim = 0.5** — 64.2% of test molecules fall inside the AD (MAE = 0.533), vs 29% higher MAE outside (MAE = 0.687). Bootstrap analysis confirmed the R² non-monotonicity between the two lowest bins was a sample size effect.

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python (RDKit) | Cheminformatics, molecular representations, descriptors |
| TDC (PyTDC) | Standardized dataset access and splits |
| scikit-learn | ML models, cross-validation, feature selection, PFI |
| XGBoost | Gradient boosting regression |
| PyTorch / PyTorch Geometric | GNN models *(Louie)* |
| GPyTorch | KRR with Tanimoto and Gaussian kernels |
| UMAP | Chemical space visualization |
| pandas, NumPy, matplotlib, seaborn | Data manipulation and visualization |

---

## Assumptions & Limitations

- No logD range filter was applied — all measured values are physically meaningful for drug-like compounds
- Missing values in physicochemical descriptors were imputed with column medians computed on training set only
- The dataset is in a data-limited regime — learning curves show performance has not plateaued, and additional logD measurements would meaningfully improve generalization
- Scaffold split creates a moderately harder evaluation (median Tanimoto similarity drops from 0.688 to 0.593), reflecting relatively homogeneous chemical space in this dataset

---

## Deliverables

- `notebooks/ML4DD_FinalProject.ipynb` — Classical ML pipeline (EDA → feature selection → model comparison → applicability domain → explainability)
- `notebooks/GNN_random_split.ipynb` — GNN models, random split *(Louie)*
- `notebooks/GNN_scaffold_split.ipynb` — GNN models, scaffold split *(Louie)*
- `report/Lipophilicity_report.pdf` — Scientific report (Abstract, Introduction, Methods, Results, Discussion)
- `presentation/Lipophilicity_presentation.pdf` — Group presentation slides

---

*Developed as part of the Machine Learning for Drug Design course at SUPSI (Dalle Molle Institute for Artificial Intelligence, DTI) during an exchange semester from Breda University of Applied Sciences.*
