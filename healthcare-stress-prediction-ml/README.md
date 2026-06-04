# Stress Prediction from Wrist Physiological Signals (WESAD)

**Viktória Kubišová, Alexandra Biddiscombe, Louie Daans, Youssef Sedra · SUPSI (Exchange) · 2026**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189B2D?style=flat)
![NeuroKit2](https://img.shields.io/badge/NeuroKit2-Physiological%20Signal%20Processing-blue?style=flat)
![Healthcare AI](https://img.shields.io/badge/Domain-Healthcare%20AI-red?style=flat)
![LOSO](https://img.shields.io/badge/Validation-Leave--One--Subject--Out-blueviolet?style=flat)

---

## Overview

This project was completed as part of the **Machine Learning and Deep Learning for Healthcare** course during my exchange semester at SUPSI. Working in a group of four, we developed machine learning models to classify affective states from wrist-worn physiological signals using the publicly available WESAD dataset (Schmidt et al., 2018).

**My contributions:** I led the initial data exploration (raw signal inspection, inter-subject variability analysis, protocol verification across all 15 subjects), and built the full modelling pipeline — feature selection, LOSO cross-validation framework, model training and evaluation, ROC analysis, per-subject F1 analysis, all visualizations — and wrote the report and presentation. My teammates handled signal preprocessing (filtering, segmentation) and feature extraction using NeuroKit2.

---

## Research Question

> *"Can we classify affective states — baseline, stress, amusement, and meditation — from wrist physiological signals alone, using subject-independent evaluation, and how does binary stress detection compare to the harder 4-class formulation?"*

---

## Dataset

**WESAD** (Wearable Stress and Affect Detection) — Schmidt et al., 2018
Access: [https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)

| Property | Detail |
|----------|--------|
| Subjects | 15 (originally 17; S1 and S12 excluded — sensor malfunction) |
| Device used | Wrist-worn Empatica E4 only (mandatory requirement) |
| Signals | BVP (64 Hz), EDA (4 Hz), TEMP (4 Hz), ACC (32 Hz) |
| Protocol | Controlled lab setting: baseline → stress (TSST) → amusement → meditation |
| Classes | Baseline (39.2%), Meditation (26.3%), Stress (22.2%), Amusement (12.4%) |
| Windows | 179,817 (60-second windows, 0.25-second step, 99.6% overlap) |

**Extension beyond the paper:** We added meditation as a 4th affective class, motivated by Schmidt et al. (2018) Section 6, which explicitly suggests it as a future direction. The original benchmark used 3 classes only.

---

## Approach

| Step | Description |
|------|-------------|
| 01 Data exploration *(mine)* | Raw signal inspection across all 15 subjects; verified sampling frequencies empirically; identified protocol versions A and B; quantified inter-subject variability (EDA during stress: 0.34–12.84 μS, 40x range) |
| 02 Signal preprocessing *(teammate)* | BVP: 4th order Butterworth bandpass (0.5–8 Hz); EDA: lowpass 1 Hz (documented deviation from paper's 5 Hz — Nyquist issue at 4 Hz sampling rate); TEMP: 5-second moving average; ACC: 3D magnitude + 1-second moving average |
| 03 Segmentation *(teammate)* | 60-second sliding windows, 0.25-second step → 179,817 windows; majority vote label assignment at 700 Hz |
| 04 Feature extraction *(teammate)* | 41 physiological features via NeuroKit2: BVP/HRV (11), EDA (9), TEMP (6), ACC (15); 2 quality flags dropped before modelling |
| 05 Feature selection *(mine)* | Step 1: hierarchical correlation clustering (threshold \|r\| ≥ 0.90) → 33 cluster representatives; Step 2: XGBoost importance ranking → top 20 features |
| 06 Modelling *(mine)* | Logistic Regression, Linear SVM, XGBoost, MLP — evaluated on both full (41) and selected (20) features |
| 07 LOSO cross-validation *(mine)* | Leave-One-Subject-Out, 15 folds; StandardScaler fit on 14 training subjects only per fold to prevent data leakage |
| 08 Evaluation *(mine)* | Macro F1 (primary), accuracy, confusion matrices, ROC curves (per-class AUC), per-subject F1 heatmaps, EDA variability vs F1 scatter |
| 09 Binary task *(mine)* | Stress vs non-stress as a separate task alongside the 4-class problem |
| 10 Critical analysis *(mine)* | Bias, fairness, XAI, GDPR Article 9, generalizability |

**Pipeline:**

```
Raw wrist signals (BVP, EDA, TEMP, ACC)
→ Signal filtering → Sliding window segmentation (60s, 0.25s step)  [teammate]
→ Feature extraction (41 features via NeuroKit2)                    [teammate]
→ Correlation clustering + XGBoost importance → 20 selected features [mine]
→ LOSO cross-validation (15 folds, StandardScaler per fold)          [mine]
→ 4 models × 2 feature sets × 2 tasks (4-class + binary)            [mine]
→ Macro F1 / Accuracy / ROC / Per-subject analysis                   [mine]
```

---

## Results

### 4-Class Classification (Baseline, Stress, Amusement, Meditation)

| Model | F1 (Full 41) | Acc (Full 41) | F1 (Selected 20) | Acc (Selected 20) |
|-------|-------------|--------------|-----------------|------------------|
| **Logistic Regression** | **0.524 ± 0.101** | 0.576 | 0.506 | 0.564 |
| XGBoost | 0.514 ± 0.087 | 0.625 | 0.519 | **0.629** |
| Linear SVM | 0.507 ± 0.087 | **0.651** | 0.487 | 0.631 |
| MLP | 0.478 ± 0.142 | 0.573 | 0.434 | 0.533 |

### Binary Classification (Stress vs Non-Stress)

| Model | F1 (Full 41) | Acc (Full 41) | F1 (Selected 20) | Acc (Selected 20) |
|-------|-------------|--------------|-----------------|------------------|
| **Logistic Regression** | **0.846** | **0.891** | 0.839 | 0.882 |
| Linear SVM | 0.841 | 0.905 | 0.810 | 0.883 |
| XGBoost | 0.833 | 0.898 | 0.845 | 0.907 |
| MLP | 0.828 | 0.894 | 0.777 | 0.861 |

Our best binary model achieves 89.1% accuracy compared to the paper benchmark of 88% (RF/AdaBoost). Direct comparison should be interpreted cautiously given differences in model setup, hyperparameter tuning strategy, and the absence of statistical significance testing.

---

## Key Findings

**Best model:** Logistic Regression with full 41 features — highest macro F1 on both tasks.

**Feature selection did not consistently help:** 3 of 4 models degraded with the selected 20 features (LR −0.018, Linear SVM −0.020, MLP −0.043 F1). Only XGBoost showed marginal improvement (+0.005). Full feature set preferred for macro F1.

**Class separability (ROC, best model):**
- Stress: AUC = 0.94 — most separable, consistent with strongest physiological response
- Meditation: AUC = 0.84
- Baseline: AUC = 0.83
- Amusement: AUC = 0.54 — near random chance; physiologically indistinguishable from wrist signals regardless of model or task formulation

**Per-subject variability:** Multi-class F1 ranges from 0.25 to 0.71 across subjects. Best subjects: S5, S14, S17. Worst: S2, S15, S16 — consistent across all models. EDA variability during stress shows only a weak correlation with F1 (r = 0.24), suggesting individual physiology drives performance differences more than signal strength alone.

**ACC dominance in feature importance:** acc_y_std (0.172), eda_scr_auc (0.153), acc_z_std (0.113) are the top 3 features. ACC accounts for 9 of the 20 selected features despite showing almost no class separation in boxplot analysis. This may reflect subject posture patterns rather than genuine stress physiology — a known risk of performing feature selection outside the LOSO loop, documented as a key limitation.

---

## Benchmark Comparison

| Task | Our best model | Our F1 | Our Accuracy | Paper (RF/AdaBoost) |
|------|---------------|--------|--------------|---------------------|
| 4-class | LR (full features) | 0.524 | 57.6% | 76% *(3-class — easier task)* |
| Binary | LR (full features) | 0.846 | 89.1% | 88% *(comparable, with caveats)* |

The 4-class gap is expected — harder task (4 vs 3 classes), no Random Forest (excluded due to computational constraints on 179,817 windows × 15 folds), and fixed hyperparameters not tuned inside LOSO.

---

## Critical Analysis & Healthcare Considerations

**Bias & fairness:** 15 subjects, 12 male / 3 female, all graduate students at a single institution, mean age 27.5. Insufficient statistical power to assess sex-based differences. Any deployed model risks systematic bias against underrepresented groups.

**XAI:** XGBoost feature importance provides signal-level interpretability. ACC feature dominance despite poor class separation highlights a risk of importance reflecting spurious correlations rather than stress physiology — SHAP per-subject analysis would be needed for individual-level explainability.

**GDPR Article 9:** Stress detection from wearables constitutes sensitive health data processing. Continuous workplace monitoring could be misused by employers. Real-world deployment would require full ethical review, data protection impact assessment, and transparent communication to users.

**Generalizability:** LOSO provides subject-independent evaluation but no external validation has been performed. The 40x inter-subject EDA range confirms that generalizing to truly novel populations remains a significant open challenge.

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python (scikit-learn) | ML models, LOSO cross-validation, feature selection |
| XGBoost | Gradient boosting + feature importance |
| NeuroKit2 | Physiological signal processing and feature extraction |
| SciPy | Signal filtering (Butterworth bandpass/lowpass) |
| pandas, NumPy | Data manipulation |
| matplotlib, seaborn | Visualizations, confusion matrices, ROC curves, per-subject heatmaps |

---

## Limitations & Future Work

Feature selection was performed outside the LOSO loop — introduces a potential data leakage risk for the importance ranking. Hyperparameters were fixed from literature values rather than tuned inside LOSO. BVP motion artifacts within the cardiac passband (0.5–8 Hz) are not fully removed by bandpass filtering. Dataset limited to 15 subjects in a lab-controlled setting with a homogeneous student population. Future work should address these through LOSO-internal tuning, end-to-end 1D CNN/LSTM on raw signals, and external validation on an independent dataset.

---

## Deliverables

- `notebooks/initial_exploration.ipynb` — Raw signal inspection, sampling frequency verification, protocol analysis, inter-subject variability
- `notebooks/feature_extraction.ipynb` — Signal preprocessing, sliding window segmentation, feature extraction via NeuroKit2
- `notebooks/multiclass_modelling.ipynb` — Feature selection, LOSO modelling, 4-class evaluation (full vs selected features)
- `notebooks/binary_modelling.ipynb` — Binary stress vs non-stress LOSO modelling and evaluation
- `report/stress_prediction_report.pdf` — Written report
- `presentation/stress_prediction_presentation.pdf` — Group presentation slides

---

*Developed as part of the Machine Learning and Deep Learning for Healthcare course at SUPSI (MeDiTech/BSP 2026) during an exchange semester from Breda University of Applied Sciences.*
