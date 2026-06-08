# SmartAccidentPredict — Traffic Accident Severity Classifier (ANWB)

**Viktória Kubišová, Kees Klijs, Jędrzej Ludwiczak, Michał Bątkowski · Breda University of Applied Sciences · 2024**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![CRISP-DM](https://img.shields.io/badge/Framework-CRISP--DM-blue?style=flat)
![EU AI Act](https://img.shields.io/badge/Governance-EU%20AI%20Act-blueviolet?style=flat)

---

## Overview

This project was developed in collaboration with ANWB (the Royal Dutch Touring Club) as the real-world client. Working in a team of four, we built SmartAccidentPredict — a machine learning pipeline that classifies traffic accident severity into three categories (Fatal, Injured, Material Damage Only) and an interactive Streamlit dashboard that puts those predictions in the hands of city planners and law enforcement. The project used a data lake of Breda-specific traffic, weather, and road data accessed via a PostgreSQL data warehouse.

The project evolved substantially during development. We originally set out to cluster unsafe driving hotspots using ANWB Safe Driving telematics data, but discovered mid-project that the available datasets lacked common join keys for meaningful spatial linkage. We pivoted to accident severity prediction using the comprehensive Accidents_17_23 dataset, which contained the richest and most reliable ground truth for modelling.

**My contributions:** I led the full data preprocessing and EDA pipeline for the Accidents_17_23 dataset — including SQL data access, cleaning, outlier removal, feature engineering, label encoding, SMOTE balancing, and saving the preprocessing config to PostgreSQL. I also trained and iteratively refined the three classification models (Logistic Regression, Random Forest, Gradient Boosting) across four training iterations incorporating class weighting, SMOTE, and hyperparameter tuning, and built the deep neural network for the excellent ILO criterion. I also handled all project documentation and reporting: both project proposals, all presentations, the Definition of Done, the deployment plan, legal framework assessments, and team contributions tracking.

---

## Research Question

> *"Which machine learning approach best predicts the severity of traffic accidents in Breda, and how can this be integrated into a dashboard to support real-time decision-making by city authorities and emergency services?"*

---

## Business Context

**Client:** ANWB (Koninklijke Nederlandse Toeristenbond) / BUas Innovation Square  
**Stakeholders:** City authorities, police, traffic management, Breda municipality  
**Problem:** Emergency services frequently arrive too late at accident scenes. Without predictive tools, resource allocation is reactive rather than proactive, increasing response times and accident impact.

**Business value:**
- **Proactive safety** — predicts severity before response, enabling priority dispatch of emergency services
- **Resource optimisation** — data-driven deployment of law enforcement and traffic management resources to high-risk areas
- **Infrastructure decisions** — actionable insights for city planners to target interventions at dangerous intersections
- **Legal compliance** — system assessed as high-risk under EU AI Act Articles 6–15; full compliance framework documented

---

## Dataset

| Property | Detail |
|----------|--------|
| Primary dataset | Accidents_17_23 (Breda, Netherlands, 2017–2023) |
| Records | 6,902 entries (after outlier removal) |
| Target variable | Accident severity: Fatal (0), Injured (1), Material Damage Only (2) |
| Class distribution | Material Damage: 85.7% / Injured: 13.6% / Fatal: 0.7% |
| Features | 14 variables: transport modes, area type, light/road/weather conditions, speed limit, street |
| Additional sources | ANWB Safe Driving (telematics), KNMI (weather), Argaleo (road/greenery) — used for EDA and clustering |
| Storage | PostgreSQL data warehouse (`group15_warehouse` schema) |

---

## Approach

| Step | Description |
|------|-------------|
| 01 Business understanding | Project proposal (two iterations), AI Canvas, stakeholder analysis, EU AI Act risk classification |
| 02 Data collection | PostgreSQL data lake access; ANWB, KNMI, BRON, Argaleo, and Accidents_17_23 datasets |
| 03 EDA | Distribution analysis, correlation heatmap (phik), categorical bar charts, outlier visualisation |
| 04 Preprocessing | Column standardisation, duplicate removal, feature selection, textual inconsistency fixes, data type conversion, IQR outlier detection and removal, feature engineering (Road Type from Speed Limit), label encoding, 70/30 stratified split |
| 05 Class balancing | SMOTE on training set; class_weight='balanced' across model iterations |
| 06 SQL data engineering | PostgreSQL connection, simple and complex queries, preprocessed table stored back to warehouse; SQL JOINs demonstrated separately (Kees) |
| 07 Clustering | K-means and DBSCAN on ANWB telematics + intersection data; silhouette score + Davies-Bouldin evaluation |
| 08 Classification models | 4 training iterations per model: baseline → class_weight → SMOTE → hyperparameter tuning (GridSearchCV / RandomizedSearchCV) |
| 09 Deep learning | 2 CNN iterations: baseline architecture → L2 regularisation + StandardScaler; EarlyStopping + ReduceLROnPlateau |
| 10 Dashboard + deployment | Streamlit app with Folium map overlay for hotspot visualisation; model serialised to `.joblib` and integrated for live severity prediction |
| 11 Legal framework | EU AI Act high-risk classification (Articles 6–15); transparency, post-market monitoring, and prohibited practices addressed |

**Pipeline:**

```
PostgreSQL data warehouse → SQL queries → Accidents_17_23 dataset
→ EDA + distribution analysis → Preprocessing (cleaning, encoding, feature engineering)
→ SMOTE balancing → 3 classifiers × 4 iterations (LR / RF / GBM)
→ Deep neural network (2 iterations) → Streamlit dashboard (Jędrzej)
→ Folium map overlay for hotspot visualisation
```

---

## Model Iterations & Results

### Classification Models (Accident Severity: Fatal / Injured / Material Damage)

| Iteration | Change | LR Accuracy | RF Accuracy | GB Accuracy |
|-----------|--------|------------|------------|------------|
| 1 | Baseline (no balancing) | 0.85 | 0.84 | 0.85 |
| 2 | class_weight='balanced' | 0.45 | 0.84 | 0.65 |
| 3 | SMOTE oversampling | 0.60 | 0.76 | 0.71 |
| **4** | **Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)** | **0.61** | **0.77** | **0.79** |

**Best classical model: Hyper-tuned Gradient Boosting Classifier (Iteration 4)**

| Metric | Fatal (0) | Injured (1) | Material Damage (2) | Weighted Avg |
|--------|----------|------------|-------------------|-------------|
| Precision | 0.07 | 0.27 | 0.87 | 0.78 |
| Recall | 0.07 | 0.25 | 0.89 | 0.79 |
| F1-score | 0.07 | 0.26 | 0.88 | 0.78 |
| **Overall accuracy** | | | | **0.79** |

**Key finding:** Accuracy alone is misleading here. The baseline models scored 0.85 by simply predicting the majority class (Material Damage Only) for almost every record. Iterations 2–4 deliberately traded overall accuracy for better recall on Fatal and Injured cases — the outcomes that actually matter for emergency response. The hyper-tuned Gradient Boosting model achieved the best balance, outperforming Random Forest (0.77) on both overall accuracy and minority class F1.

**Key challenge:** Severe class imbalance — Fatal accidents represented only 0.7% of records (50 entries out of 6,902). Despite SMOTE and class weighting, Fatal recall remained low across all models, highlighting the need for more fatal accident data in future work.

### Deep Learning (Neural Network — Keras)

| Iteration | Architecture | Test Accuracy | Macro F1 |
|-----------|-------------|--------------|----------|
| 1 | 4 dense layers, LeakyReLU, BatchNorm, Dropout | 0.85 | 0.31 (majority class dominated) |
| **2** | **+ L2 regularisation, StandardScaler, batch_size=64** | **0.71** | **0.37 (more balanced across classes)** |

Iteration 1 matched the naive baseline — 85% accuracy by predicting only Material Damage. Iteration 2 incorporated L2 regularisation and feature scaling, reducing overall accuracy to 71% but achieving meaningfully better recall on the Injured class (0.31 vs 0.00). For a safety-critical system, this trade-off is correct.

---

## Team Contributions

| Team member | Primary contributions |
|------------|----------------------|
| **Viktória Kubišová** | Data preprocessing and EDA (Accidents_17_23), all classification models (4 iterations), deep neural network, SQL data access, project proposals, presentations, DoD, legal framework, documentation |
| **Kees Klijs** | EDA, data preprocessing (ANWB/BRON datasets), intersection dataset creation, SQL JOINs, clustering exploration |
| **Michał Bątkowski** | Weather data cleaning, clustering (K-means, DBSCAN), deep learning model (CNN), week 5–6 visualisations, model testing |
| **Jędrzej Ludwiczak** | Streamlit dashboard development, model deployment to `.joblib`, accidents-per-street and speeding EDA notebooks |

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python (pandas, NumPy) | Data manipulation and analysis |
| scikit-learn | Classification models, GridSearchCV, SMOTE, label encoding |
| TensorFlow / Keras | Deep neural network (CNN iterations) |
| psycopg2 / PostgreSQL | SQL data warehouse access and storage |
| Streamlit | Interactive dashboard for accident severity prediction |
| Folium | Geospatial hotspot map overlay in dashboard |
| matplotlib, seaborn, phik | EDA visualisations and correlation analysis |
| imbalanced-learn | SMOTE oversampling |
| CRISP-DM | Project framework |

---

## Legal & Responsible AI

The SmartAccidentPredict system was classified as **high-risk** under the EU AI Act (Articles 6–15) due to its direct influence on public safety, law enforcement resource allocation, and emergency response decisions. Legal obligations addressed include transparency (Article 52), post-market monitoring (Articles 61–62), and prohibition of misuse (Article 5). A full legal framework assessment was produced covering both individual and group contributions.

---

## Assumptions & Limitations

- The project pivoted mid-development from hotspot clustering to severity prediction when dataset merging was found infeasible due to absence of common keys between ANWB telematics and accident records
- Severe class imbalance (Fatal: 0.7%) limits reliable prediction of fatal accidents despite SMOTE; a larger fatal accident dataset would be needed for production use
- The Streamlit dashboard is a proof-of-concept demonstration of what the model can do with example inputs — it is not connected to a live data stream
- Accidents_17_23 covers Breda only and a single data source; generalisation to other cities would require retraining
- Feature selection was based on a single dataset; multi-source enrichment (weather, live traffic) would improve predictive power

---

## Deliverables

- `notebooks/preprocessing/` — Data cleaning, EDA, feature engineering, SMOTE balancing, and PostgreSQL table creation
- `notebooks/` — Clustering (K-means, DBSCAN), deep learning model, logistic regression, SQL JOINs, speeding and accidents-per-street EDA
- `streamlit_app/` — Streamlit dashboard, serialised model (`model.joblib`), and model integration notebook
- `report/Creative_Brief_Report.ipynb` — Full project report notebook
- `project_plan/` — Initial and final project proposals, Definition of Done, deployment plan
- `presentations/` — Final project presentation and proposal presentation
- `docs/` — AI Canvas, EU AI Act risk assessment, legal obligations, and legal assessment for initial project idea
- `pyproject.toml` — Poetry virtual environment configuration

---

*Developed for ANWB as part of the BSc Applied Data Science & AI programme (Block 1D capstone) at Breda University of Applied Sciences.*
