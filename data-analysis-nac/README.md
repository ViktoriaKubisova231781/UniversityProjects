# NAC Breda Player Scouting Tool — Football Analytics & ML

**Viktória Kubišová · Breda University of Applied Sciences · 2024**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189B2D?style=flat)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![CRISP-DM](https://img.shields.io/badge/Framework-CRISP--DM-blue?style=flat)

---

## Overview

I built a data-driven scouting tool for NAC Breda — a professional Dutch football club — to support their player acquisition strategy within a defined market value budget. Using a proprietary dataset of 16,535 players and 114 features from the 2022–2023 season, I designed an end-to-end pipeline from data cleaning through to a machine learning model that predicts attacking performance, enabling data-backed shortlisting of affordable talent.

The project was developed for a real client, with a site visit to NAC's Rat Verlegh Stadium as part of the stakeholder engagement process. The final report covers EDA, ML methodology, ethical considerations, and strategic recommendations for NAC Breda.

> ⚠️ **Note:** This project was developed using proprietary data provided by NAC Breda under a non-disclosure agreement. The raw dataset and source code are not included in this repository. Only the final report is shared publicly.

---

## Research Question

> *"Which data-driven indicators best predict successful attacking performance, and how can they be used to guide NAC Breda's player acquisition strategy within their market value range?"*

---

## Approach

| Step | Description |
|------|-------------|
| 01 Business understanding | Defined the scouting problem; identified 11 expiring contracts including 4 attackers as the key acquisition gap |
| 02 Data collection & audit | Audited 16,535 players × 114 features from European league scouting data (2022–2023 season) |
| 03 Data cleaning | Handled missing values by position logic, removed duplicates, categorized positions (GK/DF/MF/ATT), binned market values |
| 04 EDA | Answered 20 analytical questions covering age, position, goals, xG, market value, contract status, duel stats, and more |
| 05 Feature selection | Used Random Forest feature importance + correlation analysis to identify top predictors |
| 06 ML modelling | Trained and evaluated regression, classification, and clustering models with 5-fold cross-validation |
| 07 Hyperparameter tuning | Applied GridSearchCV to optimize model performance |
| 08 Model selection | Selected best-performing model based on R², MSE, RMSE, and MAE |
| 09 Reporting | Delivered a professional report covering EDA, ML methodology, ethical analysis, and strategic recommendations |

**Pipeline:**

```
Raw scouting data (45 Excel sheets) → Data cleaning & feature engineering → EDA (20 questions)
→ Feature selection (RF importance + correlation) → ML modelling (6 models)
→ Hyperparameter tuning (GridSearchCV) → Best model → Professional report
```

---

## Key Findings

**EDA highlights:**
- Successful attackers are typically aged 20–25, scoring the most goals and commanding the highest market values
- Dribbles per 90 is the strongest single predictor of successful attacking actions (>75% feature importance)
- Height and weight show negligible correlation with goals scored (r < 0.05)
- Attackers suffer almost twice as many fouls per 90 as defenders
- 11 NAC contracts were expiring in June 2024, including 4 attackers — the direct recruitment gap addressed by this tool

**Machine learning:**

I tested four regression models (Linear Regression, kNN, Random Forest, and XGBoost) and four classification models (Logistic Regression, kNN, Random Forest, and XGBoost) to predict attacking performance, alongside K-Means clustering to segment attacker profiles. All models were evaluated using 5-fold cross-validation and tuned with GridSearchCV, with the tuned XGBoost Regressor emerging as the best performer.

**Recommendations to NAC Breda:**
- Prioritize attackers aged 20–25; also consider 25–30 for a balance of talent and financial feasibility
- Focus recruitment scouting on dribbles per 90 as the primary shortlisting metric
- Apply the model across other positions by adapting the target variable and feature set

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python (pandas, NumPy) | Data cleaning, feature engineering, EDA |
| matplotlib, seaborn, missingno | Visualisation (20+ charts, missing data profiling) |
| scikit-learn | ML modelling, cross-validation, GridSearchCV, scaling |
| XGBoost | Gradient boosting regression & classification |
| sympy | Symbolic mathematics |
| CRISP-DM | Project framework |

---

## Assumptions & Limitations

- Market value capped at €350K to reflect NAC Breda's typical acquisition budget (based on Transfermarkt averages)
- Classification models were limited by significant class imbalance in the "On loan" target variable (~8% of players on loan)
- Dataset covers the 2022–2023 season only; player performance may vary across seasons
- Analysis focuses on attackers; a full scouting tool would require position-specific models

---

## Deliverables

- `report/NAC_Breda_report.pdf` — Professional report covering EDA, ML, ethical considerations, and recommendations

---

*Developed as part of the BSc Applied Data Science & AI programme at Breda University of Applied Sciences.*
