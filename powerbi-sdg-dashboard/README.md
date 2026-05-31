# SDG Dashboard — Child Mortality & Vaccination Coverage

**Viktória Kubišová · Breda University of Applied Sciences · 2023**

![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat&logo=powerbi&logoColor=black)
![SDG 3](https://img.shields.io/badge/SDG%203-Good%20Health-4C9F38?style=flat)
![CRISP-DM](https://img.shields.io/badge/Framework-CRISP--DM-blue?style=flat)

---

## Overview

I built this dashboard to answer one question: **how does childhood vaccination coverage relate to under-five child mortality, and how has this relationship evolved over the past three decades?**

Using 30 years of data from the UN SDG Databank (1990–2020), I developed a 3-page interactive Power BI report for the SDG Hub@BUas — a real client — covering two SDG indicators: **3.2.1** (under-five mortality rate) and **3.B.1** (access to vaccines). The dashboard is designed for policymakers and researchers who need fast, evidence-based insight into global child health progress, with interactive filters by country, year, region, and income level.

---

## Research Question

> *"How can we reduce child mortality on a global scale, and what is the relationship between the trends in childhood vaccination rates and under-five mortality rates over the past three decades?"*

**SDG indicators covered:** 3.2.1 (Under-five mortality rate) · 3.B.1 (Access to vaccines)

---

## Approach

| Step | Description |
|------|-------------|
| 01 Business understanding | Defined research question aligned with SDG 3.2.1 & 3.B.1 indicators |
| 02 Data collection | Sourced 30 years of data from the UN SDG Databank (1990–2020) |
| 03 EDA & cleaning | Explored distributions, handled missing values, computed summary statistics in Power BI |
| 04 Analysis | Calculated correlation coefficients between vaccine coverage and under-five mortality rates |
| 05 Dashboard | Built a 3-page interactive Power BI report with maps, trend lines, KPIs, and filters |
| 06 Presentation | Live demo to lecturers, peers, and client at the BUas Data Science Conference |

**Pipeline:**
`UN SDG Databank (.csv)` → `Power BI Data Model` → `DAX measures` → `Interactive Dashboard` → `Conference presentation`

---

## Dashboard Pages

**1. Home**
Overview KPIs showing absolute changes across the full 30-year period: −5.49% under-five mortality rate, +5% DTP3 coverage, +54% MCV2 coverage, +47% PCV3 coverage. Summary narrative and entry point to the full report.

**2. Child Mortality**
Interactive world map with a year slicer (1990–2020), regional bar chart, income-level trend lines, and descriptive statistics (mean, median, standard deviation). Highlights the persistent gap between Sub-Saharan Africa (179 deaths/1,000) and high-income regions (14 deaths/1,000).

**3. Vaccines**
Three tabs — DTP3, MCV2, PCV3 — each with a global coverage gauge, country-level choropleth map, regional treemap, and a correlation coefficient card showing the strength of the relationship between that vaccine and under-five mortality.

---

## Key Findings

- Under-five deaths more than halved globally — from **12.5M in 1990** to **5.04M in 2020**
- All three vaccines show strong negative correlations with under-five mortality: **DTP3 (r = −0.93)**, **MCV2 (r = −0.98)**, **PCV3 (r = −0.99)**
- PCV3 shows the fastest adoption rate despite the lowest baseline worldwide coverage (32%)
- Sub-Saharan Africa and Southern Asia carry the highest burden; North America and Europe the lowest
- WHO estimates vaccination programmes prevent **2–3 million child deaths** every year

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Power BI (Power Query, DAX) | Data modelling, measures, dashboard & visualisation |
| UN SDG Databank | Primary data source (.csv) |
| CRISP-DM | Project framework |

---

## Assumptions & Limitations

- Data availability varies by country and year — some regions have gaps prior to 2000
- Correlation analysis does not imply causation; other socioeconomic factors contribute to mortality trends
- Dashboard reflects data as available in the UN SDG Databank at time of analysis (2023)

---

## Deliverables

- `dashboard/SDG_child_mortality_dashboard.pbix` — Full interactive Power BI report

---

---

# AI in Science Fiction — Minority Report Presentation

**Viktória Kubišová · Breda University of Applied Sciences · 2023**

![Presentation](https://img.shields.io/badge/Format-Presentation-purple?style=flat)
![AI Ethics](https://img.shields.io/badge/Topic-AI%20Ethics-coral?style=flat)

---

## Overview

A 7-minute individual presentation analysing AI technology as depicted in the film *Minority Report*, connecting it to real-world applications and evaluating its broader implications.

---


## Key Topics Covered

- Taxonomy of AI: domains and subdomains relevant to the chosen application
- Technical feasibility analysis across infrastructure, cost, and integration dimensions
- Ethical considerations: bias, privacy, transparency, and automation impact
- Legal considerations: GDPR and sector-specific regulation

---

## Deliverables

- `presentation/AI_in_Science_Fiction.pdf` — Presentation slides

---

*Both projects were developed as part of the BSc Applied Data Science & AI programme at Breda University of Applied Sciences.*
