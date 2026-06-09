# Predictive Maintenance Dashboard — SAM Automation

**Chrislande Duterloo, Monieka Hardjosoedarmo, Viktória Kubišová, Bartosz Kudyba, Kajetan Neweś · Breda University of Applied Sciences · 2024–2026**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LSTM](https://img.shields.io/badge/Model-LSTM%20Autoencoder-blueviolet?style=flat)
![IoT](https://img.shields.io/badge/Domain-Industrial%20IoT-orange?style=flat)
![CRISP-ML](https://img.shields.io/badge/Framework-CRISP--ML(Q)-blue?style=flat)

---

## Overview

This project was developed in collaboration with SAM Automation — a Dutch ABB Cobot Expert delivering automation systems for food and pharmaceutical packaging. The goal was to transition SAM's support model from reactive to proactive maintenance by building a real-time anomaly detection dashboard for industrial robotic arms.

The platform ingests live telemetry from a Niryo Ned 2 university lab robot via SSH and UDP connections, processes signals through an LSTM Autoencoder trained on healthy machine behaviour, and surfaces anomalies through an interactive Streamlit dashboard with a two-tier traffic-light alert system. The system runs on edge devices with local inference, making it deployable without cloud infrastructure.

The project was an interdisciplinary collaboration of five — one data scientist, one data engineer, and three analytics translators — working across an 18-week CRISP-ML(Q) lifecycle with milestone presentations to SAM's stakeholders.

**My role:** Analytics Translator and Project Manager — led daily stand-ups, maintained the Azure DevOps board, ran sprint planning and retrospectives, and managed work allocation across an interdisciplinary team that included students from non-technical backgrounds. On the research and analysis side, I was responsible for all business understanding documentation and legal compliance materials, and conducted EDA on collected robot telemetry (healthy and faulty signal analysis, baseline visualisations, data quality assessment).

---

## Research Question

> *"How can real-time telemetry data from industrial robotic arms be used to detect anomalies and provide early warnings of potential failures, enabling SAM Automation to transition from reactive to proactive maintenance?"*

---

## Business Context

**Client:** SAM Automation — ABB Cobot Expert, Netherlands  
**Contact:** Stan Jacobs (CEO)

Reactive maintenance in packaging lines can cost 5–20% of annual revenues, with reactive repairs typically three times more expensive than preventive ones. Competitors such as FANUC, ABB, and KUKA now offer predictive maintenance platforms, but these remain brand-specific. This project demonstrates a cross-brand predictive maintenance approach — validated on a university lab robot — that SAM can scale to client environments.

**Business Value:**
- Early anomaly detection reduces unplanned downtime and lowers emergency service costs
- Traffic-light alert system with evidence cards (signal snippets, reconstruction error plots) reduces alert fatigue and supports technician decision-making
- Edge-first processing means real-time inference runs locally without cloud dependency
- Cross-brand design validated on Niryo Ned 2, with architecture extendable to other cobot platforms
- Research evidence suggests predictive strategies can reduce downtime by up to 50% and cut maintenance costs by 10–40%

---

## Approach

| Phase | Description |
|-------|-------------|
| Business Understanding | Defined early warning requirements with SAM; produced BRD, research proposal, project scoping, risk management, literature review |
| Data Understanding | Connected to Niryo Ned 2 via SSH and Arduino sensors via UDP; EDA on healthy and faulty telemetry batches; data quality assessment; controlled fault simulations (vacuum leaks, misalignment, clogged filters) |
| Model Development & Data Engineering | Built unified data ingestion pipeline; implemented LSTM Autoencoder trained on healthy runs; percentile-based dynamic thresholds; two-tier alert severity; asynchronous retraining service |
| Tool Refinement & Testing | Streamlit dashboard with live monitoring, historical data, and training views; usability feedback sessions with SAM stakeholders; unit and integration tests |
| Finalisation & Launch | Final presentation and demo to SAM Automation (22/01/2026); handoff documentation and scale-up roadmap |

**Technical approach — inspired by Givnan et al. (2021):**

```
SSH (Niryo Ned 2 telemetry) + UDP (Arduino sensors)
          │
    unified_collector.py
          │
    core_ml/ (preprocessing → filters → LSTM Autoencoder → baselines)
          │
    services/ (async_trainer, state_manager)
          │
    Streamlit views:
      live.py        → real-time signals + traffic-light alerts
      history.py     → historical data exploration
      training.py    → model performance + retraining status
```

**Key design decisions:**
- Model trained on normal behaviour only — no historical failure data required
- Percentile-based dynamic thresholds adapt to robot-specific signal ranges
- Two-tier alert severity (warning / critical) with evidence cards to support technician action
- Mock mode (`mock_v1.py`) enables development and testing without hardware access

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python (Streamlit) | Interactive dashboard frontend |
| SSH / UDP collectors | Real-time robot telemetry ingestion |
| LSTM Autoencoder | Anomaly detection on time-series signals |
| scikit-learn, NumPy | Signal preprocessing and filtering |
| pandas, matplotlib | EDA and baseline visualisations |
| Azure DevOps | Sprint management, backlog, retrospectives |
| pytest | Unit and integration testing |

---

## Assumptions & Limitations

- System developed and validated on a Niryo Ned 2 university lab robot; not yet tested on SAM's production cobots
- Fault simulations (vacuum leaks, misalignment, clogged filters) approximate real failure modes but may not fully represent factory conditions
- Anomaly thresholds calibrated on available lab data; more failure event data would improve model robustness
- Real-time integration with client-side live production systems is out of scope for the MVP
- Mock mode provides a synthetic fallback when hardware is unavailable

---

## Deliverables

- `dashboard/` — Full Streamlit application: collectors, LSTM Autoencoder pipeline, services, views, utils, tests, `app.py`, `config.py`, `requirements.txt`
- `business_understanding/` — BRD, literature review, research proposal, project scoping, risk management, milestone presentations, final presentation slides
- `legal_documentation/` — Data Management Plan, GDPR checklist, FAIR checklist, codebook, framework assessment, electrical cabinet documentation
- `notebooks/` — EDA notebooks (first batch, playground, faulty data, healthy data), baseline visualisations, data quality assessment, architecture diagrams

---

*Developed for SAM Automation as part of the Year 3 Semester A Specialisation Project, BSc Applied Data Science & AI programme at Breda University of Applied Sciences.*
