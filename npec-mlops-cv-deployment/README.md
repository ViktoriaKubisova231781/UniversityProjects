# IRIS — Plant Phenotyping CV Pipeline: MLOps & Deployment (NPEC)

**Viktória Kubišová, Victoria Vicheva, [teammates] · Breda University of Applied Sciences · 2025**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Azure ML](https://img.shields.io/badge/Azure-Machine%20Learning-0078D4?style=flat&logo=microsoftazure&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat&logo=githubactions&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-Production%20Deployment-blueviolet?style=flat)

---

## Overview

This project productionises the plant root phenotyping computer vision pipeline built for the Netherlands Plant Eco-phenotyping Centre (NPEC) in the previous block. The result is **IRIS** — a browser-accessible plant analysis platform that takes Petri dish images as input and returns segmentation masks, root measurements, and downloadable PDF reports.

Working in a group of five over an 8-week Agile Scrum cycle, the team built a modular Python package, a FastAPI backend with inference, feedback, and retraining endpoints, a multi-step HTML/CSS frontend, Docker containerisation, GitHub Actions CI/CD, and Azure ML integration.

**My contributions:** Served as project manager throughout — assigning roles, maintaining the Azure DevOps board, leading daily stand-ups, sprint planning, and retrospectives. On the technical side, I designed and built the backend package architecture: API routers, scripts, and utils modules; implemented the CLI; contributed to the Azure ML setup including data upload, model deployment scripting, and endpoint creation; and wrote all technical documentation including architecture diagrams and the business understanding document.

---

## Research Question

> *"How can the proof-of-concept plant root segmentation and landmark detection pipeline be productionised into a scalable, deployable MLOps system with automated retraining and cloud deployment on Azure ML?"*

---

## Business Context

**Client:** Netherlands Plant Eco-phenotyping Centre (NPEC), Utrecht University  
**Problem:** The proof-of-concept model from Block B worked in a notebook environment but was not production-ready — no API, no versioning, no monitoring, no automated retraining, and no way for researchers to interact with it outside of a local machine.

**Business Value:**
- REST API makes the segmentation and landmark detection model accessible to any robotic platform or research tool regardless of hardware
- Multi-step browser interface allows researchers to upload images, view segmentation results, and download PDF reports without writing code
- Feedback loop allows researchers to flag incorrect predictions and trigger model retraining on corrected data
- Dockerised application enables consistent deployment across local machines, on-premise servers, and cloud environments
- Azure ML integration provides model versioning, data asset management, and a path toward automated scheduled retraining

---

## Architecture

```
Researcher / Robotic System
          │
          ▼
  IRIS Web Interface  (HTML/CSS multi-step frontend)
          │
          ▼
  FastAPI REST API  ◄──── CLI (local interaction)
          │
     ┌────┴──────────────────────┐
     │  api/routers/             │  ← analysis steps 1-4, segmentation,
     │  api/scripts/             │    feedback, model routing
     │  api/utils/               │  ← preprocessing, postprocessing,
     └────┬──────────────────────┘    analysis, I/O, segmentation utils
          │
   Docker Container
          │
   ┌──────┴──────────────────────┐
   │  Local deployment           │  Azure ML (model registry, data assets,
   └─────────────────────────────┘  endpoints, training pipeline)
```

---

## Approach

| Sprint | Focus | Key deliverables |
|--------|-------|-----------------|
| 1 | Project scoping, repo setup, Agile planning | GitHub repo, Azure DevOps board, product backlog, architecture diagrams, roadmap |
| 2 | MVP inference application | Python package, FastAPI inference endpoint, CLI, Dockerfile, on-premise deployment |
| 3 | Data pipelines + cloud training | Azure ML data assets, training job setup, frontend development, backend-frontend integration |
| 4 | Deployment, monitoring, retraining | Feedback and retraining endpoints, CI/CD with GitHub Actions, cloud deployment |
| 5 | Testing, evaluation, demo | Unit tests, model registered on Azure ML, demo day |

---

## Status

IRIS is currently under active development. The local deployment, FastAPI backend, IRIS frontend, CI/CD pipeline, and Azure ML model registration are functional. Automated cloud training and retraining pipelines are still in progress.

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python (FastAPI, Uvicorn) | REST API backend |
| Docker | Containerisation and local deployment |
| Azure ML (Python SDK) | Model registry, data assets, training jobs, endpoints |
| Azure DevOps | Sprint management, backlog, retrospectives |
| GitHub Actions | CI/CD pipeline, automated testing |
| HTML / CSS | IRIS frontend web interface |
| Apache Airflow | Pipeline scheduling |
| TensorFlow / Keras | U-Net model (from Block B) |
| OpenCV, scikit-image | Image preprocessing |
| pytest, flake8, black | Testing, linting, formatting |

---

## Assumptions & Limitations

- Cloud training and retraining pipelines are still being set up
- Feedback-triggered retraining endpoint is implemented but not yet connected to a fully automated cloud training job
- System validated on NPEC Hades images only; generalisation to other imaging systems untested

---

## Deliverables

- `api/routers/` — FastAPI routers: full analysis steps 1–4, segmentation, feedback, model routing, image and root analysis
- `api/scripts/` — Data pipeline scripts: ingestion, augmentation, filtering, patchification, preprocessing, training, feedback processing
- `api/utils/` — Utility modules: analysis, I/O, model, postprocessing, preprocessing, segmentation
- `api/cli/` — CLI (`cli_v2.py`) for train, evaluate, and predict commands
- `api/demo/` — Demo recordings of backend endpoints, frontend interface, and Docker deployment
- `api/main.py` — FastAPI application entry point
- `azureml/` — Azure ML scripts: connection, data assets, data upload, endpoint creation, model deployment, preprocessing pipeline, scoring, environment setup
- `docs/` — Architecture diagrams (cloud-based, data pipeline), business understanding, roadmap
- `frontend_design/` — IRIS web interface HTML pages: index, analysis, results, download, documentation, about, contact

---

*Developed for the Netherlands Plant Eco-phenotyping Centre (NPEC) as part of the Year 2 Block D MLOps & Deploying AI Solutions module, BSc Applied Data Science & AI programme at Breda University of Applied Sciences.*
