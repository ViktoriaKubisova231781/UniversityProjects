# RoadReader — Urban Transport Classifier (CNN + VGG16 + Human-Centered AI)

**Viktória Kubišová · Breda University of Applied Sciences · 2024**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Figma](https://img.shields.io/badge/Figma-F24E1E?style=flat&logo=figma&logoColor=white)
![CRISP-DM](https://img.shields.io/badge/Framework-CRISP--DM-blue?style=flat)
![UX Design](https://img.shields.io/badge/UX%20Design-Prototype%20%2B%20User%20Testing-blueviolet?style=flat)
![CNN](https://img.shields.io/badge/CNN-Computer%20Vision-orange?style=flat)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-VGG16-red?style=flat)

---

## Overview

I built **RoadReader** — an urban transport image classifier and app concept for the Innovation Square client at BUas. The project addresses a real societal problem: cities rely on manual or rudimentary automated processes to monitor traffic, leading to congestion, safety hazards, and inefficient urban planning.

The solution is a deep learning image classifier that identifies 8 types of terrestrial transport vehicles, paired with a Figma wireframe prototype demonstrating how it could be integrated into a camera monitoring platform for city authorities. The project spans the full ML lifecycle — from market research and dataset creation through to model training, explainability, fairness analysis, and human-centered UX design validated with user studies.

---

## Research Question

> *"Which deep learning approach best enables accurate classification of urban transport vehicles from camera images, and how can this be designed into a user-centered application for city authorities?"*

---

## Business Context

**Client:** Innovation Square @ Breda University of Applied Sciences  
**Main stakeholder:** City Authorities  
**Problem:** Traffic congestion and manual monitoring create delays, safety hazards, and economic losses in metropolitan areas. Automated vehicle classification can enable data-driven traffic management without human intervention.

**Business value:**
- **Real-time decision-making** — enables city authorities to adjust traffic signal timings, reroute vehicles, and prioritize emergency response routes based on live vehicle data
- **Cost savings** — reduces operational costs through optimized resource allocation, minimized fuel consumption from reduced congestion, and faster emergency response
- **Urban planning** — provides data-driven insights into traffic patterns, bottlenecks, and infrastructure needs
- **Security monitoring** — enables automated detection of unauthorized vehicles in restricted areas

**KPIs:** Real-time decision-making support, cost savings through reduced congestion and optimized resource allocation

---

## Dataset

I built the dataset from scratch by scraping images using Google Image Scraper and DuckDuckGo, then handpicking images to ensure quality and diversity.

| Property | Detail |
|----------|--------|
| Classes | Car, Bus, Bicycle, Train, Van, Truck, Motorcycle, Scooter |
| Images per class | 150 (handpicked) |
| Total images | 1,200 |
| Image size | 256 × 256 px |
| Split | 70% train / 20% validation / 10% test |

---

## Approach

| Step | Description |
|------|-------------|
| 01 Business understanding | Market research, stakeholder analysis (8 stakeholder groups), DAPS diagram, project proposal |
| 02 Dataset creation | Web scraping (Google Image Scraper + DuckDuckGo), manual quality filtering, 150 images/class |
| 03 Baselines | Random guess (12.5%), Human-Level Performance (95.5%), MLP baseline (48.33%) |
| 04 Preprocessing | Resizing, label encoding, one-hot encoding, train/val/test split; data augmentation for iterations 2 & 4 |
| 05 Model iterations | 4 iterations: plain CNN → CNN + augmentation → VGG16 transfer learning → VGG16 + augmentation |
| 06 Explainable AI | Applied XAI methods to analyse model interpretability and accuracy trade-off |
| 07 Error analysis | Visual analysis of 7 misclassified images; error categorisation by type and frequency |
| 08 MLP from scratch | Implemented full MLP (forward pass, loss, gradient estimation, gradient descent) without Keras/sklearn |
| 09 Human-Centered AI | Figma wireframe prototype → think-aloud study → A/B test (dropdown vs buttons) → final prototype + demo video |

**Pipeline:**

```
Web-scraped images → Manual filtering → Preprocessing → Baseline models
→ 4 CNN iterations (plain CNN / augmented / VGG16 / VGG16 + augmented)
→ XAI analysis → Error analysis → Figma prototype → User studies
```

---

## Model Iterations & Results

| Iteration | Model | Test Accuracy | Test Loss |
|-----------|-------|--------------|-----------|
| 1 | Plain CNN | 69.17% | 0.952 |
| 2 | Plain CNN + Data Augmentation | 55.65% | 1.269 |
| **3** | **VGG16 Transfer Learning** | **94.17%** | **0.164** |
| 4 | VGG16 + Data Augmentation | 92.47% | 0.233 |

**Best model: Iteration 3 — VGG16 Transfer Learning**
- Weighted average precision / recall / F1: **0.94 / 0.94 / 0.94**
- Only 7 misclassifications out of 120 test images
- Nearly matches human-level performance (95.5%) on the same task

**Key finding:** Data augmentation consistently decreased performance compared to the non-augmented equivalent, suggesting the self-scraped dataset was already sufficiently diverse. Transfer learning (VGG16) was the decisive factor in closing the gap to human-level performance.

---

## Baselines

| Baseline | Accuracy |
|----------|----------|
| Random guess (8 classes) | 12.5% |
| Basic MLP (best learning rate) | 48.33% |
| Human-Level Performance | 95.5% |
| **Best model (VGG16, Iteration 3)** | **94.17%** |

---

## Error Analysis

7 misclassifications were identified and categorised from the best model's test predictions:

| Error type | Frequency |
|------------|-----------|
| Perspective & orientation variability | 4/7 (57%) |
| Size perception | 4/7 (57%) |
| Inter-class similarity | 3/7 (43%) |
| Context & surroundings | 1/7 (14%) |
| Shape | 1/7 (14%) |

Most common failure modes involved unusual camera angles making vehicles appear elongated (e.g. a truck from behind resembling a train), size perception issues (small vans mistaken for cars), and visual similarity between vehicle types (sloped American school bus fronts resembling trucks).

---

## Human-Centered AI — RoadReader Prototype

I designed **RoadReader**, a mobile app concept for city authorities built as a clickable Figma wireframe prototype. The prototype demonstrates how the classifier could be integrated into a camera monitoring platform, with screens for multi-location camera dashboards and per-camera analysis pages showing predicted vehicle type, speed, traffic state, plate number, car brand, and live vehicle count. The interface was validated through user studies but is not connected to a live model.

**Think-aloud study (3 users):**
- Key findings: improve visibility of "Skip" and "Predictive Vehicle" elements; clarify homepage use case visuals; replace inactive hamburger menu with interactive navigation

**A/B test (independent samples t-test):**
- Tested dropdown navigation (Version A) vs location buttons (Version B)
- Version B was significantly preferred for enjoyment (p = 0.035), page connectivity (p = 0.010), and ease of switching between locations and cameras (p = 0.004 and p = 0.002)
- No significant difference in comprehension or perceived usefulness between versions

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python (TensorFlow, Keras) | Deep learning model training |
| VGG16 (ImageNet weights) | Transfer learning base |
| scikit-image, NumPy | Image loading and preprocessing |
| matplotlib, seaborn | Visualisation, learning curves, confusion matrices |
| Google Image Scraper, DuckDuckGo | Dataset collection |
| Figma | Wireframe prototype (RoadReader app) |
| Microsoft Forms | HLP survey and A/B test surveys |
| CRISP-DM | Project framework |

---

## Responsible AI

As part of the Responsible AI module, I conducted a bias analysis on the Imsitu dataset to develop fairness evaluation skills, then assessed my own dataset for potential biases — concluding that, given the classifier targets vehicle types rather than people, no sensitive attributes or significant biases were present.

**Imsitu bias analysis (skills exercise):**
- Identified historical/pre-existing gender bias: cooking images showed 81 women vs 37 men; dusting showed 120 women vs 26 men
- Identified racial representation bias: out of 24,894 "man" images, only 25 were labelled "black man"
- Proposed fairness interventions: Fairness Through Unawareness (gender bias) and Fairness Through Awareness (racial bias)
- Implemented and evaluated six group fairness metrics: Demographic Parity, Equal Selection Parity, Conditional Use Accuracy Equality, Equalized Odds, Equalized Opportunities, Predictive Equality

**Own dataset:** No significant bias identified — the classifier targets vehicle types only, with no human traits or sensitive attributes involved. Each class was balanced at 150 images across diverse environments.

---

## Assumptions & Limitations

- Dataset was self-scraped and may not fully represent all real-world environments, lighting conditions, or camera angles
- Data augmentation (horizontal flip, rotation, zoom) did not improve performance — possibly because the scraped images already had sufficient diversity
- Main failure modes are perspective variability and inter-class similarity; future work should include multi-angle training data and scale-invariant features
- The MLP from scratch was trained on a reduced dataset (20 images/class, 16×16 px grayscale) due to computational constraints
- The RoadReader app is a Figma prototype and is not connected to a live model

---

## Deliverables

- `notebook/urban_transport_classifier_roadreader.ipynb` — Full project notebook (business understanding → responsible AI → deep learning → human-centered AI)
- `presentation/roadreader_presentation.pdf` — Final project presentation slides
- `assets/DAPSdiagram.pdf` — DAPS diagram
- `assets/group_fairness_infographic.pdf` — Group fairness infographic
- `assets/think_aloud_study_conclusions.pdf` — Think-aloud study conclusions and design recommendations
- [Final Wireframe (Figma)](https://www.figma.com/file/EoVDyALiHjpYQxEZiFnbn9/Terrestrial-Classifier?type=design&node-id=63%3A99&mode=design&t=IC0gJYWtMfnL8d3s-1) — RoadReader clickable prototype
- [Demo Video](https://edubuas-my.sharepoint.com/:f:/g/personal/231781_buas_nl/EmBaYKdzsR5LoMxuDAIRGkcBXtFBRvUxGNcwpKr18bjDZw?e=1ie1lN) — RoadReader app walkthrough

---

*Developed for the Innovation Square @ BUas as part of the BSc Applied Data Science & AI programme at Breda University of Applied Sciences.*
