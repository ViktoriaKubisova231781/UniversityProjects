# Plant Root Phenotyping & Robotic Inoculation — NPEC

**Viktória Kubišová · Breda University of Applied Sciences · 2025**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![PyBullet](https://img.shields.io/badge/PyBullet-Robotics%20Simulation-grey?style=flat)
![Computer Vision](https://img.shields.io/badge/CV-U--Net%20Segmentation-orange?style=flat)
![Robotics](https://img.shields.io/badge/Robotics-PID%20Controller-blue?style=flat)
![CRISP-DM](https://img.shields.io/badge/Framework-CRISP--DM-blue?style=flat)

---

## Overview

The Netherlands Plant Eco-phenotyping Centre (NPEC) operates the Hades system — a high-throughput facility capable of processing 10,000+ *Arabidopsis thaliana* seedlings across 2,000+ Petri dishes daily. The challenge: manually measuring root growth and inoculating plants at this scale is infeasible. This project addresses both halves of the problem — computer vision to automate root analysis, and robotics to translate those measurements into physical inoculation actions.

The solution is a fully integrated pipeline: a U-Net semantic segmentation model identifies root structures from grayscale Petri dish images, post-processing extracts per-plant root lengths via skeletonisation and Dijkstra path mapping, and a 3D PID controller uses the detected root tip coordinates to guide a simulated Opentrons OT-2 liquid handling robot to each target location.

---

## Research Question

> *"How can computer vision and robotic control be combined to automate the analysis of plant root systems and enable precise, high-throughput inoculation of Arabidopsis thaliana seedlings at scale?"*

---

## Business Context

**Client:** Netherlands Plant Eco-phenotyping Centre (NPEC), Utrecht University  
**System:** Hades — in-vitro root phenotyping module  
**Problem:** Manual root measurement and inoculation of thousands of seedlings per batch is time-consuming, error-prone, and not scalable for high-throughput research.

**Business Value:**
- Automated root segmentation and length measurement replaces manual annotation
- Root system architecture (RSA) analysis provides reproducible, quantitative growth data
- Robotic inoculation controlled by CV output enables precise, consistent delivery of microbes or synthetic communities to target root tips
- Supports downstream research on root-microbe interactions and stress-resistant genotype discovery

---

## Dataset

| Property | Detail |
|----------|--------|
| Images | 126 black-and-white Petri dish images (Hades system) |
| Image size | 4202 × 3006 px |
| Species | *Arabidopsis thaliana*, 5 plants per dish |
| Annotation classes | root, shoot, seed, occluded_root |
| Annotation tool | LabKit (ImageJ plug-in) |
| Measurement dataset | 3 images with ground truth landmark coordinates and root lengths |
| Kaggle competition dataset | 11 images with hidden primary root length ground truth |
| Dataset notes | Merged two cohorts (Y2B_23 and Y2B_24); filtered to root-rich patches (≥30 pixels); sampled 10% background patches to balance training |

---

## Approach

| Step | Description |
|------|-------------|
| 01 Image annotation | Manually annotated Petri dish images using LabKit: root, shoot, seed, occluded_root classes |
| 02 ROI extraction | Traditional CV (external contour detection) to crop the Petri dish and remove black border regions |
| 03 Instance segmentation (traditional) | Connected component analysis to isolate individual plant instances without deep learning |
| 04 Semantic segmentation (U-Net) | Trained patch-based U-Net on annotated dataset; 256×256 and 512×512 patch sizes with 50% step overlap |
| 05 Instance segmentation (DL) | Applied post-processing to U-Net output to separate individual plant instances per vertical band |
| 06 Landmark detection | Detected primary root tip, root-hypocotyl junction, and lateral root tips via skeletonisation (skan + networkx) |
| 07 Morphometric analysis | Measured primary and lateral root lengths using Dijkstra's shortest path along skeletonised root masks |
| 08 Kaggle competition | Submitted primary root length predictions; iterated on dataset curation, loss functions, and path estimation logic |
| 09 Robotics — simulation env | Set up PyBullet simulation of Opentrons OT-2; determined working envelope via corner mapping |
| 10 Robotics — PID controller | Implemented a 3D PID controller (one per axis) to move the pipette to root tip coordinates extracted from CV pipeline |
| 11 Robotics — RL (partial) | Began PPO reinforcement learning training locally; not completed due to time constraints |
| 12 Integration | Connected CV pipeline output (root tip pixel coordinates → robot coordinates) to PID controller for full inoculation simulation |
| 13 Error analysis | Iterated on dataset composition, segmentation method, and length estimation logic across 4 documented iterations |

**Pipeline:**

```
Raw Petri dish image → Petri dish ROI extraction → Patch-based U-Net segmentation
→ Morphological post-processing → Per-plant instance separation (5 vertical bands)
→ Skeletonisation + Dijkstra path mapping → Primary root length + landmark coordinates
→ Coordinate transform (pixel → robot space) → 3D PID controller → OT-2 inoculation
```

---

## Results

### Computer Vision — U-Net Segmentation

| Metric | Result |
|--------|--------|
| Validation F1 score (best model) | 78–84% |
| Patch size | 256×256 (step size 128) |
| Loss function | BCE + Dice Loss (final iteration) |
| Augmentation | Real-time via ImageDataGenerator (flip, lightroom enhancement) |
| Dataset | Merged Y2B_23 + Y2B_24, filtered and augmented |

### Root Length Estimation — Kaggle Competition

| Submission | Public sMAPE | Private sMAPE |
|-----------|-------------|--------------|
| Best selected (`020_submission.csv`) | 11.870 | 4.586 |
| Best private (`015_submission.csv`) | 12.383 | **3.487** |
| First submission (`001_submission.csv`) | 13.293 | 8.712 |

The best private sMAPE of 3.487% on the held-out test set met the client's target of <5%. Iterating from the initial Dijkstra path implementation (submission 001) to the final pipeline reduced private sMAPE from 8.712% to 3.487%.

### Robotics — PID Controller

| Metric | Result |
|--------|--------|
| Accuracy (within 1 mm) | **100%** across all 5 root targets |
| Average steps to target | 62.00 |
| X/Y axis convergence | Smooth, no major overshoot |
| Z-axis | Early jitter, stabilised |
| Final distances per root | 0.000350–0.000360 m |

The 3D PID controller successfully inoculated all 5 root tips within 1 mm, demonstrating accurate droplet placement at each target location in the PyBullet simulation.

---

## Error Analysis & Key Iterations

Four documented iterations drove the performance improvements:

**Iteration 1 — Dataset curation:** Merged two annual cohorts, removed non-root masks, filtered to root-rich patches, and added balanced background samples to reduce false positive rate.

**Iteration 2 — Root length estimation method:** Initial skan/networkx top-bottom tip detection and verticality-scored paths were replaced with Dijkstra shortest-path mapping from topmost to bottommost skeleton pixel, which produced the most accurate and stable results.

**Iteration 3 — Segmentation method:** Switched from per-band component detection to global connected component analysis with centroid-based assignment, adding geometric checks for small root components (area < 400 px: angle ≥ 70°, edge margin ≥ 250 px).

**Iteration 4 — Loss function:** Replaced binary crossentropy with combined BCE + Dice Loss, improving overlap between predicted and true masks and reducing the impact of class imbalance on small root detection.

**Known limitations:** Overlapping roots make path disambiguation difficult. Lateral roots that extend above or below the primary root tip cause the Dijkstra path to overestimate or mis-start the primary root measurement — addressed partially through geometric filters, but unresolved in cases of heavy root occlusion.

---

## Pipeline Considerations

- Top 15% of image height treated as noise and excluded
- Minimum valid root component area: 100 px; aspect ratio ≥ 1.5
- Small components (< 400 px) must have orientation angle ≥ 70° and be ≥ 250 px from image edges
- Valid roots start in the top 30% of image height
- Plants segmented into 5 vertical bands; only the largest valid component per band retained
- Primary root length defined as the Dijkstra shortest path from topmost to bottommost skeleton node

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python (TensorFlow, Keras) | U-Net segmentation model training |
| OpenCV, scikit-image | Image preprocessing, morphological operations, ROI extraction |
| skan, networkx | Root skeletonisation and path analysis |
| PyBullet | OT-2 robot simulation environment |
| Stable-Baselines3 (PPO) | Reinforcement learning controller (partial) |
| Weights & Biases | Experiment tracking for RL training |
| Kaggle | Competition platform for CV pipeline evaluation |
| LabKit (ImageJ) | Manual image annotation |

---

## Assumptions & Limitations

- Trained and evaluated exclusively on *Arabidopsis thaliana* in controlled Hades imaging conditions; generalisation to other species or environments untested
- Simulation-only validation for robotics — no physical OT-2 hardware integration
- RL training was initiated locally but not completed; PID controller is the primary robotic solution
- Dijkstra path estimation underperforms on images with heavily overlapping or interleaved root systems
- Root-rich patch filtering may discard early-stage seedling images with sparse root structure

---

## Deliverables

- `CV_notebooks/` — ROI extraction, traditional and U-Net segmentation, landmark detection, morphometric analysis, and full pipeline notebooks (v15 and v20); saved U-Net model weights (model_8, model_10, model_12)
- `robotics/PID/` — 3D PID controller implementation, single and multi-target test scripts, simulation recording
- `robotics/pipeline/` — Integrated CV + PID pipeline script, per-root trajectory plots, and benchmark charts
- `robotics/reinforcement_learning/` — Custom Gymnasium wrapper, PPO training scripts, and wrapper test scripts
- `robotics/sim_env/` — PyBullet simulation environment files and simulation GIF
- `presentation/Solutions_for_a_Greener_Future.pdf` — Final project presentation

---

*Developed for the Netherlands Plant Eco-phenotyping Centre (NPEC) as part of the Year 2 Block B Computer Vision & Robotics module, BSc Applied Data Science & AI programme at Breda University of Applied Sciences.*
