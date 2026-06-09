# Business Requirement Document (BRD)

**Project Name:** SAM Automation - Smart Maintenance for Industrial Robotics

**Status:** [Version 8] 

**Last Updated:** 23/01/2026  

**Authors:** Chrislande Duterloo, Monieka Hardjosoedarmo, Viktória Kubišová, Bartosz Kudyba, Kajetan Neweś 

**Collaborators:** Stan Jacobs, Shival Indermun

## Table of Contents

1. [Sign-off Grid](#sign-off-grid)
2. [Problem Statement](#problem-statement)
3. [Background & Context](#background--context)
4. [Business Impact Metrics](#business-impact-metrics)
5. [Business Requirements](#business-requirements)
6. [Key Dates](#key-dates)
6. [Adjustments to the project](#adjustments-to-the-project)
7. [Resources](#resources)

## Sign-off Grid

| Team Member | Role | Date |
| ----------- | ---- | ---- |
| Viktória Kubišová | Analytics Translator | 30/09/2025 |
| Bartosz Kudyba | Data Scientist | 30/09/2025 |
| Kajetan Neweś | Data Engineer | 30/09/2025 |
| Chirslande Duterloo | Analytics Translator | 30/09/2025 |
| Monieka Hardjosoedarmo | Analytics Translator | 30/09/2025 |

## Problem Statement

SAM Automation currently operates under a **reactive maintenance model** for its collaborative robots in packaging lines. Failures—such as suction cup leaks or gripper malfunctions—are only addressed after they occur, triggering **unplanned production stoppages, emergency service interventions, and elevated operational costs.** (Zhu et al., 2019; Murtaza et al., 2024).

To address this critical challenge, the company requires a **predictive maintenance prototype** that leverages both **robot telemetry** and **external sensors** to deliver **early warnings** and **actionable alerts** for technicians, enabling a transition towards a **proactive and scalable service model**.

## Background & Context

SAM Automation, an **ABB Cobot Expert** based in the Netherlands, delivers tailored automation systems such as **pick-and-place**, **palletizing**, and **waterjet cutting** for the **food and pharmaceutical packaging sectors**, where **uptime** and **reliability** are business-critical. Despite this strong position, the company’s current maintenance approach remains **reactive**, with robots serviced only after failures occur. This exposes clients to **costly downtime**, **emergency interventions**, and **compliance risks** in regulated environments. At the same time, industry research shows that **unplanned downtime can cost 5–20% of annual revenues**, with **reactive repairs often three times more expensive** than preventive actions (Zhu et al., 2019; Murtaza et al., 2024). Competitors such as **FANUC, ABB, and KUKA** now offer **predictive maintenance (PdM)** platforms, but these remain **brand-specific** (Bogue, 2025). This creates a strategic opportunity for SAM to differentiate with a **cross-brand, packaging-focused solution**, supported by evidence that **predictive strategies can reduce downtime by up to 50%** and **cut maintenance costs by 10–40%** (Zhu et al., 2019).

To explore this opportunity, the project will deliver an **MVP predictive maintenance prototype** centered on a **technician-friendly, cross-brand dashboard**. The MVP will ingest signals from a **PLC** and will be complemented by **external sensors** (**pressure/vacuum and vibration**) where internal telemetry is missing or insufficient.

Analytically, the MVP will begin with **threshold/condition baselines** and progress to **unsupervised anomaly detection** (e.g., **stacked autoencoders**) trained on healthy runs (Givnan et al., 2021). Alerts will follow a **traffic-light scheme** with **evidence cards** (signal snippets, reconstruction-error plots, short explanations) to reduce **alert fatigue** and support **technician decision-making**. Because **historical failure data** are scarce, the team will run **controlled fault simulations** on the **university lab robot (Niryo Ned 2)** to create examples and evaluate **targets** (Morettini, 2021). Structured **feedback sessions** with SAM’s CEO will shape **alert thresholds**, **wording**, and **dashboard ergonomics**.

The MVP is deliberately **non-production**: no factory integration, no SLAs, and no autonomy. Its role is to:  
1. Validate that **robot + sensor signals** can surface **early warnings** of common failure modes.  
2. Validate **alert formats** and confirm the **usability** of the dashboard with technicians.  
3. Deliver **evidence-based recommendations** that SAM can consider for **future scale-up** to client robots.  

---

**In Scope (MVP)**  
- Data collection on **Niryo Ned 2** (university lab).  
- **PLC**  where available for internal telemetry.  
- **External sensors** (pressure/vacuum, vibration) where gaps exist. 
- **Threshold baselines** and **unsupervised anomaly detection** (e.g., SAE).  
- **Streamlit dashboard** with traffic-light alerts and evidence cards.  
- **Stakeholder usability feedback sessions**.  
- **Controlled fault simulations** (vacuum leaks, misalignment, clogged filters).  

**Out of Scope (for MVP)**  
- **Factory deployment** (live production integration).  
- **SLAs, cybersecurity hardening**, or compliance certifications.  
- **Closed-loop control** or autonomous actions (decision-support only).  
- **ERP/CMMS integration**, automated work orders, or full **RUL forecasting**.  
- Long-term **cloud operations** beyond basic archival and retraining experiments.  


## Business Impact Metrics  

Because this project delivers a **prototype in a controlled lab environment**, success will be measured through a mix of **technical accuracy** and **usability feedback**, building evidence for future business impact.  

### Detection and accuracy metrics  
- The system must identify at least **two distinct simulated fault types** (e.g., suction leaks, clogged filters, misaligned grippers).  
- Anomalies should be flagged with **meaningful lead time** (seconds to minutes before task failure) to allow intervention.  

### Usability and adoption metrics   
- **Red alerts** should trigger a useful technician response. 
- **False alarms** should be minimized to avoid alert fatigue.

### Validation process  
Lab runs with healthy and faulty scenarios will be **labeled and compared** against system alerts. A **4–6 week dataset** of repeated loops will provide baselines for weekly reviews and iterative tuning.  

*Although the MVP will not run on production lines, these results will show whether **robot + sensor signals** can provide reliable early warnings and whether technicians find the outputs **trustworthy and actionable**—laying the foundation for future reductions in downtime, costs, and client risk.* 


## Business Requirements  

| Requirement | Priority | Notes |  
| ----------- | -------- | ----- |  
| Business Requirement Document (BRD) | P0 | Document describing deliverables and expected outcomes |  
| Data collection and processing | P0 | Pipeline for robust collection, cleaning, and integration of robot and sensor data |  
| Anomaly detection model | P0 | Model for reconstruction and error-based anomaly detection |  
| Streaming pipeline | P0 | Inference pipeline providing streamlined data flow to the dashboard |  
| User-friendly dashboard | P0 | Technician-facing dashboard with real-time alerts and visualizations |  
| Detailed alert description | P1 | Extended alert views (evidence cards, signal snippets, explanations) |  
| Cloud-based archiving | P1 | Limited, non-production storage of anomalous events for retraining and experiments |  
| Industrial applicability | P1 | Prototype designed with cobot environments in mind, though validated only in lab robots |  


## Key Dates

| Phase | Artifact | Date | Owner | Status |
| ----- | -------- | ---- | ------ | ------ |
| Business Understanding (Weeks 1–6) | Problem Scoping Document, Business Requirements Document v1, Risk Management Document, Literature Review, Final Project Proposal | 10-Oct-25 | Whole Team | Completed |
| Milestone M1 | Green Light Presentation (client + mentor) | 2-Oct-25 | Whole Team | Completed |
| Lab Setup (Weeks 7–9) | Tool/sensor list delivered, loop plans finalized, wiring/PLC/external sensors configured | 5-Nov-25 | Data Scientist + Engineer | Completed |
| First Loop Creation (Week 9) | First operational robot loop implemented, test runs documented | 6-Nov-25 | Data Scientist + Engineer | Completed |
| Data Understanding (Weeks 9–12) | Healthy and faulty loop datasets recorded, labeled, archived in Parquet/SQLite, Data Quality Report | 28-Nov-25 | Data Engineer + Translators | Completed |
| Milestone M2 | Data Quality Review & Confirmed Signal Set | 26-Nov-25 | Whole Team | Completed |
| Cabinet Prototyping (Weeks 10–16) | **3D design abandoned**, cardboard mock-up, and 3D-printed cabinet prototype for edge components | 15-Jan-26 | Translators | Completed |
| Model Dev & Data Engineering (Weeks 11–14) | Baseline models (threshold, Isolation Forest, SAE), feature builder, ingestion scripts, SAE reconstruction error plots (serial/PLC → Parquet/SQLite) | 12-Dec-25 | Data Scientist + Engineer | Completed |
| Dashboard Development (Weeks 12–14) | Streamlit MVP Option A (log-only) tested on Niryo; pilot report #1 | 12-Dec-25 | Data Engineer + Translators | Completed |
| Pipeline & Dashboard Refinement (Weeks 14–16) | Streamlit MVP Option B (real-time inference integration), hyperparameter tuning, tunable thresholds, alert robustness improvements | 9-Jan-26 | Data Scientist + Engineer | Completed |
| Milestone M3 | Models v1 + Pipelines v1 + Initial Dashboard Design | 18-Dec-25 | Whole Team | Completed |
| Pilot #2 (Weeks 13–16) | Cross-validation on UFactory robot with PLC data | 9-Jan-26 | Data Scientist + Engineer | **Cancelled (Out of Scope – Agreed)**  |
| Milestone M4 | Prototype v1.0 Presentation (2 robots tested) | 8-Jan-26 | Whole Team | **Cancelled (Scope Adjusted)** |
| Finalization & Testing (Weeks 16–17) | Unit testing, debugging, CI/CD setup, code polishing, pipeline & dashboard functionality improvements | 15-Jan-26 | Data Scientist + Engineer | Completed |
| Launch Prep (Weeks 17–18) | Technical Design Document (system overview + scale-up recommendations) + Packaging (docs, datasets, code, demo script) | 23-Jan-26 | Whole Team | Completed |
| Milestone M5 | Final Launch Presentation (client + mentors) | 22-Jan-26 | Whole Team | Completed |
| Hand-in | Submission of all deliverables | 23-Jan-26 | Whole Team | Completed |


## Adjustments to the project

**16.10.2025 - Incorporation of Arduino Uno (Bartek)** 

Direct connection to Niryo Ned 2 base (Raspberry Pi) was abandoned, due to frequency requirements for external sensors and ease of implementation. 

**28.10.2025 - Vibration senor replacement (Bartek)**

Piezo Electric Vibration sensor has been replaced with MPU-6050 accelerometer. This allows for higher sensitivity readings combined with magnitudes of variety per vibration frequency.  

**30.10.2025 - Meeting with IXON (Bartek)**

We had a meeting with IXON together with the client to discuss potential connectivity options for the project. During the discussion, the client agreed to proceed with using an IXON router and to explore how it can be integrated within the scope of our predictive maintenance prototype. Following the meeting, the client confirmed that the router has been ordered.

Additionally, the client requested that we design a prototype cabinet to house the IXON router together with the necessary electrical components. The idea is to create a compact, self-contained unit in which all required parts are locally mounted, allowing the cabinet to be transported easily and offered as a complete product to end clients.

**26.11.2025 – Discontinuation of IXON Usage (Bartek)**

After the initial setup and evaluation of the edge gateway, the client decided not to proceed with using the IXON router as part of the MVP implementation. As a result, the MVP will be executed using a student’s PC as the primary edge device for data ingestion, processing, and dashboard integration.

Despite this change, the team will still deliver a prototype cabinet design that demonstrates how an edge gateway (such as an IXON router or equivalent device) and the required electrical components could be housed in a compact, portable enclosure. This cabinet prototype remains relevant as a conceptual and product-oriented deliverable, supporting future integration options beyond the current MVP.

**18.12.2025 – Scope Adjustment: Pilot #2 and Prototype Focus (Viktoria)**

Due to a slight delay in the initial lab setup and a longer-than-expected data collection phase, the project timeline was reassessed together with the client. It was jointly agreed that proceeding with **Pilot #2 on a second robot** would not be feasible within the remaining timeframe.

The current MVP is **highly robot- and operation-specific**. Valid testing on a second robot would require:
- Rebuilding the full hardware and signal setup,
- Collecting a new baseline of **healthy operation data**,
- Re-running **fault simulations**,
- Retraining and retuning the **anomaly detection models** from scratch.

Given these constraints, and considering the limited time remaining, this approach was deemed **unreasonable for the MVP phase**.

As a result, the following planned items were **formally removed from scope**:
- **Pilot #2 (Weeks 13–16):** Cross-validation on a second robot (UFactory),
- **Milestone M4 (original):** Prototype validation across two robots.

Instead, effort was reallocated—by agreement with the client—toward **deepening the value of the single-robot MVP**, with focus on:
- Implementing **real-time inference functionality**,
- Performing **hyperparameter tuning** of the anomaly detection models,
- Introducing **configurable and tunable alert thresholds**,
- Improving **robustness, interpretability, and technician usability** of alerts.

This adjustment ensures that the final MVP remains **technically sound, well-validated, and demonstrably valuable**, while staying **realistic and achievable** within the project timeframe.

**18.12.2025 – Final Deliverable Format Adjustment (Viktoria)**

Based on guidance from the project mentor, it was clarified that there is **no fixed mandatory format** for the final project deliverable. As such, a separate final project report and standalone scale-up plan are **not required**.

By agreement with the client, the team decided to **consolidate all project documentation into a single Technical Design Document**. This document will capture the full scope of the project, including system architecture, data pipelines, modeling approach, validation results, dashboard design, key decisions, and **scale-up recommendations**.

**08.01.2026 – Cabinet Prototype Scope Adjustment (Minor Students)**

Due to the approaching project deadline and the need for the Analytics Translators to focus more heavily on **final documentation and deliverables**, the cabinet prototyping scope was reviewed together with the client.

It was jointly agreed that the team will **not deliver a fully 3D-printed cabinet enclosure** as part of the MVP. Instead, the cabinet prototype will consist of:
- A **cardboard cabinet mock-up** representing form factor and layout  
- **3D-printed internal components** (e.g., mounts, holders, brackets) to demonstrate how the edge device and electrical components would be placed and secured  

This adjustment preserves the **concept validation and commercial idea** of a portable cabinet solution, while keeping the scope realistic within the remaining timeframe. The prototype will still clearly demonstrate how the solution could be packaged and offered to clients as a complete unit.


## Resources

* Technical Design Doc: <https://edubuas-my.sharepoint.com/:w:/g/personal/231781_buas_nl/IQBD3nxf8V_OSbrB16PfY3gQAdtThEudK14R6YKZCIATmz0?e=4y8KS7>
* Related Docs: <https://github.com/BredaUniversityADSAI/2025-26ab-fai3-specialisation-project-team-sam_automation/blob/2fb645e2db63d1655c78226243dd96f2aad3eec6/Business_Understanding/SAM_Automation_Milestone_Presentation.pdf>
* Artifacts: <https://github.com/BredaUniversityADSAI/2025-26ab-fai3-specialisation-project-team-sam_automation/tree/305b2b4c0711f2354b0d322e56d2367f40614709/Business_Understanding>


## References  

- Bogue, R. (2025). Predictive maintenance for industrial robots. *Industrial Robot: The International Journal of Robotics Research and Application, 52*(3), 305–311. Emerald Publishing Limited. https://doi.org/10.1108/IR-03-2025-0081  

- Givnan, S., Chalmers, C., Fergus, P., Ortega, S., & Whalley, T. (2021). Real-time predictive maintenance using autoencoder reconstruction and anomaly detection. *arXiv preprint arXiv:2110.01447.* https://arxiv.org/abs/2110.01447  

- Morettini, M. (2021). *Machine learning in predictive maintenance of industrial robots* [Master’s thesis, Uppsala University]. DiVA Portal. http://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-452321  

- Murtaza, M., da Silva, R., Dall’Oglio, A., & others. (2024). Paradigm shift for predictive maintenance and condition monitoring from Industry 4.0 to Industry 5.0. *Journal of Industrial Information Integration, 34,* 100567. Elsevier. https://doi.org/10.1016/j.jii.2023.100567  

- Zhu, J., Djurdjanovic, D., Qiu, H., Lee, J., Ni, J., & Liao, H. (2019). A survey of predictive maintenance: Systems, purposes and approaches. *IISE Transactions, 51*(11), 1190–1206. Taylor & Francis. https://doi.org/10.1080/24725854.2018.1555380  

