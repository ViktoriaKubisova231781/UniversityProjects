# Ethics & Legal Framework Assessment

**Project:** SAM Automation - Predictive Maintenance for Industrial Robots  
**Institution:** Breda University of Applied Sciences (BUas)  
**Duration:** September 2024 - January 2025 (18 weeks)  
**Authors:** Chrislande Duterloo, Monieka Hardjosoedarmo, Viktória Kubišová, Bartosz Kudyba, Kajetan Neweś 

---

## Introduction

### Purpose of This Document

This Ethics & Legal Framework Assessment provides a comprehensive evaluation of the ethical and legal considerations relevant to the SAM Automation predictive maintenance project. The document serves four primary purposes: 
- demonstrating compliance with applicable legal and ethical requirements during both the MVP development phase and potential future production deployment
- identifying and assessing ethical, legal, and technical risks associated with the project along with mitigation strategies
- documenting all design decisions, limitations, and assumptions to ensure responsible development practices
- providing clear guidance to BUas, SAM Automation, and potential clients about the system's capabilities, boundaries, and ethical considerations

### Project Context

The SAM Automation project develops a predictive maintenance system for industrial collaborative robots (cobots) using anomaly detection techniques. The system collects multi-modal sensor data from robots during operation, including joint positions, motor temperatures and voltages, vacuum pressure, and vibration measurements, then applies machine learning algorithms to detect early warning signs of potential failures.

The MVP is a laboratory-based prototype validated on a single university lab robot (Niryo Ned 2) using controlled operational loops and simulated fault scenarios. The system functions strictly as a decision-support tool, generating alerts based on detected sensor anomalies without controlling the robot or making autonomous decisions. Its data architecture processes only technical robot and sensor data and is explicitly designed to operate without collecting or storing any personal or human-identifiable information. User interaction is limited to a technician-facing dashboard that displays real-time sensor trends and alert notifications, allowing technicians to independently interpret alerts and decide on appropriate maintenance actions. The system is therefore intentionally designed as a monitoring and alerting prototype rather than an autonomous control system, ensuring full human oversight at all decision points.

### Assessment Approach

This assessment evaluates the relevance and applicability of major ethical and legal frameworks in two contexts:

1. **MVP Phase (Current):** The laboratory prototype being developed and tested at BUas from September 2024 to January 2025
2. **Scale-up Phase (Future):** Potential production deployment in SAM client factories with real-world robots and operational data

For each framework, the assessment provides an applicability determination of whether the framework is relevant to the project, justification based on project characteristics and legal or ethical criteria, current compliance status showing what has been implemented or documented, and recommendations for actions required for full compliance or best practices. The assessment prioritizes transparency about system limitations and proactive risk mitigation over claiming universal applicability or zero-risk operation.

---

## 1. Executive Summary

This assessment evaluates six major ethical and legal frameworks (GDPR, FAIR, ALTAI, DPIA, EU AI Act, DEDA) to determine their relevance to the SAM Automation predictive maintenance project across both MVP development and potential production deployment phases.

**Key Findings:**
- **MVP Phase:** Low ethical/legal risk - no personal data processed, lab environment only, decision-support system with human oversight
- **Applicable Frameworks:** FAIR principles (data management for internal reuse), Responsible AI practices (bias mitigation, explainability, security), DEDA (ethical reflection), comprehensive risk assessment
- **Not Applicable:** GDPR (system designed to process only robot sensor data without personal information - applies to both MVP and production phases), ALTAI (useful for reflection but not mandatory for low-risk application), DPIA (no high-risk personal data processing), EU AI Act (R&D exemption; if deployed would likely classify as medium-risk requiring transparency and human oversight already built into design)
- **Scale-up Considerations:** Enhanced security measures, field validation in diverse factory environments, formal ALTAI assessment recommended; GDPR remains non-applicable due to sensor-only data architecture

---

## 2. Framework Applicability Assessment

### 2.1 GDPR (General Data Protection Regulation)

**Status:** **NOT APPLICABLE by design** (MVP and production)

**Justification:**

The SAM Automation system is architecturally designed to avoid GDPR applicability through data minimization by design. The system processes exclusively technical sensor data from robots—including joint positions, motor temperatures and voltages, vacuum pressure, accelerometer/vibration measurements, error flags, and timestamps—without any collection, processing, or storage of personal information.

The system does not process operator identities, names, employee IDs, user accounts, login credentials, alert acknowledgment tracking, technician performance metrics, response times, or any audio/video recordings of individuals. The dashboard displays sensor readings and generates alert pop-ups when anomalies are detected (e.g., "Vacuum pressure dropped below threshold") without tracking or logging who views alerts or how technicians respond. The sole purpose is fault prediction, not human monitoring.

Under GDPR Article 4(1), personal data means "any information relating to an identified or identifiable natural person." Since the SAM system processes only robot sensor measurements and does not relate any information to identifiable individuals, it falls outside GDPR scope. This privacy-by-design architecture is maintained regardless of deployment scale. In both lab and production environments, robots generate sensor data and alerts are displayed to technicians without identity tracking. 

The system's core function—predictive maintenance through sensor anomaly detection—does not require or benefit from personal data collection. Its design relies exclusively on machine-level signals, ensuring that GDPR remains non-applicable throughout all forms of deployment.

---

### 2.2 FAIR Data Principles

**MVP Status:** **HIGHLY RELEVANT**  
**Scale-up Status:** **RELEVANT** (modified application)

**Justification**

The FAIR data principles are highly relevant for this academic MVP, as the project emphasises reproducibility, transparency, and structured knowledge transfer to the industry partner (SAM Automation), while aligning with BUas requirements for responsible research data management.

**In the MVP phase**, the project demonstrates strong FAIR alignment within the constraints of a non-production, lab-based prototype.

- **Findable:**  
  All datasets and documentation are stored in a structured, version-controlled GitHub repository. Data is organised using a consistent timestamp-based folder structure and supported by comprehensive README files, description logs, and a detailed codebook documenting all 92 variables. Although no persistent identifiers or public repositories are used, the data is easily discoverable for authorised stakeholders (project team, BUas supervisors, and SAM Automation).

- **Accessible:**  
  Data is accessible through standard tools and protocols (Git, local filesystem access, Python-based loaders) with access restricted via GitHub permissions. Clear documentation ensures that authorised users can retrieve and interpret the data without reliance on undocumented knowledge.

- **Interoperable:**  
  Interoperability is ensured through the use of open, non-proprietary formats, including Apache Parquet for time-series data and TXT/Markdown for metadata and documentation. Standard conventions such as ISO 8601 timestamps and consistent variable naming are applied throughout.

- **Reusable (with limitations):**  
  The data is technically reusable due to its clear structure, extensive technical documentation, and adherence to open formats and domain conventions. However, the datasets are highly specific to the university lab setup (Niryo Ned 2 robot, custom external sensors, and bespoke data pipeline). While technically reusable, the data has no practical reuse value beyond this project without substantial re-ampling, reconfiguration, and retraining. Its primary value lies in demonstrating the pipeline architecture and analytical approach rather than reuse of the data itself.

**For production scale-up,** FAIR principles remain important for data management quality but with privacy adaptations for client factories. Data remains findable through internal factory systems, accessible only to authorized personnel at each client site with strict access controls, interoperable through standard formats to ensure system compatibility, and reusable within each organization with clear documentation. Client factory data stays private and isolated per site, ensuring sensitive operational data never leaves the factory environment.

The data is well-structured and documented for effective handoff to SAM Automation after project completion. See separate FAIR Data Checklist document for detailed compliance assessment.

---

### 2.3 ALTAI (Assessment List for Trustworthy AI)

**MVP Status:** **NOT MANDATORY** (useful for reflection)  
**Scale-up Status:** **HIGHLY RECOMMENDED**

**Justification:**

**For the MVP phase,** ALTAI is a useful framework for ethical reflection but not mandatory. The system qualifies as low-risk due to its decision-support function without autonomous control, lab environment with no real production impact, maintained human oversight where technicians review all alerts before taking action, and limited prototype scope with validation on a single university lab robot.

**Informal ALTAI Assessment:**

1. **Human Agency & Oversight:** Strong - alerts only, humans decide actions
2. **Technical Robustness:** Moderate - limited to simulated faults in lab
3. **Privacy:** Strong - no personal data
4. **Transparency:** Good - reconstruction error plots,  two-tier alert severity system 
5. **Fairness:** Limited scope - only 1 robot brand tested
6. **Societal Well-being:** Positive - reduces waste, extends equipment life
7. **Accountability:** Clear - BUas team responsible for MVP

The system demonstrates good alignment with trustworthy AI principles despite being a low-risk prototype. **For production deployment,** a formal ALTAI assessment becomes highly recommended due to real production impact, client trust requirements, and the need to ensure robustness and comprehensive trustworthy AI practices across diverse factory environments and robot brands beyond the lab-tested scope.

---

### 2.4 DPIA (Data Protection Impact Assessment)

**MVP Status:** **NOT REQUIRED**  
**Scale-up Status:** **NOT REQUIRED** (if sensor-only architecture maintained)

**Justification:**

Under GDPR Article 35, a DPIA is mandatory only for data processing "likely to result in high risk" to individuals. The assessment criteria for high-risk processing include large-scale systematic monitoring of publicly accessible areas, processing special category data such as health or biometric information at scale, automated decision-making producing legal or similarly significant effects on individuals, processing data of vulnerable groups, and use of innovative technology with inherent privacy risks.

The SAM Automation system does not meet any of the above-mentioned high-risk criteria.

**For the MVP,** the system processes no personal data, operates in a controlled lab environment, and affects no individuals. **For production scale-up,** the system's architecture—processing only robot sensor data without any capability to identify, monitor, or evaluate people—ensures that DPIA requirements remain inapplicable as long as the sensor-only design is maintained. However, if future system modifications were to introduce personal data processing capabilities (such as operator behavior logging or employee monitoring features), a DPIA would become necessary and consultation with qualified data protection expertise would be required before implementation.

---

### 2.5 EU AI Act (2024)

**MVP Status:** **NOT APPLICABLE** (R&D exemption)  
**Scale-up Status:** **LIKELY MEDIUM-RISK** (if deployed)

**Justification:**

**For the MVP phase,** the EU AI Act exempts research and development activities. As a lab prototype without market deployment, the system has no AI Act obligations during the MVP phase.

**For production scale-up,** the system would likely classify as medium-risk rather than high-risk. The system functions as decision-support rather than safety-critical autonomous control, placing it outside the high-risk category. However, as industrial automation with potential operational impact, it would likely fall under medium-risk classification requiring transparency measures, maintained human oversight, accuracy testing documentation, and cybersecurity safeguards—all of which are already implemented in the current design. Risk classification assessment should be conducted before production deployment to confirm regulatory requirements and ensure compliance with evolving AI Act provisions.

---

### 2.6 DEDA (Data Ethics Decision Aid)

**MVP Status:** **USEFUL FRAMEWORK** (for ethical reflection)  
**Scale-up Status:** **HIGHLY RELEVANT** (ethical guidance)

**Justification:**

**For the MVP phase,** DEDA provides a valuable framework for ethical reflection throughout the project development. The framework helps identify potential ethical issues beyond legal compliance and ensures responsible design decisions.

**Key Ethical Considerations:**

**Goal & Context:**
- Clear benefit: Reduce robot downtime, extend equipment lifespan
- Stakeholders: SAM Automation(efficiency), clients (cost savings), environment (less waste)
- Proportional: Data collection minimal and necessary

**Data & Methods:**
- Data minimization: Only technical sensor data collected
- Transparency: Methodology fully documented
- Quality assurance: Calibration procedures, validation testing

**Responsibility:**
- Clear ownership: BUas research team accountable for MVP
- Error mitigation: False alert reduction and technician feedback loops, technician feedback loops
- Documentation: All decisions traceable

**Impact:**
- Human-centered: Technician usability testing integrated
- Potential harms: Alert fatigue (false positives), missed faults (false negatives)
- Mitigation: Performance targets, iterative tuning with user feedback

**Fairness:**
- Limited diversity: One robot platform, lab conditions only
- Acknowledged: Limitations documented, not claimed as universal solution

**For production scale-up,** DEDA becomes even more important for addressing stakeholder concerns, power dynamics in factory settings, and long-term societal impacts of automation technology.

The project demonstrates ethical awareness and proactive risk mitigation throughout both MVP development and scale-up planning.

---

## 3. Responsible AI Documentation

### 3.1 Bias Assessment

**Overview:**

The SAM Automation MVP acknowledges several inherent limitations that create potential biases when generalizing from lab conditions to production environments. These biases stem from differences in equipment scale, environmental conditions, data labelling, fault simulation capabilities, sensor quality, and robot precision.

**Identified Biases:**

**1. Robot Diversity Bias**

**Issue:** Data collected and model tested exclusively on Niryo Ned2 robot.

**Impact:** Model may not generalize to other robot brands (FANUC, ABB, KUKA, UFactory) used in SAM client factories. Single robot platform cannot represent the mechanical, software, and operational diversity across industrial robotics manufacturers.

**Limitation:** Different robot brands have varying kinematics, motor controllers, sensor integrations, and operating characteristics that may produce different baseline patterns and fault signatures.

**Scale-up Mitigation:** The system architecture is deliberately designed to support model retraining on newly collected data and automatic baseline recalculation per robot instance. For each deployment, thresholds are computed from site- and robot-specific healthy data rather than reused from the lab baseline. Sensitivity can be further adjusted through configurable threshold parameters, allowing technicians or engineers to tune alert strictness based on robot type, task characteristics, and operational variability. This design ensures that differences across robot brands and configurations are addressed through data-driven recalibration and retraining, rather than assuming cross-robot generalization from the MVP model.

**2. Equipment Scale & Object Characteristics Bias**

**Issue:** Niryo Ned2 is a small educational robot with single small suction cup handling lightweight 3D blocks (few grams) with controlled shapes and surface textures.

**Impact & Limitation:** Fundamentally incomparable to industrial systems across multiple dimensions: equipment scale (industrial robots lift boxes weighing kilograms to tens of kilograms using foam suction systems with multiple cup arrays and industrial-grade air pumps), object characteristics (3D-printed plastic blocks versus cardboard boxes, metal parts, packaged goods with varying materials, weights, centers of gravity, and surface properties), and operational complexity (single suction point versus distributed suction arrays with complex pneumatic systems). Force dynamics, vacuum requirements, vibration patterns, motor loads, and failure modes are non-linear with scale and fundamentally different across material types. Learned patterns from controlled plastic blocks cannot transfer to diverse real-world objects with unpredictable surface conditions, weight distributions, and handling requirements.

**Scale-up Mitigation:** Establish entirely new baseline datasets on production-scale equipment handling actual production items during pilot phase. Use industrial sensors calibrated to appropriate force and pressure ranges for target object weights and materials. Acknowledge MVP as proof-of-concept only; treat production deployment as new model development with transfer learning from MVP architecture rather than direct model deployment.

**3. Lab vs. Factory Environment Bias**

**Issue:** Controlled lab conditions (stable temperature, minimal external vibration, consistent lighting, quiet environment) versus noisy, variable factory floors with ambient machinery, temperature fluctuations, electromagnetic interference, and floor vibrations.

**Impact & Limitation:** Model trained in clean conditions will produce high false positive rates in real factories where background noise levels are significantly higher. Baseline "healthy" patterns established in lab do not account for normal operational variability in production environments. Real factories have unpredictable disturbances (forklift traffic, nearby machine vibrations, HVAC cycles, shift changes) that cannot be systematically simulated in controlled lab settings.

**Scale-up Mitigation:** Collect minimum 4-6 weeks of healthy operation data in target factory environment to establish production-specific baselines. Implement adaptive thresholding that learns normal noise patterns for each installation site. Use robust anomaly detection methods less sensitive to background noise (e.g., relative deviation from local baseline rather than absolute thresholds). Deploy site-specific model tuning during installation phase.

**4. Simulated Fault & Task-Specific Bias**

**Issue:** Model trained on a single composite lab task (pick–place–convey–re-pick–stack loop) with artificially created fault scenarios within lab capabilities (e.g., heating motors with hairdryer, adjusting pressure via manual valve, introducing vibrations by shaking the table).

**Impact & Limitation:** Model cannot generalize to different operational tasks or varied production workflows. Each factory scenario (different pick-and-place sequences, product types, cycle times, process steps) requires entirely new baseline data collection and model retraining. Additionally, artificial fault scenarios created in lab do not match the sensor signatures of real-world failures such as bearing wear, seal degradation, contamination, electrical faults, or material fatigue. Many factory fault types cannot be simulated in lab environment including oil contamination, thermal cycling damage, cumulative wear from millions of cycles, corrosion, material embrittlement, or subtle mechanical loosening.

**Scale-up Mitigation:** Treat each factory deployment as unique implementation requiring site-specific baseline collection (minimum 4-6 weeks) for the specific production task. Implement continuous learning system that labels and incorporates real factory fault events as they occur. Collaborate with maintenance teams to document actual failure cases with corresponding sensor data. Build factory-specific and task-specific fault libraries over initial 6-12 month deployment period.

**5. Gradual Degradation Detection Bias**

**Issue:** Unable to properly simulate gradual equipment degradation leading to total failure. Most designed loops either work perfectly or fail immediately without observable degradation patterns in recorded data.

**Impact & Limitation:** Model may not detect slow-developing faults (most common in real factories) because training data lacks examples of progressive deterioration. The system is trained primarily on binary states (working/failed) rather than degradation trajectories. Real-world bearing wear, seal deterioration, filter clogging, and tube degradation happen over weeks or months, not within single experimental sessions, limiting the MVP's core predictive maintenance value of catching faults before catastrophic failure.

**Scale-up Mitigation:** Implement trend analysis alongside anomaly detection to track long-term drift in sensor baselines (e.g., gradual pressure decline over weeks, slowly increasing vibration amplitude). Monitor moving averages and statistical distributions over extended windows (daily, weekly, monthly) to detect degradation patterns. Establish maintenance correlation system that links sensor trends to actual component replacement events to build degradation signature library over time.

**6. Sensor Quality Bias**

**Issue:** Small, inexpensive sensors used in project (budget educational-grade equipment: MPU-6050 accelerometer, NXP MPXV6115VC6U pressure sensor) are less sensitive than industrial-grade sensors SAM would deploy on factory robots.

**Impact & Limitation:** Training data misses subtle signal nuances that higher-quality sensors would capture. Model thresholds calibrated to low-sensitivity sensors cannot leverage full capabilities of industrial equipment with higher sampling rates, better noise rejection, and finer resolution. Educational sensors have limited frequency response, dynamic range, and signal-to-noise ratio compared to industrial alternatives, meaning anomalies detectable with industrial sensors may be invisible to training equipment.

**Scale-up Mitigation:** Deploy industrial-grade sensors for production systems and collect new baseline data with higher-resolution equipment. Retrain model on high-quality sensor data during pilot phase. Validate that improved sensor sensitivity provides actionable signal improvements before full deployment. Document sensor specifications as system requirements for client installations.

**7. Robot Calibration Drift Bias**

**Issue:** Niryo robot calibrated at loop start but develops internal offset over time. Hardcoded positions become imprecise after repeated cycles due to mechanical tolerances and servo drift.

**Impact & Limitation:** Position data includes calibration drift noise that wouldn't exist in industrial robots with precise, regularly maintained calibration systems and higher mechanical rigidity. Industrial robots have regular professional calibration (weekly or monthly maintenance schedules), absolute encoders, and higher mechanical precision (repeatability within 0.1mm vs. several mm for educational robots). Learned tolerance for position variability from Niryo drift may be inappropriate for production equipment where position deviations indicate genuine mechanical problems rather than normal calibration drift.

**Scale-up Mitigation:** Establish tighter position tolerance thresholds based on industrial robot specifications during production baseline collection. Integrate with robot maintenance schedules to account for calibration events in data interpretation. Use relative position changes rather than absolute positions where possible to reduce sensitivity to calibration state. Document expected position precision for target robot models and adjust anomaly detection sensitivity accordingly.

**8. Data Labeling & Annotation Bias**

**Issue:** Fault events were documented manually using session-level and timestamped text notes rather than automated or sample-level labeling. Label precision varies across sessions, with some faults documented at minute-level resolution and others described only qualitatively at session level.

**Impact & Limitation:** Inconsistent and coarse-grained labels limit the model’s ability to learn precise fault onset, progression, and severity. The system is primarily validated on distinguishing healthy versus faulty sessions rather than detecting exact fault boundaries or degradation trajectories. Label uncertainty introduces noise into evaluation metrics and reduces confidence in fine-grained fault localization or early-warning timing.

**Scale-up Mitigation:** Implement automated labeling mechanisms in production environments, such as logging technician acknowledgements, system alert timestamps, or auxiliary signals (e.g., failed pickup counts, vision-based failure detection). Standardize labeling formats and severity scales across deployments. Use production feedback loops to continuously refine labels and improve supervised validation of anomaly detection outputs.

---

### 3.2 Explainability & Transparency

This section documents the current level of transparency and explainability in the SAM Automation MVP, as well as known limitations inherent to the chosen modeling approach.

#### Current Explainability and Transparency (MVP)

**Model-Level Transparency**
- The anomaly detection system is based on a neural-network model (autoencoder/LSTM-based), which functions as a black-box model.
- Model design choices, training approach, and configuration are documented in the Technical Design Document.
- Anomaly scores are produced based on deviation from learned healthy behavior, with alerts triggered when scores exceed predefined thresholds.
- Simple baseline approaches (threshold-based rules) are used to contextualize model behavior during development, but are not exposed as explainability tools to end users.

**Prediction-Level Transparency**
- The system provides **anomaly scores and threshold-based alerts** rather than feature-level explanations.
- Alerts indicate *that* an abnormal condition has occurred, but do not explicitly explain *which specific sensor variables* contributed most to the anomaly.
- No feature attribution, contribution analysis, or root-cause decomposition is currently available in the MVP.

**System-Level Transparency**
- All collected data (robot telemetry and external sensor signals) is stored locally in open formats (Parquet) with timestamp-based organization.
- Threshold logic, baseline definitions, and system behavior are documented in project documentation.
- The full data pipeline—from data collection to alert generation—is described in the Technical Design Document, enabling traceability and auditability at system level.

#### Transparency for Technicians (End Users)

- Alerts are presented as clear threshold exceedances or anomaly detections rather than opaque predictions.
- The system is intentionally designed as **decision-support only**: technicians interpret alerts and decide on maintenance actions independently.
- No automated or hidden decision-making affects robot behavior.

#### Known Explainability Limitations

- The current model does not provide explanations for *why* a specific alert was triggered or which sensor readings contributed most.
- The MVP does not support confidence scores, feature importance visualizations, or natural-language explanations.
- Explainability is therefore limited to alert presence and anomaly magnitude, which is appropriate for proof-of-concept validation but insufficient for production deployment.

#### Scale-up Enhancements (Planned)

For production deployment, additional explainability mechanisms would be required, including:
- Feature contribution analysis (e.g., SHAP or LIME-style methods) to identify dominant signals driving anomalies.
- Visualization of sensor contributions and temporal context for alerts.
- Model-specific explainability techniques (e.g., attention visualization for sequence models).
- Natural-language summaries to support technician understanding and trust.

These enhancements are documented as part of the production roadmap and are not implemented in the current MVP.

---

### 3.3 Security Measures

**Current Security Implementation:**

**1. Access Control:**
- **Restricted repository access:** Project data and code stored in a private GitHub repository accessible only to team members and BUas supervisors
- **Local clones:** Each team member maintains a local copy of the repository, providing distributed access
- **SSH authentication:** Secure, authenticated connection to the robot’s Raspberry Pi for data collection
- **Read-only telemetry:** Data collection does not interfere with robot control or execution

**2. Data Protection:**
- **No personal data processed:** GDPR compliance by design
- **No operational cloud infrastructure:** The MVP does not rely on cloud platforms (e.g., Azure, AWS, GCP) for data ingestion, storage, model training, or continuous retraining; all processing occurs locally
- **Version control:** Git and a private GitHub repository used for code and data versioning, not for live data processing

**3. System Robustness:**
- **Sensor failure detection:** Basic checks for out-of-range values
- **Network resilience:** UDP packets may be lost; gaps in data flagged but not handled
- **Adversarial testing:** Limited testing with injected noise or malicious inputs

**Security Limitations (Acknowledged):**
- No encryption for data at rest (local machines and repository storage)
- No formal penetration testing
- No role-based access control within the dashboard
- No formal incident response or disaster recovery procedures beyond Git-based redundancy

**Scale-up Security Requirements:**
1. Encryption: Data in transit (TLS) and at rest
2. Authentication: Multi-factor authentication for factory access
3. Network segmentation: Isolate AI system from production network (IXON IX6000 available)
4. Audit logging: Track all system access and alert responses
5. Backup strategy: Redundant storage, disaster recovery procedures
6. Cybersecurity standards: Align with IEC 62443 (industrial automation security)

---

### 3.4 Model Performance & Limitations

**Performance Targets (MVP):**
- Lead time: Seconds to minutes before task failure (allow intervention)
- Avoid false negatives: ≥60% recall on simulated faults

**Known Limitations:**

**1. Environmental Constraints:**
- Trained on lab conditions (controlled temperature, minimal vibration)
- May not perform well in high-noise factory environments, it should be validated on-site

**2. Robot Specificity:**
- Only validated on Niryo Ned2 robot
- Generalization to other brands (FANUC, ABB, KUKA) not tested

**3. Fault Coverage:**
- Limited to simulated faults (vacuum leak, clogged filter, extensive vibrations, overheating)
- Novel fault types haven't been evaluated

**4. Temporal Scope:**
- Short-term anomaly detection (real-time alerts)
- No long-term RUL (Remaining Useful Life) forecasting

**Transparency Commitment:**
- All limitations documented in final report
- Clear communication to SAM about system boundaries
- Recommendations for validation before production deployment

---

## 5. Data Ownership & Usage Rights

### 5.1 Ownership

**Data Ownership:**
- **Student team and BUas jointly own raw sensor data** (collected by students using BUas lab equipment and robots)

**Intellectual Property:**
- **Student team owns models and algorithms** (developed solely by the team for SAM's project purpose)
- **Students retain credit** for all project work and deliverables
- **SAM retains right to use** the architecture, models, and algorithms in their service offerings and client deployments (subject to project agreement)
- **BUas retains right to showcase** project insights and outcomes for institutional purposes (e.g., LinkedIn posts, educational materials, promotional content)

---

### 5.2 Usage Rights

**Internal Use (MVP):**
```
Data collected by BUas student team for SAM Automation project. 
Authorized for internal use by student team, BUas, and SAM Automation only. 
Contact: 231781@buas.nl for questions.
```

**Post-Project Use:**
- SAM can use models, architecture, and documentation for client deployments and service offerings
- SAM can extend, modify, and improve the system for production use
- BUas can use project insights and outcomes for institutional showcasing (promotional materials, educational examples, public presentations)
- Student team retains right to include project in portfolios and professional profiles
- External sharing of data or detailed technical implementation requires mutual consent from all parties

**Recommendation:** Formalize ownership and usage rights in collaboration agreement signed by student team, BUas, and SAM before project handoff.