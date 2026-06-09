# Robot Predictive Maintenance Dataset Codebook

**Project:** SAM Automation - Predictive Maintenance for Industrial Robots  
**Institution:** Breda University of Applied Sciences (BUas)  
**Duration:** September 2024 - January 2025 (18 weeks)  
**Authors:** Chrislande Duterloo, Monieka Hardjosoedarmo, Viktória Kubišová, Bartosz Kudyba, Kajetan Neweś 

---

## 1. Introduction

This codebook describes the data collected for the SAM Automation predictive maintenance project. The project aims to develop an anomaly detection system for collaborative robots (cobots) used in pick-and-place operations, with a focus on detecting tooling failures before they occur.

### 1.1 Project Overview

The predictive maintenance system collects data from both internal robot sensors (via SSH/ROS) and external sensors (via Arduino and Raspberry Pi) to monitor the health of the Niryo Ned 2 robotic arm during repetitive pick-and-place tasks. The system uses a mixed-methods approach combining quantitative sensor analysis with controlled failure simulations in a laboratory environment to train anomaly detection models.

**Key Features:**
- Real-time data collection from robot internal sensors (joint positions, motor temperatures, motor voltages)
- External sensor monitoring (vacuum pressure, vibration with FFT spectrum analysis)
- Controlled fault simulation for labeled training data (vibrations, reduced suction, heat application, tape on blocks)
- Multi-file data structure (robot, sensor, unified) for flexible analysis and debugging
- Two-tier alert severity system (Amber/Red) for technician-friendly anomaly reporting

---

## 2. Data Collection Methodology

### 2.1 Collection Process

Data is collected simultaneously from **two parallel streams** that operate at different frequencies, supplemented by manual documentation:

1. **Robot Internal Stream (~2.5 Hz)**: Joint positions, motor temperatures, motor voltages, and error states collected via SSH connection to ROS topics
2. **External Sensor Stream (~10 Hz)**: Pressure readings and vibration measurements with FFT preprocessing computed on the Arduino and transmitted via UDP to the laptop
3. **Manual Logging**: Hand-written description files documenting fault scenarios, timestamps, and experimental observations

Both automated streams are timestamped using system time and merged using nearest-neighbor matching (within a 1-second tolerance window) to create the unified dataset.

### 2.2 Robot Platform and Environment

**Robot**: Niryo Ned 2 - 6-axis collaborative desktop robotic arm with Raspberry Pi 4 controller running ROS (Robot Operating System). Equipped with vacuum pump and single suction cup for pick-and-place operations. Data accessed via SSH connection to ROS topics.

**Location**: Makerspace - robotics laboratory, Breda University of Applied Sciences, Netherlands

**Setup**:
- Niryo Ned 2 mounted on stable workbench with conveyor belt and tilted platform attached to it for gravity-assisted block alignment
- Laptop connected via Ethernet cable or robot's WiFi hotspot
- Stable laboratory power supply
- Indoor environment with controlled lighting and minimal air drafts

**Why Niryo Ned 2?** Represents collaborative robots used in production, provides accessible ROS-based architecture for detailed data collection, and enables controlled fault simulations without production risk.

### 2.3 Operational Task

The robot performs repetitive pick-and-place operations that simulate industrial packaging workflows:

**Task Description:**
- **Objective**: Continuously handle objects using a palletizing-style pick-and-place workflow representative of SAM Automation use cases
- **Process**: Pick blocks from a tilted platform using vacuum suction → Place blocks onto a conveyor belt → Conveyor transports blocks → Re-pick blocks from the conveyor → Stack blocks at the destination → Release suction → Return to start position
- **Duration**: Continuous loops running for 5–60 minutes per session
- **Objects**: Small 3D-printed blocks with controlled shapes, surface textures, and weights

**Purpose:**
- Generate consistent baseline data under "healthy" operating conditions
- Create realistic scenarios where failures can be introduced and detected
- Simulate typical cobot behavior in food/pharmaceutical packaging lines

### 2.4 Fault Simulation Methodology

To create labeled training data for anomaly detection, various fault scenarios are deliberately introduced during recording sessions:

**Physical Disturbances:**
- Fist impacts: Sudden strikes to the robot table creating shock vibrations
- Table shaking: Gradual and sudden shaking to simulate structural vibrations
- Variable intensity: Small shakes, medium shakes, big shakes documented with timestamps

**Tooling Failures:**
- Reduced suction: Partially opening valve on vacuum hose to weaken grip
- Tape on blocks: Placing tape on object surfaces to prevent proper seal

**Environmental Factors:**
- Heat application: Using hair dryer to increase motor temperatures (30-60 second intervals)

**Simulated Mechanical Issues:**
- Joint drift: Modified robot movement scripts adding random position offsets

All fault scenarios are documented primarily at session level in the central `README.md` file within the data folder, with some entries including approximate timestamps derived from hand-written experimental notes.

---

## 3. File Organization and Structure

### 3.1 Directory Structure
```
data/
├── YYYYMMDD_HHMMSS/                    # Recording session folder
│   ├── robot_data_robot_YYYYMMDD_HHMMSS.parquet
│   ├── robot_data_sensor_YYYYMMDD_HHMMSS.parquet
│   └── robot_data_unified_YYYYMMDD_HHMMSS.parquet
├── README.md                           # Session-level fault descriptions and notes
├── YYYYMMDD_descriptions.txt           # Legacy fault notes (deprecated)
└── ...
```

### 3.2 Data Streams and Collection Process

**Robot Internal Stream:**
- **Source:** Niryo Ned 2 internal sensors via ROS
- **Sampling Rate:** ~2.5 Hz (one sample every ~400ms)
- **Protocol:** SSH connection to Raspberry Pi ROS topics
- **Process:** Robot sensors → ROS topics → SSH subscription → Python script timestamps → Stored as `robot_data_robot_*.parquet`

**External Sensor Stream:**
- **Source:** Pressure sensor + Accelerometer via Arduino
- **Sampling Rate:** ~10 Hz (one sample every ~100ms)
- **Protocol:** UDP packets over network
- **Process:** Sensors → Arduino (high-pass filtering + FFT preprocessing) → UDP packets to laptop → Python script timestamps → Stored as `robot_data_sensor_*.parquet`

**Unified Dataset Creation:**
- **Method:** Nearest-neighbor timestamp matching with ±1 second tolerance
- **Process:** 
  1. Load robot data and sensor data files for the same session
  2. Match each robot timestamp to nearest sensor timestamp (skip if no match within tolerance)
  3. Merge matched records into unified dataframe
  4. Calculate derived features (avg/max motor temp, avg motor voltage)
  5. Save as `robot_data_unified_*.parquet`
- **Primary Timeline:** Robot frequency (~2.5 Hz) used as baseline; sensor data (~10 Hz) matched to robot timestamps
- **Timestamp Format:** ISO 8601 with nanosecond precision (e.g., `2025-12-01 10:06:54.917059`)

### 3.3 File Naming Convention

**Format:** `robot_data_{source}_{YYYYMMDD}_{HHMMSS}.parquet`

- `{source}`: Data source identifier
  - `robot`: Robot internal sensors only
  - `sensor`: External sensors only  
  - `unified`: Combined robot + sensor data with derived features
- `{YYYYMMDD}`: Date of recording session (Year-Month-Day)
- `{HHMMSS}`: End time of recording session (Hour-Minute-Second)

**Example:** `robot_data_unified_20251112_145646.parquet`

### 3.4 File Types

#### 3.4.1 Robot Data Files (`robot_data_robot_*.parquet`)

Contains internal robot sensor readings including joint positions, motor temperatures, voltages, and system states.

- **Columns:** 24
- **Sampling Rate:** ~2.5 Hz (one sample every ~400ms)
- **Purpose:** Robot telemetry analysis, debugging

#### 3.4.2 Sensor Data Files (`robot_data_sensor_*.parquet`)

Contains external sensor readings including pressure measurements and vibration analysis with frequency spectrum decomposition (64 FFT bins from 1.6-98.4 Hz).

- **Columns:** 69
- **Sampling Rate:** ~10 Hz (one sample every ~100ms)
- **Purpose:** High-resolution vibration and pressure signal analysis

#### 3.4.3 Unified Data Files (`robot_data_unified_*.parquet`)

Combined dataset with robot and sensor data time-aligned and merged using nearest-neighbor matching, including calculated derived features (avg/max motor temp, avg motor voltage...).

- **Columns:** 92 (24 robot + 69 sensor - 1 duplicate timestamp)
- **Primary Timeline:** Robot data frequency (~2.5 Hz)
- **Purpose:** Primary dataset for anomaly detection model training

All data files mentioned above are stored in **Apache Parquet format** for efficient columnar storage, fast I/O, and smaller file sizes compared to CSV.

#### 3.4.4 Session Documentation (`README.md`)

The central, authoritative documentation for all data collection sessions. This file consolidates and replaces earlier per-session text files and provides a master inventory of recording sessions and fault scenarios.

- **Format:** Markdown
- **Content:** Mapping of timestamped session folders to fault scenarios, qualitative descriptions of induced faults, approximate timing information where available, and contextual notes on experimental conditions
- **Granularity:** Primarily session-level; some entries include approximate intra-session time ranges
- **Purpose:** Contextual interpretation of anomalous periods and approximate ground-truth labeling for model development and evaluation

#### 3.4.5 Legacy Description Files (`YYYYMMDD_descriptions.txt`)

Supplementary, hand-written notes created during early data collection sessions and retained for traceability.

- **Format:** Plain text
- **Content:** Informal descriptions of induced faults, sometimes including approximate timestamps or time ranges, and qualitative observations
- **Labeling Consistency:** Variable; some files contain timestamped events (HH:MM:SS), while others provide only session-level descriptions
- **Status:** Deprecated — all relevant information has been consolidated into the central `README.md`
- **Purpose:** Historical reference and additional context; not used as precise ground-truth labels

---

## 4. Variable Definitions

### 4.1 Robot Data Variables (24 columns)

#### 4.1.1 Temporal Variables

| Variable | Type | Description | Units | Example |
|----------|------|-------------|-------|---------|
| `timestamp` | datetime64[ns] | Timestamp of data collection | ISO 8601 format | 2025-12-01 10:06:54.917059 |

#### 4.1.2 Joint Position Variables

| Variable | Type | Description | Units | Range | Notes |
|----------|------|-------------|-------|-------|-------|
| `j1` | float64 | Joint 1 position (base rotation) | radians | -2.949 to 2.949 | ~±169° range |
| `j2` | float64 | Joint 2 position (shoulder) | radians | -1.83 to 0.61 | ~-105° to 35° |
| `j3` | float64 | Joint 3 position (elbow) | radians | -1.34 to 1.57 | ~-77° to 90° |
| `j4` | float64 | Joint 4 position (wrist 1) | radians | -2.089 to 2.089 | ~±120° range |
| `j5` | float64 | Joint 5 position (wrist 2) | radians | -1.919 to 1.922 | ~±110° range |
| `j6` | float64 | Joint 6 position (wrist 3/tool) | radians | -2.53 to 2.53 | ~±145° range |

**Note:** Joint positions are reported in radians as per Niryo Ned2 specifications. Values outside these ranges indicate potential calibration issues or data collection errors.

#### 4.1.3 Temperature Variables

| Variable | Type | Description | Units | Typical Range (P5–P95) | Notes |
|----------|------|-------------|-------|------------------------|-------|
| `rpi_temp` | float64 | Raspberry Pi CPU temperature | °C | 43–54 | Embedded controller temperature |
| `motor1_temp` | float64 | Joint 1 motor temperature | °C | 43–54 | Base motor |
| `motor2_temp` | float64 | Joint 2 motor temperature | °C | 43–52 | Shoulder motor |
| `motor3_temp` | float64 | Joint 3 motor temperature | °C | 44–53 | Elbow motor |
| `motor4_temp` | float64 | Joint 4 motor temperature | °C | 35–42 | Wrist 1 motor |
| `motor5_temp` | float64 | Joint 5 motor temperature | °C | 37–42 | Wrist 2 motor |
| `motor6_temp` | float64 | Joint 6 motor temperature | °C | 22–25 | Wrist 3 motor |

**Note:** Robot operating temperature range: 5-45°C (ambient). Motor temperatures may exceed ambient during operation. Ranges correspond to the 5th–95th percentile of the full dataset collected in the lab, including both healthy operation and controlled fault simulations. As a result, these ranges describe the empirical distribution of observed values rather than strict healthy-only baselines. Healthy operation typically occupies a narrower sub-range within these bounds.

#### 4.1.4 Derived Temperature Features

| Variable | Type | Description | Units | Calculation | Notes |
|----------|------|-------------|-------|-------------|-------|
| `avg_motor_temp` | float64 | Average temperature across all motors | °C | Mean of motor1-6_temp | Overall thermal load indicator |
| `max_motor_temp` | float64 | Maximum temperature among all motors | °C | Max of motor1-6_temp | Hottest motor identifier |

#### 4.1.5 Voltage Variables

| Variable | Type | Description | Units | Typical Range | Notes |
|----------|------|-------------|-------|---------------|-------|
| `motor1_voltage` | float64 | Joint 1 motor voltage | V | 11.90–11.98 | Base motor power (12V DC system) |
| `motor2_voltage` | float64 | Joint 2 motor voltage | V | 11.92–12.00 | Shoulder motor power (12V DC system) |
| `motor3_voltage` | float64 | Joint 3 motor voltage | V | 11.91–12.00 | Elbow motor power (12V DC system) |
| `motor4_voltage` | float64 | Joint 4 motor voltage | V | 11.20–11.30 | Wrist 1 motor power (12V DC system) |
| `motor5_voltage` | float64 | Joint 5 motor voltage | V | 11.10–11.20 | Wrist 2 motor power (12V DC system) |
| `motor6_voltage` | float64 | Joint 6 motor voltage | V | 4.90–5.00 | Wrist 3 motor power (5V DC system) |

**Note:** Robot power supply: DC 12V - 7A; 5V - 7A. Motors 1-5 operate on the 12V system, while motor 6 operates on the 5V system. I/O power supply: 5V. Voltage ranges reflect the empirical 5th–95th percentile distribution of the full lab dataset rather than nominal power supply limits. Values outside typical operating patterns may indicate load changes, electrical instability, or fault conditions.

#### 4.1.6 Derived Voltage Features

| Variable | Type | Description | Units | Calculation | Notes |
|----------|------|-------------|-------|-------------|-------|
| `avg_motor_voltage` | float64 | Average voltage across all motors | V | Mean of motor1-6_voltage | Overall power consumption indicator |

#### 4.1.7 System State Variables

| Variable | Type | Description | Units | Values | Notes |
|----------|------|-------------|-------|--------|-------|
| `error_state` | int64 | Robot error state flag | Binary | 0 = Normal, 1 = Error | System health indicator |

---

### 4.2 Sensor Data Variables (69 columns)

#### 4.2.1 Temporal Variables

| Variable | Type | Description | Units | Example |
|----------|------|-------------|-------|---------|
| `timestamp` | datetime64[ns] | Timestamp of sensor reading | ISO 8601 format | 2025-12-01 10:06:54.118721 |

#### 4.2.2 Pressure Variables

| Variable | Type | Description | Units | Typical Range (P5–P95) | Notes |
|----------|------|-------------|-------|------------------------|-------|
| `pressure` | int64 | Vacuum pressure sensor reading | Raw ADC units | 30,103–32,332 | Higher values indicate stronger suction |

**Interpretation:**
- Higher values indicate stronger suction
- Sudden drops may indicate loss of contact or leaks
- Sustained low values suggest tooling failure

**Note:**  
Pressure values are reported as **raw ADC readings** from the vacuum pressure sensor and are not converted to calibrated physical units (e.g., kPa). The reported range corresponds to the **5th–95th percentile of the full lab dataset**, including both healthy operation and controlled fault simulations. During data quality analysis, indications of potential scaling or offset effects were observed in the pressure signal; however, these effects were not investigated further, as the primary analytical focus was on **relative pressure changes over time** (e.g., sudden drops or sustained declines) rather than absolute pressure values. This approach is sufficient for detecting suction loss, leaks, and tooling-related anomalies within the scope of the MVP.

#### 4.2.3 Derived Vibration Summary Statistics

| Variable | Type | Description | Units | Notes |
|----------|------|-------------|-------|-------|
| `vib_mean` | float64 | Mean vibration amplitude | g (relative) | Average vibration intensity |
| `vib_max` | float64 | Maximum vibration amplitude | g (relative) | Peak vibration during measurement window |
| `vib_min` | float64 | Minimum vibration amplitude | g (relative) | Baseline vibration level |
| `dominant_freq` | float64 | Dominant frequency component | Hz | Primary vibration frequency |

**Interpretation:**
- `vib_mean`: Indicates overall vibration level; increases suggest mechanical issues
- `vib_max`: Detects sudden shocks or impacts
- `vib_min`: Baseline noise level
- `dominant_freq`: Identifies characteristic failure frequencies

#### 4.2.4 Vibration Frequency Spectrum (64 bins)

Frequency spectrum decomposition using Fast Fourier Transform (FFT). Each variable represents the vibration amplitude at a specific frequency.

| Variable Pattern | Type | Description | Units | Frequency Range |
|-----------------|------|-------------|-------|-----------------|
| `vib_{freq}` | float64 | Vibration amplitude at {freq} Hz | g (relative) | 1.6-98.4 Hz |

**Complete List of Frequency Bins (Hz):**

1.6, 3.1, 4.7, 6.3, 7.8, 9.4, 10.9, 12.5, 14.1, 15.6, 17.2, 18.7, 20.3, 21.9, 23.4, 25.0, 26.6, 28.1, 29.7, 31.2, 32.8, 34.4, 35.9, 37.5, 39.1, 40.6, 42.2, 43.7, 45.3, 46.9, 48.4, 50.0, 51.6, 53.1, 54.7, 56.2, 57.8, 59.4, 60.9, 62.5, 64.1, 65.6, 67.2, 68.8, 70.3, 71.9, 73.4, 75.0, 76.6, 78.1, 79.7, 81.3, 82.8, 84.4, 85.9, 87.5, 89.1, 90.6, 92.2, 93.8, 95.3, 96.9, 98.4

**Example Variables:**
- `vib_1.6`: Vibration amplitude at 1.6 Hz (very low frequency oscillations)
- `vib_25.0`: Vibration amplitude at 25 Hz (common mechanical frequency)
- `vib_50.0`: Vibration amplitude at 50 Hz (electrical interference, motor frequency)
- `vib_98.4`: Vibration amplitude at 98.4 Hz (high frequency components)

**Interpretation:**
- Low frequencies (< 10 Hz): Structural vibrations, impacts, movement-related oscillations
- Mid frequencies (10-50 Hz): Motor vibrations, mechanical resonances, bearing issues
- High frequencies (> 50 Hz): Electrical noise, high-speed component vibrations
- Sudden increases in specific bands may indicate component degradation

---

### 4.3 Unified Data Variables (92 columns)

The unified dataset combines all robot data variables (24) and sensor data variables (69), minus one timestamp column (merged on timestamp).

**Total Columns:** 92
- 1 timestamp column
- 23 robot-specific variables
- 68 sensor-specific variables

All variable definitions from Sections 4.1 and 4.2 apply. Data is time-aligned using nearest-neighbor interpolation at the robot data sampling frequency. (~2.5 Hz).

---

## 5. Additional Resources

This codebook serves as a reference guide for understanding the variables and structure of the SAM Automation predictive maintenance dataset. It is part of a comprehensive documentation suite developed during the 18-week project at Breda University of Applied Sciences.

For a complete understanding of the project, readers are encouraged to consult the complementary documentation. The **Technical Design Document** provides comprehensive coverage of all technical aspects including hardware configuration, sensor integration, data processing pipelines, model architecture, preprocessing procedures, software implementation, dashboard design, fault simulation design, system architecture, and deployment planning. The **Data Quality Report** addresses data validation procedures, known limitations, and quality assessment metrics.

Project governance is documented in the **Data Management Plan**, which outlines data handling protocols, storage procedures, access control, and retention policies. Legal and ethical considerations, including privacy, security, and compliance requirements, are covered in the **Framework Assessment**. 

Project context and strategic objectives are documented in the **Business Requirements Document** and **Research Proposal**, which outline the problem statement, methodology, and expected outcomes. Implementation code, usage examples, and practical guidelines are available in the **project GitHub repository**.

For questions, clarifications, or access to additional project materials, please contact the project team at Breda University of Applied Sciences or reach out to the industry partner, SAM Automation.
