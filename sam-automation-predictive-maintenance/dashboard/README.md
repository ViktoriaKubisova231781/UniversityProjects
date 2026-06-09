# Real-Time Anomaly Detection Dashboard

## 📖 Overview

This repository hosts a full-stack Python monitoring dashboard designed to collect, analyze, and visualize data in real-time. The system aggregates metrics from **SSH connections** (robot telemetry) and **UDP sensors** (Arduino hardware data), processes them through a machine learning pipeline for anomaly detection, and presents the results in an interactive interface.

### Key Features

* **Multi-Channel Data Collection:** Unified ingestion from SSH and UDP sources.
* **ML-Driven Analysis:** Live anomaly detection using custom baseline models and filters.
* **Asynchronous Training:** Background service (`async_trainer`) to update models without blocking the UI.
* **Interactive Views:** Specialized dashboards for Live Monitoring, Historical Data, and Model Training status.

## 📂 Project Structure
```text
dashboard/
├── app.py                              # Main application entry point (UI launch)
├── config.py                           # Global configuration settings
├── requirements.txt                    # Python dependencies
├── README.txt                          # Legacy documentation
│
├── collectors/                         # Data ingestion layer
│   ├── ssh_data_collector_fast.py     # High-speed server metric collector
│   ├── udp_sensor_collector_raw.py    # Raw data ingestion from sensors
│   └── unified_collector.py           # Aggregator for multiple sources
│
├── core_ml/                            # Machine Learning Logic
│   ├── anomaly_detection.py           # Logic for flagging outliers
│   ├── baselines.py                   # Baseline performance metrics
│   ├── filters.py                     # Signal processing and noise reduction
│   ├── models.py                      # ML model definitions
│   └── preprocessing.py               # Data cleaning and normalization
│
├── mock/                               # Simulation data
│   └── mock_v1.py                     # Synthetic data generator for testing
│
├── services/                           # Background processes
│   ├── async_trainer.py               # Handles model retraining in background
│   └── state_manager.py               # Manages application/session state
│
├── utils/                              # Shared utilities
│   └── visualization.py               # Plotting and graphing helper functions
│
├── views/                              # UI Frontend Pages
│   ├── history.py                     # Historical data exploration
│   ├── live.py                        # Real-time monitoring dashboard
│   └── training.py                    # ML Model training performance view
│
└── tests/                              # Unit and Integration Tests
    ├── test_collectors/               # Tests for data ingestion
    ├── test_core_ml/                  # Tests for ML accuracy/logic
    ├── test_services/                 # Tests for background workers
    └── test_views/                    # Tests for UI components
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+ (Recommended)
- Network access to target SSH servers or UDP streams (for live mode)

### Installation

**1. Clone the repository:**
```bash
git clone [repository-url]
cd dashboard
```

**2. Create a Virtual Environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### Running the Dashboard

To start the main application interface:
```bash
# If using Streamlit (recommended if app.py is a Streamlit entry)
streamlit run app.py

# Standard Python execution
python app.py
```

### Configuration

Edit `config.py` to adjust:

- Target IP addresses for SSH/UDP collectors
- ML Hyperparameters (thresholds, window sizes)
- Toggle between Live Mode and Mock Mode (using `mock/mock_v1.py`)
