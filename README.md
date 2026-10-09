# SentinelAI - Industrial Failure Intelligence System

SentinelAI is an industrial AI project designed to detect unusual equipment behaviour and support evidence-based investigation of potential failures.

## Project Goals

- Analyse industrial time-series sensor data.
- Detect unusual equipment behaviour using machine learning.
- Group anomalous observations into incident episodes.
- Investigate incidents using sensor evidence.
- Retrieve technical knowledge using Retrieval-Augmented Generation (RAG).
- Develop an AI agent that coordinates investigation tools.
- Generate evidence-based reports and recommendations.
- Evaluate system reliability and build a practical dashboard.

## Dataset

The current dataset is the Industrial Centrifuge Condition Monitoring Dataset associated with the Sulzer centrifuge unit at Arab Potash Company, Jordan.

- 525,600 observations.
- One-minute sampling intervals.
- Date range: 2022-09-05 01:00 to 2023-09-05 00:59.
- Sensors: torque, vibration, solid-end bearing temperature, and liquid-end bearing temperature.

Timestamp validation found no duplicate timestamps or irregular intervals.

Some liquid-end bearing temperature readings are zero. Their validity has not been independently established. The modelling pipeline preserves the original readings and records a separate indicator when zero readings are masked for modelling.

## Implemented Components

### Data Processing
- Raw Excel dataset loading and validation.
- Chronological ordering and Parquet conversion.
- Explicit indicators for zero readings.
- Temporal feature engineering.

### Anomaly Detection
- Four-sensor Isolation Forest baseline.
- Three-sensor baseline for comparison.
- Temporal-feature-based anomaly detection.
- Analysis of model disagreements and sensor behaviour.

### Incident Investigation
- Grouping flagged observations into incident episodes.
- Incident-level sensor summaries.
- Before/during/after evidence analysis.
- JSON evidence reports for selected incidents.

## Current Findings and Limitations

- The current incident-generation output contains 55 anomaly episodes.
- Two example incident evidence reports have passed structural validation.
- The temporal detector uses 15 features derived from torque, vibration, and solid-end bearing temperature.
- Missing values in those features are handled through median imputation.
- Verified failure labels are not currently available.

An anomaly is an investigation lead, not proof of equipment failure. The system does not yet establish confirmed failures or root causes.

## Technology Stack

- Python
- pandas
- NumPy
- scikit-learn
- PyArrow / Parquet
- openpyxl

RAG, LLM-based investigation, agent orchestration, evaluation, and dashboard technologies will be documented as those components are implemented.

## Repository Structure

```text
SentinelAI/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â””â”€â”€ processed/
â”œâ”€â”€ reports/
â”‚   â”œâ”€â”€ figures/
â”‚   â””â”€â”€ incident_evidence/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ analysis/
â”‚   â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ evidence/
â”‚   â””â”€â”€ ml/
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .gitignore
â””â”€â”€ README.md


SentinelAI is an industrial AI system designed to detect abnormal equipment behavior and support evidence-based failure investigation.



\## Project Goal



The system combines:



\- Industrial time-series data

\- Machine-learning-based anomaly detection

\- Retrieval-Augmented Generation (RAG)

\- Large Language Models (LLMs)

\- Tool calling and AI agents

\- Evidence tracking and source attribution

\- Evaluation and monitoring

\- A practical engineering dashboard



\## Planned System Flow



Industrial Sensor Data

&#x20;       â†“

Anomaly Detection

&#x20;       â†“

Incident Detection

&#x20;       â†“

Investigation Agent

&#x20;       â†“

Sensor / History / Knowledge Tools

&#x20;       â†“

RAG + Evidence Retrieval

&#x20;       â†“

Evidence Analysis

&#x20;       â†“

Possible Root Causes

&#x20;       â†“

Evidence-Based Recommendation

&#x20;       â†“

Investigation Report

&#x20;       â†“

Dashboard



\## Current Status



Project initialization.



Dataset and architecture will be validated before implementation.



\## Team



\- Nihal

\- Rayyan



\## Principles



\- Build before adding unnecessary frameworks

\- Validate datasets before committing to them

\- Measure model and system performance

\- Preserve evidence and source provenance

\- Avoid unsupported conclusions

\- Keep the system reproducible



