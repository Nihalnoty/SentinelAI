"""Tools for retrieving and investigating SentinelAI incidents."""

import json
from pathlib import Path

import pandas as pd

from src.evidence.incident_evidence_analyzer import (
    INCIDENTS_PATH,
    REPORTS_DIR,
    investigate,
)


INCIDENT_REQUIRED_COLUMNS = [
    "incident_id",
    "start",
    "end",
    "flagged_minutes",
    "mean_anomaly_score",
    "max_anomaly_score",
    "status",
    "interpretation",
]


def get_incident_details(incident_id: str) -> dict:
    """Retrieve the recorded details of an incident."""

    if not isinstance(incident_id, str) or not incident_id.strip():
        raise ValueError("incident_id must be a non-empty string.")

    incident_id = incident_id.strip()

    if not INCIDENTS_PATH.exists():
        raise FileNotFoundError(
            f"Incident report not found: {INCIDENTS_PATH}"
        )

    incidents = pd.read_csv(
        INCIDENTS_PATH,
        dtype={"incident_id": str},
    )

    missing_columns = [
        column
        for column in INCIDENT_REQUIRED_COLUMNS
        if column not in incidents.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Incident report is missing columns: {missing_columns}"
        )

    matches = incidents.loc[
        incidents["incident_id"] == incident_id
    ]

    if matches.empty:
        raise ValueError(f"Incident ID not found: {incident_id}")

    if len(matches) > 1:
        raise ValueError(
            f"Duplicate records found for incident: {incident_id}"
        )

    incident = matches.iloc[0]

    return {
        "incident_id": str(incident["incident_id"]),
        "start": str(incident["start"]),
        "end": str(incident["end"]),
        "flagged_minutes": int(incident["flagged_minutes"]),
        "mean_anomaly_score": float(
            incident["mean_anomaly_score"]
        ),
        "max_anomaly_score": float(
            incident["max_anomaly_score"]
        ),
        "status": str(incident["status"]),
        "interpretation": str(incident["interpretation"]),
    }


def analyze_incident_evidence(incident_id: str) -> dict:
    """Run the existing evidence analyzer and return its JSON report."""

    # Validate the ID before running the analysis.
    get_incident_details(incident_id)

    # Reuse the existing analyzer instead of duplicating its logic.
    investigate(incident_id)

    report_path = (
        REPORTS_DIR / f"{incident_id}_evidence.json"
    )

    if not report_path.exists():
        raise FileNotFoundError(
            f"Evidence report was not created: {report_path}"
        )

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if report.get("incident", {}).get("incident_id") != incident_id:
        raise ValueError(
            "Evidence report incident ID does not match the request."
        )

    return report