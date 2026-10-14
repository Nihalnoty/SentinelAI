"""Tools for retrieving and investigating SentinelAI incidents."""

import json

import pandas as pd

from src.evidence.incident_evidence_analyzer import (
    INCIDENTS_PATH,
    REPORTS_DIR,
    investigate,
)

from src.schemas.incident_schemas import (
    IncidentRequest,
    IncidentDetails,
    ToolResult,
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
    """Retrieve and validate the recorded details of an incident."""

    request = IncidentRequest(incident_id=incident_id)
    incident_id = request.incident_id

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

    details = IncidentDetails(
        incident_id=str(incident["incident_id"]),
        start=incident["start"],
        end=incident["end"],
        flagged_minutes=int(incident["flagged_minutes"]),
        mean_anomaly_score=float(
            incident["mean_anomaly_score"]
        ),
        max_anomaly_score=float(
            incident["max_anomaly_score"]
        ),
        status=str(incident["status"]),
        interpretation=str(incident["interpretation"]),
    )

    return details.model_dump(mode="json")


def analyze_incident_evidence(incident_id: str) -> dict:
    """Run evidence analysis and return a validated tool result."""

    request = IncidentRequest(incident_id=incident_id)
    incident_id = request.incident_id

    # Verify that the requested incident exists before analysis.
    get_incident_details(incident_id)

    # Reuse the existing evidence analyzer.
    investigate(incident_id)

    report_path = REPORTS_DIR / f"{incident_id}_evidence.json"

    if not report_path.exists():
        raise FileNotFoundError(
            f"Evidence report was not created: {report_path}"
        )

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    report_incident_id = (
        report.get("incident", {}).get("incident_id")
    )

    if report_incident_id != incident_id:
        raise ValueError(
            "Evidence report incident ID does not match the request."
        )

    result = ToolResult(
        tool_name="analyze_incident_evidence",
        incident_id=incident_id,
        data=report,
    )

    return result.model_dump(mode="json")