"""Tests for the incident evidence-report schema."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.schemas.incident_schemas import IncidentEvidenceReport


@pytest.fixture
def valid_report():
    """Minimal report matching the evidence analyzer's output structure."""

    sensors = {
        "Torque": {
            "valid_readings": 2,
            "missing_readings": 0,
            "mean": 0.5,
            "minimum": 0.2,
            "maximum": 0.8,
            "standard_deviation": 0.3,
        }
    }

    return {
        "incident": {
            "incident_id": "SNT-0024",
            "start": "2023-08-13 13:54:00",
            "end": "2023-08-13 14:08:00",
            "flagged_minutes": 15,
            "mean_anomaly_score": 0.71,
            "max_anomaly_score": 0.72,
            "status": "requires_investigation",
            "interpretation": "anomaly_detected_not_confirmed_failure",
        },
        "analysis_windows": {
            "before": {
                "start": "2023-08-13 12:54:00",
                "end": "2023-08-13 13:53:00",
                "observations": 2,
                "expected_minute_observations": 60,
                "sensors": sensors,
            },
            "during": {
                "start": "2023-08-13 13:54:00",
                "end": "2023-08-13 14:08:00",
                "observations": 2,
                "expected_minute_observations": 15,
                "sensors": sensors,
            },
            "after": {
                "start": "2023-08-13 14:09:00",
                "end": "2023-08-13 15:08:00",
                "observations": 2,
                "expected_minute_observations": 60,
                "sensors": sensors,
            },
        },
        "sensor_comparison": {
            "Torque": {
                "before_mean": 0.5,
                "during_mean": 0.4,
                "after_mean": 0.45,
                "during_vs_before_percent": -20.0,
                "after_vs_during_percent": 12.5,
            }
        },
        "within_incident_blocks": [],
        "findings": ["Torque mean decreased during the incident."],
        "conclusion": {
            "confirmed_failure": False,
            "reason": "Sensor evidence alone does not confirm equipment failure.",
            "requires_human_review": True,
        },
    }


def test_valid_evidence_report_is_accepted(valid_report):
    report = IncidentEvidenceReport.model_validate(valid_report)

    assert report.incident.incident_id == "SNT-0024"
    assert set(report.analysis_windows) == {"before", "during", "after"}
    assert report.conclusion.confirmed_failure is False
    assert report.conclusion.requires_human_review is True


def test_missing_analysis_window_is_rejected(valid_report):
    report = deepcopy(valid_report)
    del report["analysis_windows"]["after"]

    with pytest.raises(ValidationError):
        IncidentEvidenceReport.model_validate(report)


def test_non_finite_sensor_value_is_rejected(valid_report):
    report = deepcopy(valid_report)
    report["analysis_windows"]["during"]["sensors"]["Torque"]["mean"] = float(
        "nan"
    )

    with pytest.raises(ValidationError):
        IncidentEvidenceReport.model_validate(report)


def test_inconsistent_sensor_reading_counts_are_rejected(valid_report):
    report = deepcopy(valid_report)
    report["analysis_windows"]["during"]["sensors"]["Torque"][
        "valid_readings"
    ] = 3

    with pytest.raises(ValidationError):
        IncidentEvidenceReport.model_validate(report)


def test_confirmed_failure_claim_is_rejected(valid_report):
    report = deepcopy(valid_report)
    report["conclusion"]["confirmed_failure"] = True

    with pytest.raises(ValidationError):
        IncidentEvidenceReport.model_validate(report)


def test_reversed_analysis_window_is_rejected(valid_report):
    report = deepcopy(valid_report)
    report["analysis_windows"]["during"]["end"] = "2023-08-13 13:00:00"

    with pytest.raises(ValidationError):
        IncidentEvidenceReport.model_validate(report)