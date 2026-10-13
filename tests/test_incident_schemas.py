"""Tests for SentinelAI incident schemas."""

import pytest
from pydantic import ValidationError

from src.schemas.incident_schemas import (
    IncidentRequest,
    IncidentDetails,
    IncidentConclusion,
    ToolResult,
)


def test_incident_request_strips_whitespace():
    request = IncidentRequest(incident_id=" SNT-0015 ")

    assert request.incident_id == "SNT-0015"


def test_incident_request_rejects_empty_id():
    with pytest.raises(ValidationError):
        IncidentRequest(incident_id="   ")


def test_incident_request_rejects_extra_fields():
    with pytest.raises(ValidationError):
        IncidentRequest(
            incident_id="SNT-0015",
            unexpected_field="not allowed",
        )


def test_incident_details_rejects_invalid_time_range():
    with pytest.raises(ValidationError):
        IncidentDetails(
            incident_id="SNT-0015",
            start="2023-01-02T00:00:00",
            end="2023-01-01T00:00:00",
            flagged_minutes=10,
            mean_anomaly_score=-0.5,
            max_anomaly_score=-0.2,
            status="requires_investigation",
            interpretation="Anomaly detected; failure not confirmed.",
        )


def test_conclusion_requires_human_review():
    conclusion = IncidentConclusion(
        reason="Sensor measurements do not independently confirm failure."
    )

    assert conclusion.confirmed_failure is False
    assert conclusion.requires_human_review is True


def test_tool_result_uses_success_status():
    result = ToolResult(
        tool_name="get_incident_details",
        incident_id="SNT-0015",
        data={"status": "requires_investigation"},
    )

    assert result.status == "success"
    assert result.incident_id == "SNT-0015"