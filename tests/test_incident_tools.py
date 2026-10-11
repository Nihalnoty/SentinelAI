"""Tests for SentinelAI incident tools."""

import pytest

from src.tools.incident_tools import (
    get_incident_details,
    analyze_incident_evidence,
)


def test_get_incident_details_returns_requested_incident():
    incident = get_incident_details("SNT-0015")

    assert incident["incident_id"] == "SNT-0015"
    assert incident["start"]
    assert incident["end"]
    assert incident["flagged_minutes"] > 0


def test_get_incident_details_rejects_unknown_incident():
    with pytest.raises(ValueError, match="Incident ID not found"):
        get_incident_details("SNT-9999")


def test_get_incident_details_rejects_empty_id():
    with pytest.raises(ValueError, match="non-empty string"):
        get_incident_details("")


def test_analyze_incident_evidence_returns_report():
    report = analyze_incident_evidence("SNT-0024")

    assert report["incident"]["incident_id"] == "SNT-0024"
    assert "sensor_comparison" in report
    assert "findings" in report
    assert report["conclusion"]["confirmed_failure"] is False
    assert report["conclusion"]["requires_human_review"] is True