"""Tests for SentinelAI investigation orchestration."""

import pytest

from src.agents.investigation_agent import investigate_incident


def test_investigation_returns_structured_result():
    result = investigate_incident("SNT-0024")

    assert result["incident_id"] == "SNT-0024"
    assert result["status"] == "investigation_completed"
    assert result["incident_details"]["incident_id"] == "SNT-0024"
    assert result["evidence_report"]["incident"]["incident_id"] == "SNT-0024"


def test_investigation_requires_human_review():
    result = investigate_incident("SNT-0024")

    assert result["requires_human_review"] is True
    assert result["conclusion"]["confirmed_failure"] is False
    assert result["conclusion"]["requires_human_review"] is True


def test_investigation_rejects_empty_incident_id():
    with pytest.raises(ValueError):
        investigate_incident("")

def test_investigation_rejects_mismatched_evidence_incident(monkeypatch):
    import src.agents.investigation_agent as agent

    original = agent.analyze_incident_evidence

    def mismatched_evidence(incident_id):
        result = original(incident_id)
        result["data"]["incident"]["incident_id"] = "SNT-9999"
        return result

    monkeypatch.setattr(
        agent,
        "analyze_incident_evidence",
        mismatched_evidence,
    )

    with pytest.raises(ValueError, match="does not match"):
        agent.investigate_incident("SNT-0024")


def test_investigation_rejects_missing_analysis_window(monkeypatch):
    import src.agents.investigation_agent as agent

    original = agent.analyze_incident_evidence

    def malformed_evidence(incident_id):
        result = original(incident_id)
        del result["data"]["analysis_windows"]["after"]
        return result

    monkeypatch.setattr(
        agent,
        "analyze_incident_evidence",
        malformed_evidence,
    )

    with pytest.raises(ValueError):
        agent.investigate_incident("SNT-0024")


def test_investigation_rejects_inconsistent_timestamps(monkeypatch):
    import src.agents.investigation_agent as agent

    original = agent.analyze_incident_evidence

    def inconsistent_evidence(incident_id):
        result = original(incident_id)
        result["data"]["incident"]["start"] = "2023-08-13 13:55:00"
        return result

    monkeypatch.setattr(
        agent,
        "analyze_incident_evidence",
        inconsistent_evidence,
    )

    with pytest.raises(ValueError, match="inconsistent timestamps"):
        agent.investigate_incident("SNT-0024")