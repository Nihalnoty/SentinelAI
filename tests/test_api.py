"""Automated tests for the SentinelAI API."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "SentinelAI"


def test_list_incidents():
    response = client.get("/incidents")

    assert response.status_code == 200
    assert response.json()["count"] > 0
    assert len(response.json()["incidents"]) > 0


def test_get_existing_incident():
    response = client.get("/incidents/SNT-0015")

    assert response.status_code == 200
    assert response.json()["incident_id"] == "SNT-0015"


def test_get_nonexistent_incident():
    response = client.get("/incidents/SNT-9999")

    assert response.status_code == 404


def test_invalid_detector():
    response = client.post(
        "/investigate/SNT-0015",
        json={
            "detector": "random_detector",
            "include_llm_answer": False,
        },
    )

    assert response.status_code == 422


def test_report_not_found():
    response = client.get("/reports/LSTM-9999")

    assert response.status_code == 404


def test_invalid_report_id():
    response = client.get("/reports/INVALID@ID")

    assert response.status_code == 400

def test_successful_investigation():
    response = client.post(
        "/investigate/SNT-0015",
        json={
            "detector": "isolation_forest",
            "include_llm_answer": False,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["incident_id"] == "SNT-0015"
    assert data["status"] == "investigation_completed"
    assert data["requires_human_review"] is True
    assert "evidence_report" in data["result"]
def test_missing_investigation_data_returns_503(monkeypatch):
    def simulate_missing_data(**kwargs):
        raise FileNotFoundError("Required dataset is missing.")

    monkeypatch.setattr(
        "src.api.routes.investigations.investigate_incident",
        simulate_missing_data,
    )

    response = client.post(
        "/investigate/SNT-0015",
        json={
            "detector": "isolation_forest",
            "include_llm_answer": False,
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Required investigation data is unavailable."
    )


def test_unexpected_investigation_error_returns_500(monkeypatch):
    def simulate_unexpected_error(**kwargs):
        raise RuntimeError("Simulated internal failure.")

    monkeypatch.setattr(
        "src.api.routes.investigations.investigate_incident",
        simulate_unexpected_error,
    )

    response = client.post(
        "/investigate/SNT-0015",
        json={
            "detector": "isolation_forest",
            "include_llm_answer": False,
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "The investigation could not be completed."
    )