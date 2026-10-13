"""Deterministic orchestration for SentinelAI incident investigations."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.incident_schemas import (
    IncidentConclusion,
    IncidentEvidenceReport,
    IncidentRequest,
)
from src.tools.incident_tools import (
    analyze_incident_evidence,
    get_incident_details,
)


class InvestigationResult(BaseModel):
    """Structured result of an incident investigation."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    incident_details: dict[str, Any]
    evidence_report: dict[str, Any]
    conclusion: IncidentConclusion
    requires_human_review: bool = True


def investigate_incident(incident_id: str) -> dict[str, Any]:
    """Coordinate incident retrieval, analysis, and evidence validation."""

    # 1. Validate the requested incident ID.
    request = IncidentRequest(incident_id=incident_id)
    validated_id = request.incident_id

    # 2. Retrieve the existing incident metadata.
    details = get_incident_details(validated_id)

    # 3. Run the existing evidence-analysis tool.
    tool_result = analyze_incident_evidence(validated_id)

    # 4. Validate the tool-result envelope.
    if not isinstance(tool_result, dict):
        raise ValueError("Incident evidence tool returned an invalid result.")

    if tool_result.get("status") != "success":
        raise ValueError("Incident evidence analysis did not succeed.")

    if tool_result.get("incident_id") != validated_id:
        raise ValueError(
            "Tool result does not match the requested incident."
        )

    evidence = tool_result.get("data")

    if not isinstance(evidence, dict):
        raise ValueError("Evidence tool returned an invalid report.")

    # 5. Validate the complete evidence report.
    validated_evidence = IncidentEvidenceReport.model_validate(evidence)

    # 6. Ensure the report belongs to the requested incident.
    if validated_evidence.incident.incident_id != validated_id:
        raise ValueError(
            "Evidence report does not match the requested incident."
        )

    # 7. Ensure the retrieved metadata and evidence refer to the same
    # incident and time range.
    details_id = details.get("incident_id")

    if details_id != validated_id:
        raise ValueError(
            "Incident details do not match the requested incident."
        )

    if (
        validated_evidence.incident.start
        != __import__("pandas").Timestamp(details["start"]).to_pydatetime()
        or validated_evidence.incident.end
        != __import__("pandas").Timestamp(details["end"]).to_pydatetime()
    ):
        raise ValueError(
            "Incident details and evidence report have inconsistent timestamps."
        )

    # 8. Assemble the final result only after validation succeeds.
    result = InvestigationResult(
        incident_id=validated_id,
        status="investigation_completed",
        incident_details=details,
        evidence_report=validated_evidence.model_dump(mode="json"),
        conclusion=validated_evidence.conclusion,
        requires_human_review=True,
    )

    return result.model_dump(mode="json")