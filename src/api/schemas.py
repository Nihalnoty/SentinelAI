"""Request and response schemas for the SentinelAI API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.incident_schemas import (
    DetectorType,
    IncidentDetails,
)


class APIResponse(BaseModel):
    """Standard API response envelope."""

    model_config = ConfigDict(extra="forbid")

    status: str
    message: str


class HealthResponse(APIResponse):
    """API health-check response."""

    service: str
    version: str


class IncidentListResponse(BaseModel):
    """Response containing detected incidents."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    incidents: list[IncidentDetails]


class InvestigationRequest(BaseModel):
    """Options for investigating an incident."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    detector: DetectorType = "isolation_forest"
    include_llm_answer: bool = False


class InvestigationResponse(BaseModel):
    """Response containing the complete investigation."""

    model_config = ConfigDict(extra="allow")

    incident_id: str
    status: str
    requires_human_review: Literal[True] = True
    result: dict[str, Any]


class ReportResponse(BaseModel):
    """Response containing a generated investigation report."""

    model_config = ConfigDict(extra="allow")

    report_metadata: dict[str, Any]
    investigation_report: dict[str, Any]
    raw_investigation_data: dict[str, Any]
    final_safety_statement: dict[str, Any]


class ErrorResponse(BaseModel):
    """Consistent API error response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    error_code: str
    message: str
    details: dict[str, Any] | None = None