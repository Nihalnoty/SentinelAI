"""Validated input and output contracts for SentinelAI incidents."""

from datetime import datetime
from math import isfinite
from typing import Literal, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class IncidentRequest(BaseModel):
    """Input for a tool that operates on a specific incident."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    incident_id: str = Field(
        min_length=1,
        description="Existing SentinelAI incident ID, e.g. SNT-0015.",
    )

    @field_validator("incident_id")
    @classmethod
    def validate_incident_id(cls, value: str) -> str:
        if not value:
            raise ValueError("incident_id must not be empty.")
        return value


class IncidentDetails(BaseModel):
    """Structured details recorded for a detected incident."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(min_length=1)
    start: datetime
    end: datetime
    flagged_minutes: int = Field(ge=1)
    mean_anomaly_score: float
    max_anomaly_score: float
    status: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)

    @field_validator("mean_anomaly_score", "max_anomaly_score")
    @classmethod
    def validate_finite_scores(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("Anomaly scores must be finite.")
        return value

    @model_validator(mode="after")
    def validate_incident(self):
        if self.end < self.start:
            raise ValueError("Incident end cannot precede incident start.")

        if self.max_anomaly_score < self.mean_anomaly_score:
            raise ValueError(
                "Maximum anomaly score cannot be below the mean score."
            )

        return self


class IncidentConclusion(BaseModel):
    """Conclusion that preserves uncertainty about equipment failures."""

    model_config = ConfigDict(extra="forbid")

    confirmed_failure: Literal[False] = False
    reason: str = Field(min_length=1)
    requires_human_review: Literal[True] = True


class SensorStatistics(BaseModel):
    """Descriptive statistics for one sensor in an analysis window."""

    model_config = ConfigDict(extra="forbid")

    valid_readings: int = Field(ge=0)
    missing_readings: int = Field(ge=0)
    mean: float | None
    minimum: float | None
    maximum: float | None
    standard_deviation: float | None

    @field_validator(
        "mean",
        "minimum",
        "maximum",
        "standard_deviation",
    )
    @classmethod
    def validate_finite_statistics(cls, value: float | None):
        if value is not None and not isfinite(value):
            raise ValueError("Sensor statistics must be finite.")
        return value

    @model_validator(mode="after")
    def validate_statistics(self):
        if self.standard_deviation is not None:
            if self.standard_deviation < 0:
                raise ValueError("Standard deviation cannot be negative.")

        if self.valid_readings == 0:
            if any(
                value is not None
                for value in (
                    self.mean,
                    self.minimum,
                    self.maximum,
                    self.standard_deviation,
                )
            ):
                raise ValueError(
                    "Statistics must be null when there are no valid readings."
                )

        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("Minimum cannot exceed maximum.")

        if self.valid_readings > 0 and self.mean is None:
            raise ValueError("Mean is required when valid readings exist.")

        if self.missing_readings < 0:
            raise ValueError("Missing readings cannot be negative.")

        return self


class AnalysisWindow(BaseModel):
    """Sensor measurements for a time window."""

    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime
    observations: int = Field(ge=0)
    expected_minute_observations: int = Field(ge=0)
    sensors: dict[str, SensorStatistics] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_window(self):
        if self.end < self.start:
            raise ValueError("Analysis-window end precedes its start.")

        for name, stats in self.sensors.items():
            if stats.valid_readings + stats.missing_readings != self.observations:
                raise ValueError(
                    f"Sensor {name!r} reading counts do not match "
                    "the window observation count."
                )

        return self


class SensorComparison(BaseModel):
    """Comparison of sensor means across incident windows."""

    model_config = ConfigDict(extra="forbid")

    before_mean: float | None
    during_mean: float | None
    after_mean: float | None
    during_vs_before_percent: float | None
    after_vs_during_percent: float | None

    @field_validator(
        "before_mean",
        "during_mean",
        "after_mean",
        "during_vs_before_percent",
        "after_vs_during_percent",
    )
    @classmethod
    def validate_finite_comparison(cls, value: float | None):
        if value is not None and not isfinite(value):
            raise ValueError("Comparison values must be finite.")
        return value


class IncidentEvidenceReport(BaseModel):
    """Validated report produced by the incident evidence analyzer."""

    model_config = ConfigDict(extra="forbid")

    incident: IncidentDetails
    analysis_windows: dict[str, AnalysisWindow]
    sensor_comparison: dict[str, SensorComparison]
    within_incident_blocks: list[dict[str, Any]]
    findings: list[str] = Field(min_length=1)
    conclusion: IncidentConclusion

    @model_validator(mode="after")
    def validate_report(self):
        required_windows = {"before", "during", "after"}

        if set(self.analysis_windows) != required_windows:
            raise ValueError(
                "Evidence report must contain before, during, and after windows."
            )

        if not self.within_incident_blocks:
            # Empty blocks can be valid when no sensor observations exist,
            # so this is intentionally not treated as a validation failure.
            pass

        for finding in self.findings:
            if not finding.strip():
                raise ValueError("Findings cannot contain empty strings.")

        return self


class ToolResult(BaseModel):
    """Standard envelope for a successful tool execution."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    incident_id: str = Field(min_length=1)
    status: Literal["success"] = "success"
    data: dict[str, Any]