"""SentinelAI incident investigation orchestration.

Combines deterministic sensor analysis, RAG technical evidence,
and an optional joint Gemini investigation.

The LLM cannot confirm equipment failure or root cause.
Human review is always required.
"""

import json
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.rag.llm_adapter import generate_response
from src.schemas.incident_schemas import (
    IncidentConclusion,
    IncidentEvidenceReport,
    IncidentRequest,
)
from src.tools.incident_tools import (
    analyze_incident_evidence,
    get_incident_details,
)
from src.tools.rag_evidence_tool import retrieve_maintenance_evidence


DetectorType = Literal[
    "isolation_forest",
    "lstm",
    "temporal_isolation_forest",
]


class InvestigationAssessment(BaseModel):
    """Deterministic assessment based on validated evidence."""

    model_config = ConfigDict(extra="forbid")

    observed_changes: list[str]
    data_quality_warnings: list[str]
    evidence_limitations: list[str]
    recommended_follow_up_checks: list[str]

    evidence_sufficiency: Literal[
        "insufficient_for_root_cause"
    ] = "insufficient_for_root_cause"

    evidence_conflicts: list[str] = Field(default_factory=list)

    conflict_assessment_status: Literal[
        "structured_claims_checked",
        "no_structured_claims_available",
    ] = "no_structured_claims_available"

    root_cause_confirmed: Literal[False] = False
    human_review_required: Literal[True] = True


class InvestigationResult(BaseModel):
    """Complete SentinelAI investigation result."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(min_length=1)
    status: str = Field(min_length=1)

    incident_details: dict[str, Any]
    evidence_report: dict[str, Any]
    maintenance_evidence: dict[str, Any]

    assessment: InvestigationAssessment
    conclusion: IncidentConclusion

    requires_human_review: bool = True

    joint_investigation: dict[str, Any] | None = None


def _detect_structured_evidence_conflicts(
    retrieved_items: list[dict[str, Any]],
) -> tuple[list[str], bool]:
    """Detect conflicts only from explicit structured claims."""

    claims_by_key: dict[str, list[tuple[str, str]]] = {}

    for item in retrieved_items:
        if not isinstance(item, dict):
            continue

        key = item.get("claim_key")

        if not isinstance(key, str) or not key.strip():
            continue

        if "claim_value" not in item:
            continue

        value = item.get("claim_value")

        if value is None:
            continue

        source = item.get("source", "unknown source")

        claims_by_key.setdefault(
            key.strip().casefold(), []
        ).append(
            (str(source), str(value).strip().casefold())
        )

    conflicts = []

    for key, claims in sorted(claims_by_key.items()):
        distinct_values = sorted({value for _, value in claims})

        if len(distinct_values) > 1:
            details = "; ".join(
                f"{source} asserts {value!r}"
                for source, value in claims
            )

            conflicts.append(
                f"Conflicting structured claims for {key!r}: "
                f"{details}. Resolve the discrepancy against "
                "authoritative records before use. "
                "Human review is required."
            )

    return conflicts, bool(claims_by_key)


def _build_assessment(
    evidence: IncidentEvidenceReport,
    maintenance_evidence: dict[str, Any],
) -> InvestigationAssessment:
    """Build the deterministic assessment before calling Gemini."""

    observed_changes = []
    data_quality_warnings = []
    evidence_limitations = []

    for sensor, comparison in evidence.sensor_comparison.items():

        before = comparison.before_mean
        during = comparison.during_mean
        after = comparison.after_mean
        change = comparison.during_vs_before_percent

        if during is None:
            observed_changes.append(
                f"{sensor}: no valid readings during the incident."
            )

        elif before is None:
            observed_changes.append(
                f"{sensor}: during mean={during}; "
                "pre-incident mean unavailable."
            )

        elif change is not None:
            observed_changes.append(
                f"{sensor}: mean changed by {change:+.2f}% "
                "compared with the pre-incident window."
            )

        else:
            observed_changes.append(
                f"{sensor}: before mean={before}, "
                f"during mean={during}."
            )

        if during is not None and after is not None:
            observed_changes.append(
                f"{sensor}: post-incident mean={after}. "
                "This does not independently prove recovery."
            )

    for window_name, window in evidence.analysis_windows.items():

        if window.observations < window.expected_minute_observations:
            data_quality_warnings.append(
                f"{window_name} window has {window.observations} "
                f"observations; expected "
                f"{window.expected_minute_observations}."
            )

        for sensor, stats in window.sensors.items():

            if stats.availability != "available":
                data_quality_warnings.append(
                    f"{sensor} is {stats.availability} in "
                    f"the {window_name} window."
                )

    evidence_limitations.extend([
        "Anomaly detection identifies unusual patterns. "
        "It does not independently confirm equipment failure.",

        "No verified fault labels or maintenance-event records "
        "were provided to establish the root cause.",

        "Sensor measurements alone cannot distinguish every "
        "possible operating-state or instrumentation change.",
    ])

    retrieved_items = maintenance_evidence.get("evidence", [])

    conflicts, structured_claims_available = (
        _detect_structured_evidence_conflicts(retrieved_items)
    )

    if not structured_claims_available:
        evidence_limitations.append(
            "Automated contradiction checking was limited because "
            "retrieved passages did not contain comparable structured "
            "claim keys and values."
        )

    if conflicts:
        evidence_limitations.append(
            "Conflicting structured claims were detected. "
            "The system has not selected a winning claim."
        )

    if not retrieved_items:
        evidence_limitations.append(
            "No maintenance-document passages were retrieved."
        )

    else:
        synthetic_sources = sorted({
            item.get("source", "unknown")
            for item in retrieved_items
            if item.get("is_synthetic") is True
        })

        unverified_sources = sorted({
            item.get("source", "unknown")
            for item in retrieved_items
            if item.get("source_quality") == "unverified"
        })

        if synthetic_sources:
            evidence_limitations.append(
                "Synthetic testing material was retrieved from: "
                + ", ".join(synthetic_sources)
                + ". It must not be treated as authoritative "
                "maintenance guidance."
            )

        if unverified_sources:
            evidence_limitations.append(
                "The authority of these sources remains unverified: "
                + ", ".join(unverified_sources)
                + ". Verify against approved site procedures "
                "or manufacturer documentation."
            )

        evidence_limitations.append(
            "Retrieved passages are supporting information only. "
            "Similarity scores do not establish correctness "
            "or root cause."
        )

    recommended_checks = [
        "Review equipment operating logs and process setpoints "
        "during the incident.",

        "Compare timestamps with operator notes, alarms, "
        "shutdowns, load changes, and maintenance records.",

        "Verify sensor readings against original historian "
        "or control-system records.",

        "Have qualified personnel verify retrieved references "
        "against approved procedures before maintenance decisions.",
    ]

    if data_quality_warnings:
        recommended_checks.append(
            "Resolve or document missing readings and incomplete "
            "windows before drawing conclusions."
        )

    return InvestigationAssessment(
        observed_changes=observed_changes,
        data_quality_warnings=data_quality_warnings,
        evidence_limitations=evidence_limitations,
        recommended_follow_up_checks=recommended_checks,
        evidence_conflicts=conflicts,
        conflict_assessment_status=(
            "structured_claims_checked"
            if structured_claims_available
            else "no_structured_claims_available"
        ),
        root_cause_confirmed=False,
        human_review_required=True,
    )
def _prepare_joint_investigation_prompt(
    incident_id: str,
    incident_details: dict[str, Any],
    evidence_report: dict[str, Any],
    maintenance_evidence: dict[str, Any],
    assessment: dict[str, Any],
) -> str:
    """Prepare one prompt containing both sensor and RAG evidence."""

    sensor_data = json.dumps(
        evidence_report,
        indent=2,
        default=str,
    )

    technical_data = json.dumps(
        maintenance_evidence,
        indent=2,
        default=str,
    )

    assessment_data = json.dumps(
        assessment,
        indent=2,
        default=str,
    )

    incident_data = json.dumps(
        incident_details,
        indent=2,
        default=str,
    )

    return f"""
You are the evidence investigation component of SentinelAI,
an industrial failure intelligence system.

Your task is to investigate an industrial anomaly by comparing
sensor evidence with retrieved technical-document evidence.

INCIDENT ID:
{incident_id}

INCIDENT METADATA:
{incident_data}

SENSOR INVESTIGATION:
{sensor_data}

RETRIEVED TECHNICAL DOCUMENT EVIDENCE:
{technical_data}

DETERMINISTIC ASSESSMENT:
{assessment_data}

IMPORTANT INVESTIGATION RULES:

1. Examine sensor measurements before, during, and after
   the incident.

2. Identify important abnormal sensor patterns.

3. Compare sensor patterns with retrieved technical passages.

4. Clearly distinguish observed facts, document statements,
   and your own hypotheses.

5. Do not assume a retrieved document describes this exact
   centrifuge or equipment model.

6. Never treat synthetic material or unverified sources
   as authoritative manufacturer guidance.

7. Do not invent operating conditions, alarms, maintenance
   events, specifications, or sensor thresholds.

8. Sensor correlation alone does not establish root cause.

9. Do not confirm equipment failure or root cause.

10. Do not issue specific repair instructions.

CONFIDENCE CALIBRATION:

- LOW: Default confidence when evidence is indirect,
  incomplete, or based mainly on sensor patterns.

- MEDIUM: Use only when multiple independent evidence types
  provide direct corroboration for a hypothesis.

- HIGH: Use only when verified, independent records strongly
  corroborate a hypothesis.

Do not assign medium or high confidence merely because
multiple sensors changed together.

Confidence is qualitative, not a statistical probability.
If evidence is weak, use LOW.

RETURN FORMAT:

Return valid JSON only. Do not use Markdown code fences.
Use exactly these seven top-level keys:

{{
  "A_INCIDENT_OVERVIEW": {{
    "incident_id": "string",
    "summary": "string"
  }},
  "B_OBSERVED_SENSOR_ABNORMALITIES": [
    "string"
  ],
  "C_RELEVANT_TECHNICAL_DOCUMENT_EVIDENCE": [
    {{
      "source": "exact retrieved source name",
      "summary": "string",
      "applicability_limit": "string"
    }}
  ],
  "D_POSSIBLE_EXPLANATIONS": [
    {{
      "hypothesis": "string",
      "supporting_sensor_evidence": "string",
      "supporting_technical_evidence": "string",
      "contradicting_or_missing_evidence": "string",
      "confidence_level": "low",
      "why_confidence_level_selected": "string"
    }}
  ],
  "E_EVIDENCE_GAPS": [
    "string"
  ],
  "F_RECOMMENDED_INVESTIGATION_CHECKS": [
    "string"
  ],
  "G_FINAL_ASSESSMENT": {{
    "root_cause_confirmed": false,
    "equipment_failure_confirmed": false,
    "human_review_required": true,
    "assessment_summary": "string"
  }}
}}

For technical document evidence, only refer to sources
actually present in the retrieved evidence.

If no relevant technical evidence exists, return an empty
list for C_RELEVANT_TECHNICAL_DOCUMENT_EVIDENCE.

For possible explanations, identify both supporting and
contradicting or missing evidence.

The final assessment must always preserve these values:
- root_cause_confirmed: false
- equipment_failure_confirmed: false
- human_review_required: true

Return JSON only.
"""


REQUIRED_GEMINI_SECTIONS = {
    "A_INCIDENT_OVERVIEW",
    "B_OBSERVED_SENSOR_ABNORMALITIES",
    "C_RELEVANT_TECHNICAL_DOCUMENT_EVIDENCE",
    "D_POSSIBLE_EXPLANATIONS",
    "E_EVIDENCE_GAPS",
    "F_RECOMMENDED_INVESTIGATION_CHECKS",
    "G_FINAL_ASSESSMENT",
}


def _validate_joint_gemini_json(
    response: str,
    incident_id: str,
    maintenance_evidence: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse and validate Gemini's structured investigation response."""

    try:
        parsed = json.loads(response)

    except json.JSONDecodeError as exc:
        return None, f"Gemini returned invalid JSON: {exc}"

    if not isinstance(parsed, dict):
        return None, "Gemini response must be a JSON object."

    missing_sections = REQUIRED_GEMINI_SECTIONS - set(parsed.keys())

    if missing_sections:
        return None, (
            "Gemini response is missing sections: "
            + ", ".join(sorted(missing_sections))
        )

    overview = parsed["A_INCIDENT_OVERVIEW"]

    if not isinstance(overview, dict):
        return None, "Incident overview must be an object."

    if overview.get("incident_id") != incident_id:
        return None, "Gemini returned a different incident ID."

    if not isinstance(overview.get("summary"), str):
        return None, "Incident overview summary must be a string."

    for section in (
        "B_OBSERVED_SENSOR_ABNORMALITIES",
        "C_RELEVANT_TECHNICAL_DOCUMENT_EVIDENCE",
        "D_POSSIBLE_EXPLANATIONS",
        "E_EVIDENCE_GAPS",
        "F_RECOMMENDED_INVESTIGATION_CHECKS",
    ):
        if not isinstance(parsed[section], list):
            return None, f"{section} must be a list."

    for section in (
        "B_OBSERVED_SENSOR_ABNORMALITIES",
        "E_EVIDENCE_GAPS",
        "F_RECOMMENDED_INVESTIGATION_CHECKS",
    ):
        if not all(
            isinstance(item, str)
            for item in parsed[section]
        ):
            return None, f"Every item in {section} must be a string."

    retrieved_sources = {
        str(item.get("source"))
        for item in maintenance_evidence.get("evidence", [])
        if isinstance(item, dict) and item.get("source") is not None
    }

    for item in parsed["C_RELEVANT_TECHNICAL_DOCUMENT_EVIDENCE"]:

        if not isinstance(item, dict):
            return None, "Technical evidence entries must be objects."

        source = item.get("source")

        if not isinstance(source, str) or source not in retrieved_sources:
            return None, (
                "Gemini referenced a technical source that was "
                "not present in the retrieved evidence."
            )

        if not isinstance(item.get("summary"), str):
            return None, "Technical evidence summary must be a string."

        if not isinstance(item.get("applicability_limit"), str):
            return None, (
                "Technical evidence applicability_limit "
                "must be a string."
            )

    required_hypothesis_fields = [
        "hypothesis",
        "supporting_sensor_evidence",
        "supporting_technical_evidence",
        "contradicting_or_missing_evidence",
        "confidence_level",
        "why_confidence_level_selected",
    ]

    for hypothesis in parsed["D_POSSIBLE_EXPLANATIONS"]:

        if not isinstance(hypothesis, dict):
            return None, "Every possible explanation must be an object."

        for field in required_hypothesis_fields:
            if not isinstance(hypothesis.get(field), str):
                return None, (
                    f"Every hypothesis must contain a string "
                    f"field named {field}."
                )

        if hypothesis["confidence_level"] not in {
            "low",
            "medium",
            "high",
        }:
            return None, (
                "Hypothesis confidence must be low, medium, or high."
            )

    final_assessment = parsed["G_FINAL_ASSESSMENT"]

    if not isinstance(final_assessment, dict):
        return None, "Final assessment must be an object."

    if final_assessment.get("root_cause_confirmed") is not False:
        return None, "Root cause confirmation must remain false."

    if final_assessment.get("equipment_failure_confirmed") is not False:
        return None, "Equipment failure confirmation must remain false."

    if final_assessment.get("human_review_required") is not True:
        return None, "Human review must remain required."

    if not isinstance(final_assessment.get("assessment_summary"), str):
        return None, "Final assessment summary must be a string."

    return parsed, None
def _run_joint_gemini_investigation(
    incident_id: str,
    incident_details: dict[str, Any],
    evidence_report: dict[str, Any],
    maintenance_evidence: dict[str, Any],
    assessment: dict[str, Any],
    generate_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Run Gemini using sensor and retrieved technical evidence together."""

    prompt = _prepare_joint_investigation_prompt(
        incident_id=incident_id,
        incident_details=incident_details,
        evidence_report=evidence_report,
        maintenance_evidence=maintenance_evidence,
        assessment=assessment,
    )

    try:
        response = generate_fn(prompt)

        if not isinstance(response, str):
            raise TypeError(
                "Gemini response must be a string."
            )

        parsed, validation_error = _validate_joint_gemini_json(
            response=response,
            incident_id=incident_id,
            maintenance_evidence=maintenance_evidence,
        )

        if validation_error:
            return {
                "status": "validation_failed",
                "analysis": None,
                "error": validation_error,
                "root_cause_confirmed": False,
                "human_review_required": True,
            }

        return {
            "status": "success",
            "analysis": parsed,
            "root_cause_confirmed": False,
            "human_review_required": True,
            "evidence_sources": {
                "sensor_evidence_included": True,
                "rag_evidence_included": True,
            },
        }

    except Exception as exc:
        return {
            "status": "failed",
            "analysis": None,
            "error": f"{type(exc).__name__}: {exc}",
            "root_cause_confirmed": False,
            "human_review_required": True,
        }


def investigate_incident(
    incident_id: str,
    detector: DetectorType = "isolation_forest",
    *,
    include_llm_answer: bool = False,
    llm_generate_response: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """
    Run the complete SentinelAI incident investigation.

    Deterministic sensor analysis and RAG retrieval always run.

    When include_llm_answer=True, Gemini receives both evidence
    sets together for a joint investigation.
    """

    # 1. Validate incident ID.
    request = IncidentRequest(incident_id=incident_id)
    validated_id = request.incident_id

    # 2. Retrieve incident metadata.
    if detector == "isolation_forest":
        details = get_incident_details(validated_id)
    else:
        details = get_incident_details(
            validated_id,
            detector=detector,
        )

    if not isinstance(details, dict):
        raise ValueError(
            "Incident details tool returned an invalid result."
        )

    if details.get("detector") != detector:
        raise ValueError(
            "Incident details detector does not match the request."
        )

    if details.get("incident_id") != validated_id:
        raise ValueError(
            "Incident details do not match the requested incident."
        )

    # 3. Analyze sensor evidence.
    if detector == "isolation_forest":
        tool_result = analyze_incident_evidence(validated_id)
    else:
        tool_result = analyze_incident_evidence(
            validated_id,
            detector=detector,
        )

    # 4. Validate sensor analysis.
    if not isinstance(tool_result, dict):
        raise ValueError(
            "Incident evidence tool returned an invalid result."
        )

    if tool_result.get("status") != "success":
        raise ValueError(
            "Incident evidence analysis did not succeed."
        )

    if tool_result.get("incident_id") != validated_id:
        raise ValueError(
            "Evidence tool returned a different incident ID."
        )

    raw_evidence = tool_result.get("data")

    if not isinstance(raw_evidence, dict):
        raise ValueError(
            "Evidence tool returned an invalid report."
        )

    evidence = IncidentEvidenceReport.model_validate(raw_evidence)

    if evidence.incident.incident_id != validated_id:
        raise ValueError(
            "Evidence report does not match the requested incident."
        )

    if evidence.incident.detector != detector:
        raise ValueError(
            "Evidence report detector does not match the request."
        )

    # 5. Cross-check incident timestamps.
    details_start = pd.Timestamp(
        details["start"]
    ).to_pydatetime()

    details_end = pd.Timestamp(
        details["end"]
    ).to_pydatetime()

    if (
        evidence.incident.start != details_start
        or evidence.incident.end != details_end
    ):
        raise ValueError(
            "Incident metadata and sensor evidence have "
            "inconsistent timestamps."
        )

    # 6. Retrieve relevant technical evidence.
    maintenance_evidence = retrieve_maintenance_evidence(
        "Industrial centrifuge operating conditions, "
        "sensor abnormalities, vibration, torque, bearing "
        "temperature, shutdowns, and maintenance guidance."
    )

    if not isinstance(maintenance_evidence, dict):
        raise ValueError(
            "Maintenance evidence tool returned an invalid result."
        )

    retrieved_items = maintenance_evidence.get("evidence")

    if not isinstance(retrieved_items, list):
        raise ValueError(
            "Maintenance evidence must contain an evidence list."
        )

    if maintenance_evidence.get("chunks_retrieved") != len(
        retrieved_items
    ):
        raise ValueError(
            "Retrieved evidence count does not match its passages."
        )

    # 7. Build deterministic assessment.
    assessment = _build_assessment(
        evidence,
        maintenance_evidence,
    )

    evidence_dict = evidence.model_dump(mode="json")
    assessment_dict = assessment.model_dump(mode="json")

    # 8. Run joint Gemini investigation when requested.
    joint_investigation = None

    if include_llm_answer:
        generate_fn = (
            llm_generate_response or generate_response
        )

        joint_investigation = _run_joint_gemini_investigation(
            incident_id=validated_id,
            incident_details=details,
            evidence_report=evidence_dict,
            maintenance_evidence=maintenance_evidence,
            assessment=assessment_dict,
            generate_fn=generate_fn,
        )

    # 9. Assemble final result.
    result = InvestigationResult(
        incident_id=validated_id,
        status="investigation_completed",
        incident_details=details,
        evidence_report=evidence_dict,
        maintenance_evidence=maintenance_evidence,
        assessment=assessment,
        conclusion=evidence.conclusion,
        requires_human_review=True,
        joint_investigation=joint_investigation,
    )

    return result.model_dump(mode="json")