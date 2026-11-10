"""API routes for retrieving SentinelAI incidents."""

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError

from src.api.schemas import IncidentListResponse
from src.schemas.incident_schemas import DetectorType, IncidentDetails
from src.tools.incident_tools import (
    DETECTOR_REPORTS,
    DETECTOR_PREFIXES,
    INCIDENT_REQUIRED_COLUMNS,
    get_incident_details,
)


logger = logging.getLogger("sentinelai.api.incidents")

router = APIRouter(
    prefix="/incidents",
    tags=["Incidents"],
)


@router.get(
    "",
    response_model=IncidentListResponse,
)
async def list_incidents(
    detector: DetectorType | None = Query(
        default=None,
        description="Filter incidents by detector.",
    ),
):
    """List incidents from the available detector reports."""

    selected_detectors = (
        [detector] if detector else list(DETECTOR_REPORTS)
    )

    all_incidents = []

    for selected_detector in selected_detectors:

        report_path = DETECTOR_REPORTS[selected_detector]

        if not report_path.exists():
            logger.error(
                "Detector report missing: %s",
                selected_detector,
            )
            raise HTTPException(
                status_code=503,
                detail=f"Incident report unavailable for {selected_detector}.",
            )

        try:
            dataframe = pd.read_csv(
                report_path,
                dtype={"incident_id": str},
            )

            missing_columns = [
                column
                for column in INCIDENT_REQUIRED_COLUMNS
                if column not in dataframe.columns
            ]

            if missing_columns:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Invalid incident report for {selected_detector}. "
                        f"Missing columns: {missing_columns}"
                    ),
                )

            for record in dataframe.to_dict(orient="records"):

                incident_data = {
                    column: record[column]
                    for column in INCIDENT_REQUIRED_COLUMNS
                }

                incident_data["detector"] = selected_detector

                try:
                    incident = IncidentDetails.model_validate(
                        incident_data
                    )

                except ValidationError as exc:
                    logger.exception(
                        "Invalid incident record in %s",
                        selected_detector,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="An incident record failed validation.",
                    ) from exc

                all_incidents.append(incident)

        except HTTPException:
            raise

        except Exception as exc:
            logger.exception(
                "Failed to read incident report: %s",
                selected_detector,
            )
            raise HTTPException(
                status_code=500,
                detail="Unable to retrieve incident records.",
            ) from exc

    logger.info(
        "Retrieved %d incidents",
        len(all_incidents),
    )

    return IncidentListResponse(
        count=len(all_incidents),
        incidents=all_incidents,
    )


@router.get(
    "/{incident_id}",
    response_model=IncidentDetails,
)
async def get_incident(
    incident_id: str,
    detector: DetectorType | None = Query(
        default=None,
        description="Detector type. If omitted, inferred from incident ID.",
    ),
):
    """Retrieve details for one specific incident."""

    selected_detector = detector

    if selected_detector is None:
        selected_detector = next(
            (
                name
                for name, prefix in DETECTOR_PREFIXES.items()
                if incident_id.startswith(prefix)
            ),
            None,
        )

    if selected_detector is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to identify detector from incident ID.",
        )

    try:
        details = get_incident_details(
            incident_id,
            detector=selected_detector,
        )

        return IncidentDetails.model_validate(details)

    except FileNotFoundError as exc:
        logger.error("Incident report unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Incident report is currently unavailable.",
        ) from exc

    except ValueError as exc:
        message = str(exc)

        if "Incident ID not found" in message:
            raise HTTPException(
                status_code=404,
                detail="Incident not found.",
            ) from exc

        raise HTTPException(
            status_code=400,
            detail=message,
        ) from exc

    except ValidationError as exc:
        logger.exception("Incident details validation failed")
        raise HTTPException(
            status_code=500,
            detail="Retrieved incident details failed validation.",
        ) from exc

    except Exception as exc:
        logger.exception("Incident retrieval failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve incident details.",
        ) from exc