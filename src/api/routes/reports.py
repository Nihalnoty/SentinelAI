"""API routes for retrieving SentinelAI investigation reports."""

import json
import logging
import re

from fastapi import APIRouter, HTTPException

from src.api.schemas import ReportResponse
from src.reporting.investigation_report import REPORT_DIRECTORY


logger = logging.getLogger("sentinelai.api.reports")

router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
)


@router.get(
    "/{incident_id}",
    response_model=ReportResponse,
)
def get_investigation_report(incident_id: str):
    """Retrieve an existing investigation report."""

    if not re.fullmatch(r"[A-Za-z0-9_-]+", incident_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid incident ID format.",
        )

    report_path = (
        REPORT_DIRECTORY
        / f"{incident_id}_investigation_report.json"
    )

    logger.info("Report retrieval requested for %s", incident_id)

    if not report_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Investigation report not found.",
        )

    try:
        with report_path.open("r", encoding="utf-8") as file:
            report_data = json.load(file)

        return ReportResponse(**report_data)

    except (json.JSONDecodeError, ValueError) as exc:
        logger.exception("Invalid report data for %s", incident_id)
        raise HTTPException(
            status_code=500,
            detail="The stored investigation report is invalid.",
        ) from exc

    except OSError as exc:
        logger.exception("Unable to read report for %s", incident_id)
        raise HTTPException(
            status_code=500,
            detail="Unable to read the investigation report.",
        ) from exc