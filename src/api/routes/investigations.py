"""API routes for SentinelAI incident investigations."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from src.api.schemas import (
    InvestigationRequest,
    InvestigationResponse,
)
from src.agents.investigation_agent import investigate_incident


logger = logging.getLogger("sentinelai.api.investigations")

router = APIRouter(
    prefix="/investigate",
    tags=["Investigations"],
)


@router.post(
    "/{incident_id}",
    response_model=InvestigationResponse,
)
def run_investigation(
    incident_id: str,
    request: InvestigationRequest,
):
    """Execute the SentinelAI investigation pipeline."""

    logger.info(
        "Investigation requested for %s using %s",
        incident_id,
        request.detector,
    )

    try:
        result = investigate_incident(
            incident_id=incident_id,
            detector=request.detector,
            include_llm_answer=request.include_llm_answer,
        )

        response = InvestigationResponse(
            incident_id=result["incident_id"],
            status=result["status"],
            requires_human_review=True,
            result=result,
        )

        logger.info(
            "Investigation completed for %s",
            incident_id,
        )

        return response

    except FileNotFoundError as exc:
        logger.exception(
            "Required investigation data unavailable for %s",
            incident_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Required investigation data is unavailable.",
        ) from exc

    except ValidationError as exc:
        logger.exception(
            "Investigation validation failed for %s",
            incident_id,
        )
        raise HTTPException(
            status_code=422,
            detail="Investigation input or evidence validation failed.",
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

    except Exception as exc:
        logger.exception(
            "Investigation failed for %s",
            incident_id,
        )
        raise HTTPException(
            status_code=500,
            detail="The investigation could not be completed.",
        ) from exc