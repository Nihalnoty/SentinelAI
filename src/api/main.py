"""SentinelAI FastAPI application."""

import logging

from fastapi import FastAPI

from src.api.schemas import HealthResponse

from src.api.routes.incidents import router as incidents_router
from src.api.routes.investigations import router as investigations_router
from src.api.routes.reports import router as reports_router


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("sentinelai.api")


# FastAPI application
app = FastAPI(
    title="SentinelAI API",
    description=(
        "Industrial Failure Intelligence System API. "
        "Provides incident retrieval, investigation, and reporting."
    ),
    version="1.0.0",
)


# Register API routers
app.include_router(incidents_router)
app.include_router(investigations_router)
app.include_router(reports_router)


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health_check():
    """Check whether the SentinelAI API is running."""

    logger.info("Health check requested")

    return HealthResponse(
        status="healthy",
        message="SentinelAI API is running.",
        service="SentinelAI",
        version="1.0.0",
    )