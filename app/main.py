import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.ml.registry import ModelRegistry
from app.services.pipeline_service import PipelineService
from app.services.translation_service import TranslationService

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load shared AI resources once at startup and release control on shutdown."""
    # Models are expensive to load, so one registry is shared by every request.
    registry = ModelRegistry(settings)
    registry.load_all()

    # Load translation service only if an API key is configured
    translation: TranslationService | None = None
    if settings.gemini_api_key:
        try:
            translation = TranslationService(settings.gemini_api_key)
            logger.info("TranslationService loaded — multilingual route available")
        except Exception as error:
            logger.warning("TranslationService failed to load: %s", error)
    else:
        logger.warning(
            "GEMINI_API_KEY not set — /predict/full/multilingual will return 503"
        )

    # app.state lets FastAPI dependencies access these shared objects safely.
    application.state.registry = registry
    application.state.pipeline = PipelineService(registry, translation)
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.app_name,
        description="Three-model NLP pipeline with location, community extraction, and multilingual support",
        version=settings.app_version,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Add all endpoints collected by app/api/router.py.
    application.include_router(api_router)
    return application


app = create_app()