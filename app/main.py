import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

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
    registry = ModelRegistry(settings)
    registry.load_all()

    # MyMemory needs no API key — always loads successfully
    translation = TranslationService()
    logger.info("TranslationService loaded — using MyMemory (free, no key needed)")

    application.state.registry = registry
    application.state.pipeline = PipelineService(registry, translation)
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        description="Three-model NLP pipeline with multilingual support via MyMemory",
        version=settings.app_version,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(api_router)
    return application


app = create_app()