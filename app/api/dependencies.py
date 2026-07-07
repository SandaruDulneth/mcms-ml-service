from fastapi import HTTPException, Request, status

from app.ml.registry import ModelRegistry
from app.services.pipeline_service import PipelineService


def get_registry(request: Request) -> ModelRegistry:
    """Return the shared model registry created during application startup."""
    return request.app.state.registry


def get_pipeline(request: Request) -> PipelineService:
    """Return the shared pipeline service for FastAPI route injection."""
    return request.app.state.pipeline


def require_component(registry: ModelRegistry, name: str, component: object) -> None:
    """Return HTTP 503 when an endpoint's required AI component failed to load."""
    if component is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{name} not loaded: {registry.load_errors.get(name)}",
        )


def require_text(text: str) -> str:
    """Reject empty or whitespace-only report text and return the valid text."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="text field is empty")
    return text
