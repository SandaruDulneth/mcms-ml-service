from fastapi import APIRouter, Depends

from app.api.dependencies import get_pipeline, require_component, require_text
from app.schemas.requests import TextRequest
from app.services.pipeline_service import PipelineService


router = APIRouter(prefix="/predict", tags=["Predict"])


@router.post("/crisis-type")
def predict_crisis_type(request: TextRequest, pipeline: PipelineService = Depends(get_pipeline)) -> dict:
    """Classify the submitted text into a crisis/disaster category."""
    text = require_text(request.text)
    require_component(pipeline.registry, "model1", pipeline.registry.model1)
    return pipeline.timed(text, pipeline.predictions.predict_crisis_type)


@router.post("/message-type")
def predict_message_type(request: TextRequest, pipeline: PipelineService = Depends(get_pipeline)) -> dict:
    """Classify the humanitarian purpose of the submitted message."""
    text = require_text(request.text)
    require_component(pipeline.registry, "model2", pipeline.registry.model2)
    return pipeline.timed(text, pipeline.predictions.predict_message_type)


@router.post("/urgency")
def predict_urgency(request: TextRequest, pipeline: PipelineService = Depends(get_pipeline)) -> dict:
    """Predict whether the submitted report has High, Medium, or Low urgency."""
    text = require_text(request.text)
    require_component(pipeline.registry, "model3", pipeline.registry.model3)
    return pipeline.timed(text, pipeline.predictions.predict_urgency)


@router.post("/full")
def predict_full(request: TextRequest, pipeline: PipelineService = Depends(get_pipeline)) -> dict:
    """Run all classifiers and extractors and return one combined response."""
    return pipeline.predict_full(require_text(request.text))
