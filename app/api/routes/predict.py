from fastapi import APIRouter, Depends

from app.api.dependencies import get_pipeline, require_component, require_text, require_translation
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


@router.post("/full/multilingual")
def predict_full_multilingual(request: TextRequest, pipeline: PipelineService = Depends(get_pipeline)) -> dict:
    """
    Auto-detect language, translate to English if needed, then run the full pipeline.

    Accepts input in any language (Sinhala, Tamil, Arabic, French, etc.).
    Adds translation metadata alongside the standard /predict/full response:
    - original_text       : raw input as received
    - detected_language   : e.g. Sinhala, Tamil, Arabic
    - language_code       : ISO 639-1 code e.g. si, ta, ar
    - was_translated      : true if translation was performed
    - translated_text     : English text used for model analysis
    - translation_confidence : high / medium / low

    Requires GEMINI_API_KEY to be set in .env
    """
    require_translation(pipeline)
    return pipeline.predict_full_multilingual(require_text(request.text))