from fastapi import APIRouter, Depends

from app.api.dependencies import get_pipeline, require_component, require_text
from app.schemas.requests import TextRequest
from app.services.pipeline_service import PipelineService


router = APIRouter(prefix="/extract", tags=["Extract"])


@router.post("/location")
def extract_location(request: TextRequest, pipeline: PipelineService = Depends(get_pipeline)) -> dict:
    """Extract locations using spaCy NER and the Sri Lankan gazetteer."""
    text = require_text(request.text)
    require_component(pipeline.registry, "spacy", pipeline.registry.nlp)
    return pipeline.timed(text, pipeline.extractions.extract_locations)


@router.post("/community")
def extract_community(request: TextRequest, pipeline: PipelineService = Depends(get_pipeline)) -> dict:
    """Extract affected community groups using configured keyword rules."""
    return pipeline.timed(require_text(request.text), pipeline.extractions.extract_communities)
