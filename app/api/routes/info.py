from fastapi import APIRouter, Depends

from app.api.dependencies import get_registry
from app.ml.registry import ModelRegistry


router = APIRouter(tags=["Info"])


@router.get("/")
def root() -> dict:
    """Describe the API and show the available analysis endpoints."""
    return {
        "name": "MCMS API",
        "models": {
            "model1": "Crisis Type (10 classes) - /predict/crisis-type",
            "model2": "Message Type (11 classes) - /predict/message-type",
            "model3": "Urgency (3 classes) - /predict/urgency",
        },
        "extraction": {
            "location": "spaCy NER - /extract/location",
            "community": "Rule-based - /extract/community",
        },
        "combined": "/predict/full - everything in one call",
    }


@router.get("/health")
def health(registry: ModelRegistry = Depends(get_registry)) -> dict:
    """Report the device and loading status of every AI/NLP component."""
    return {
        "device": str(registry.device),
        "models": {
            "model1_crisis_type": registry.status("model1", registry.model1 is not None),
            "model2_message_type": registry.status("model2", registry.model2 is not None),
            "model3_urgency": registry.status("model3", registry.model3 is not None),
            "spacy_ner": registry.status("spacy", registry.nlp is not None),
        },
    }
