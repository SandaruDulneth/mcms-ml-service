import time
from typing import Any, Callable

from app.ml.registry import ModelRegistry
from app.services.extraction_service import ExtractionService
from app.services.prediction_service import PredictionService


class PipelineService:
    def __init__(self, registry: ModelRegistry) -> None:
        """Build the prediction and extraction services around one model registry."""
        self.registry = registry
        self.predictions = PredictionService(registry, registry.settings)
        self.extractions = ExtractionService(registry.nlp)

    @staticmethod
    def timed(text: str, operation: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        """Run one operation and add its latency and original input to the result."""
        started_at = time.perf_counter()
        result = operation(text)
        result["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
        result["input_text"] = text[:200]
        return result

    def predict_full(self, text: str) -> dict[str, Any]:
        """Run every available analysis component and produce the full API result."""
        started_at = time.perf_counter()
        # A failed optional component becomes None, while loaded components still run.
        result = {
            "input_text": text[:200],
            "crisis_type": self.predictions.predict_crisis_type(text) if self.registry.model1 else None,
            "message_type": self.predictions.predict_message_type(text) if self.registry.model2 else None,
            "urgency": self.predictions.predict_urgency(text) if self.registry.model3 else None,
            "location_extraction": self.extractions.extract_locations(text) if self.registry.nlp else None,
            "community_extraction": self.extractions.extract_communities(text),
        }
        result["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
        result["summary"] = self._build_summary(result)
        return result

    @staticmethod
    def _build_summary(result: dict[str, Any]) -> str:
        """Convert the structured pipeline result into a short human-readable summary."""
        parts = []
        crisis = result["crisis_type"]
        message = result["message_type"]
        urgency = result["urgency"]
        location = result["location_extraction"]
        community = result["community_extraction"]

        if crisis:
            parts.append(f"Crisis: {crisis['crisis_type']} ({crisis['confidence']}%)")
        if message:
            parts.append(f"Type: {message['message_type']} ({message['confidence']}%)")
        if urgency:
            parts.append(f"Urgency: {urgency['emoji']} {urgency['urgency_level']} ({urgency['confidence']}%)")
        if location and location["locations"]:
            locations = ", ".join(item["text"] for item in location["locations"])
            parts.append(f"Location: {locations}")
        if community["affected_communities"]:
            communities = ", ".join(item["community"] for item in community["affected_communities"])
            parts.append(f"Communities: {communities}")
        return " | ".join(parts)
