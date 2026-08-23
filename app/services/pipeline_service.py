import logging
import time
from typing import Any, Callable

from app.ml.registry import ModelRegistry
from app.services.extraction_service import ExtractionService
from app.services.prediction_service import PredictionService
from app.services.translation_service import TranslationService

logger = logging.getLogger(__name__)


class PipelineService:
    def __init__(
        self,
        registry: ModelRegistry,
        translation: TranslationService | None = None,
    ) -> None:
        self.registry    = registry
        self.predictions = PredictionService(registry, registry.settings)
        self.extractions = ExtractionService(registry.nlp)
        self.translation = translation

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def timed(text: str, operation: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
        started_at = time.perf_counter()
        result = operation(text)
        result["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
        result["input_text"] = text[:200]
        return result

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def predict_full(self, text: str) -> dict[str, Any]:
        """Run all classifiers and extractors on English text."""
        started_at = time.perf_counter()
        result = {
            "input_text"          : text[:200],
            "crisis_type"         : self.predictions.predict_crisis_type(text) if self.registry.model1 else None,
            "message_type"        : self.predictions.predict_message_type(text) if self.registry.model2 else None,
            "urgency"             : self.predictions.predict_urgency(text) if self.registry.model3 else None,
            "location_extraction" : self.extractions.extract_locations(text) if self.registry.nlp else None,
            "community_extraction": self.extractions.extract_communities(text),
        }
        result["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
        result["summary"]    = self._build_summary(result)
        return result

    # ── Multilingual pipeline ─────────────────────────────────────────────────

    def predict_full_multilingual(self, text: str) -> dict[str, Any]:
        """
        Full pipeline with multilingual support.

        For non-English input:
          Step 1 — Translate to English (Gemini)
          Step 2 — Run classifiers + spaCy NER on English text
          Step 3 — Extract locations directly from original text (Gemini)
          Step 4 — Merge Step 2 + Step 3 location results (deduplicated)

        This two-pass location strategy ensures place names are captured
        even when translation loses or alters proper nouns.
        """
        started_at = time.perf_counter()

        # ── Step 1: Translate ─────────────────────────────────────────────────
        translation_meta = self.translation.detect_and_translate(text)
        english_text     = translation_meta["translated_text"]
        is_english       = translation_meta["is_english"]

        # ── Step 2: Full pipeline on English text ─────────────────────────────
        pipeline_result = self.predict_full(english_text)

        # ── Step 3: Gemini location extraction on original text ───────────────
        # Only run for non-English input — no benefit for English text
        gemini_locations: list[str] = []
        if not is_english:
            gemini_locations = self.translation.extract_locations_from_original(text)
            logger.info(
                "Gemini extracted %d location(s) from original text: %s",
                len(gemini_locations),
                gemini_locations,
            )

        # ── Step 4: Merge locations ───────────────────────────────────────────
        if gemini_locations:
            pipeline_result["location_extraction"] = self._merge_locations(
                spacy_result   = pipeline_result.get("location_extraction"),
                gemini_locations = gemini_locations,
            )

        # ── Assemble final response ───────────────────────────────────────────
        result = {
            "original_text"          : text,
            "detected_language"      : translation_meta["detected_language"],
            "language_code"          : translation_meta["language_code"],
            "was_translated"         : translation_meta["was_translated"],
            "translated_text"        : english_text,
            "translation_confidence" : translation_meta["confidence"],
            **pipeline_result,
            # Override so input_text always shows the original, not the translation
            "input_text"             : text[:200],
            "latency_ms"             : round((time.perf_counter() - started_at) * 1000, 1),
        }
        # Rebuild summary with the merged locations
        result["summary"] = self._build_summary(result)
        return result

    # ── Location merge ────────────────────────────────────────────────────────

    @staticmethod
    def _merge_locations(
        spacy_result    : dict[str, Any] | None,
        gemini_locations: list[str],
    ) -> dict[str, Any]:
        """
        Merge spaCy NER results with Gemini-extracted location names.

        spaCy results come as full objects {text, label, start, end, source}.
        Gemini results are plain strings — we wrap them in the same shape
        so the response format stays consistent for the frontend.
        Duplicates are removed case-insensitively.
        """
        existing_locations: list[dict[str, Any]] = (
            spacy_result.get("locations", []) if spacy_result else []
        )

        # Build a set of already-known names (lowercase) for deduplication
        seen: set[str] = {loc["text"].lower() for loc in existing_locations}

        # Add Gemini locations that aren't already present
        for name in gemini_locations:
            if name.lower() not in seen:
                seen.add(name.lower())
                existing_locations.append({
                    "text"  : name,
                    "label" : "GPE",
                    "start" : -1,        # no character offset for Gemini extractions
                    "end"   : -1,
                    "source": "gemini",  # clearly tagged so frontend can distinguish
                })

        return {
            "locations"     : existing_locations,
            "location_count": len(existing_locations),
            "has_location"  : len(existing_locations) > 0,
        }

    # ── Summary builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(result: dict[str, Any]) -> str:
        parts     = []
        crisis    = result.get("crisis_type")
        message   = result.get("message_type")
        urgency   = result.get("urgency")
        location  = result.get("location_extraction")
        community = result.get("community_extraction")

        if crisis:
            parts.append(f"Crisis: {crisis['crisis_type']} ({crisis['confidence']}%)")
        if message:
            parts.append(f"Type: {message['message_type']} ({message['confidence']}%)")
        if urgency:
            parts.append(f"Urgency: {urgency['emoji']} {urgency['urgency_level']} ({urgency['confidence']}%)")
        if location and location.get("locations"):
            locs = ", ".join(item["text"] for item in location["locations"])
            parts.append(f"Location: {locs}")
        if community and community.get("affected_communities"):
            comms = ", ".join(item["community"] for item in community["affected_communities"])
            parts.append(f"Communities: {comms}")

        return " | ".join(parts)