import logging
from typing import Any

import google.generativeai as genai

logger = logging.getLogger(__name__)

# Gemini prompt — asks for JSON only so parsing is reliable
_PROMPT_TEMPLATE = """You are a language detection and translation assistant for a disaster management system.

Analyze the following text and respond with ONLY a valid JSON object, no explanation, no markdown, no code blocks.

Text: "{text}"

Respond with exactly this JSON structure:
{{
  "detected_language": "<full language name in English, e.g. Sinhala, Tamil, Arabic, French>",
  "language_code": "<ISO 639-1 code, e.g. si, ta, ar, fr, en>",
  "is_english": <true or false>,
  "translated_text": "<English translation of the text, or the original text if already English>",
  "confidence": "<high, medium, or low>"
}}"""


class TranslationService:
    """Detects language and translates to English using the Gemini Flash API."""

    def __init__(self, api_key: str) -> None:
        """Configure the Gemini client with the provided API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("TranslationService ready — using gemini-1.5-flash")

    def detect_and_translate(self, text: str) -> dict[str, Any]:
        """
        Detect the language of the input and translate to English if needed.

        Returns a dict with:
          - detected_language  : full name (e.g. "Sinhala")
          - language_code      : ISO 639-1 code (e.g. "si")
          - is_english         : bool
          - translated_text    : English text (original if already English)
          - was_translated     : bool (False when input was already English)
          - confidence         : "high" / "medium" / "low"
        """
        import json
        import re

        prompt = _PROMPT_TEMPLATE.format(text=text.strip())

        try:
            response = self.model.generate_content(prompt)
            raw = response.text.strip()

            # Strip markdown code fences if Gemini adds them despite instructions
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            parsed: dict[str, Any] = json.loads(raw)

            return {
                "detected_language": parsed.get("detected_language", "Unknown"),
                "language_code"    : parsed.get("language_code", "??"),
                "is_english"       : bool(parsed.get("is_english", False)),
                "translated_text"  : parsed.get("translated_text", text),
                "was_translated"   : not bool(parsed.get("is_english", False)),
                "confidence"       : parsed.get("confidence", "medium"),
            }

        except json.JSONDecodeError as error:
            logger.warning("Gemini returned non-JSON response: %s", error)
            # Graceful fallback — treat as English, pass original text through
            return self._fallback(text, reason=f"JSON parse error: {error}")

        except Exception as error:
            logger.error("Gemini API error: %s", error)
            return self._fallback(text, reason=str(error))

    @staticmethod
    def _fallback(text: str, reason: str) -> dict[str, Any]:
        """Return a safe fallback response when the Gemini call fails."""
        return {
            "detected_language": "Unknown",
            "language_code"    : "??",
            "is_english"       : True,
            "translated_text"  : text,
            "was_translated"   : False,
            "confidence"       : "low",
            "fallback_reason"  : reason,
        }