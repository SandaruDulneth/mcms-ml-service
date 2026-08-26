import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# MyMemory API — free, no API key required
# Docs: https://mymemory.translated.net/doc/spec.php
MYMEMORY_URL = "https://api.mymemory.translated.net/get"

# Language codes for MyMemory
# MyMemory uses standard ISO 639-1 codes with optional region
LANG_CODES = {
    "si": "si",      # Sinhala
    "ta": "ta",      # Tamil
    "ar": "ar",      # Arabic
    "fr": "fr",      # French
    "de": "de",      # German
    "es": "es",      # Spanish
    "zh": "zh",      # Chinese
    "ja": "ja",      # Japanese
    "ko": "ko",      # Korean
    "pt": "pt",      # Portuguese
    "ru": "ru",      # Russian
    "hi": "hi",      # Hindi
    "ur": "ur",      # Urdu
    "bn": "bn",      # Bengali
    "en": "en",      # English
}

# Language detection patterns — check script/charset first (fastest)
# then fall back to MyMemory's detected language
SCRIPT_RANGES = {
    "Sinhala": (0x0D80, 0x0DFF),
    "Tamil"  : (0x0B80, 0x0BFF),
    "Arabic" : (0x0600, 0x06FF),
    "Devanagari": (0x0900, 0x097F),  # Hindi, Nepali
    "Bengali": (0x0980, 0x09FF),
    "Chinese": (0x4E00, 0x9FFF),
    "Japanese Hiragana": (0x3040, 0x309F),
    "Korean" : (0xAC00, 0xD7AF),
}

SCRIPT_TO_CODE = {
    "Sinhala"   : "si",
    "Tamil"     : "ta",
    "Arabic"    : "ar",
    "Devanagari": "hi",
    "Bengali"   : "bn",
    "Chinese"   : "zh",
    "Japanese Hiragana": "ja",
    "Korean"    : "ko",
}

# Sri Lanka place name mappings — used as a fallback after translation
# to correct common mistranslations of Sri Lankan proper nouns
SL_PLACE_CORRECTIONS = {
    "city of gems"       : "Ratnapura",
    "gem city"           : "Ratnapura",
    "new city"           : "Nuwara Eliya",
    "city of lights"     : "Nuwara Eliya",
    "new eliya"          : "Nuwara Eliya",
    "the city"           : "",            # too vague — remove
    "kande"              : "Kandy",
    "gal le"             : "Galle",
}


class TranslationService:
    """
    Language detection and translation using MyMemory API (free, no key needed).
    """

    def __init__(self) -> None:
        logger.info("TranslationService ready — using MyMemory API (free)")

    # ── Public methods ────────────────────────────────────────────────────────

    def detect_and_translate(self, text: str) -> dict[str, Any]:
        """
        Detect language via script analysis then translate to English via MyMemory.
        Returns the standard translation metadata dict.
        """
        text = text.strip()

        # Step 1 — detect language from Unicode script ranges
        lang_code, lang_name = self._detect_language(text)

        # Step 2 — if already English skip translation
        if lang_code == "en":
            return {
                "detected_language": "English",
                "language_code"    : "en",
                "is_english"       : True,
                "translated_text"  : text,
                "was_translated"   : False,
                "confidence"       : "high",
            }

        # Step 3 — translate via MyMemory
        translated = self._translate_mymemory(text, lang_code)

        # Step 4 — apply place name corrections
        translated = self._apply_corrections(translated)

        return {
            "detected_language": lang_name,
            "language_code"    : lang_code,
            "is_english"       : False,
            "translated_text"  : translated,
            "was_translated"   : True,
            "confidence"       : "medium",
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_language(self, text: str) -> tuple[str, str]:
        """
        Detect language by checking Unicode script ranges.
        Returns (language_code, language_name).
        Falls back to "en" if no non-Latin script detected.
        """
        for char in text:
            cp = ord(char)
            for script_name, (start, end) in SCRIPT_RANGES.items():
                if start <= cp <= end:
                    code = SCRIPT_TO_CODE.get(script_name, "und")
                    return code, script_name
        return "en", "English"

    def _translate_mymemory(self, text: str, source_lang: str) -> str:
        """
        Call MyMemory REST API to translate text to English.
        MyMemory has a 500 char limit per request — splits longer text.
        """
        chunks = self._split_text(text, 500)
        translated_parts = []

        for chunk in chunks:
            params = urllib.parse.urlencode({
                "q"   : chunk,
                "langpair": f"{source_lang}|en",
                "de"  : "mcms-disaster-system@university.ac.uk",
            })
            url = f"{MYMEMORY_URL}?{params}"

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MCMS/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    translated = data.get("responseData", {}).get("translatedText", chunk)
                    if "MYMEMORY WARNING" in str(translated):
                        logger.warning("MyMemory quota warning — using original chunk")
                        translated_parts.append(chunk)
                    else:
                        translated_parts.append(translated)
            except Exception as error:
                logger.error("MyMemory translation error: %s", error)
                translated_parts.append(chunk)

        return " ".join(translated_parts)

    @staticmethod
    def _split_text(text: str, max_chars: int) -> list[str]:
        """Split text into chunks of max_chars, breaking at sentence boundaries."""
        if len(text) <= max_chars:
            return [text]
        chunks = []
        while text:
            if len(text) <= max_chars:
                chunks.append(text)
                break
            split_at = text.rfind(". ", 0, max_chars)
            if split_at == -1:
                split_at = max_chars
            chunks.append(text[:split_at + 1].strip())
            text = text[split_at + 1:].strip()
        return chunks

    @staticmethod
    def _apply_corrections(text: str) -> str:
        """
        Fix common MyMemory mistranslations of Sri Lankan place names.
        Case-insensitive word-boundary replacement.
        """
        for wrong, correct in SL_PLACE_CORRECTIONS.items():
            if not correct:
                continue
            pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
            text = pattern.sub(correct, text)
        return text

    @staticmethod
    def _translation_fallback(text: str, reason: str) -> dict[str, Any]:
        return {
            "detected_language": "Unknown",
            "language_code"    : "??",
            "is_english"       : True,
            "translated_text"  : text,
            "was_translated"   : False,
            "confidence"       : "low",
            "fallback_reason"  : reason,
        }