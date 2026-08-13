import re
from typing import Any

from app.data.extraction_data import (
    COMMUNITY_KEYWORDS,
    LOCATION_EXCLUSION_WORDS,
    LOCATION_LABELS,
    SRI_LANKA_PLACES,
)


class ExtractionService:
    def __init__(self, nlp: Any) -> None:
        """Store spaCy and precompile lookup patterns once for faster requests."""
        self.nlp = nlp
        self.location_exclusion_words = {
            self._normalize_location(value) for value in LOCATION_EXCLUSION_WORDS
        }

        places = sorted(SRI_LANKA_PLACES, key=len, reverse=True)
        self.gazetteer_pattern = re.compile(
            r"\b(" + "|".join(re.escape(place) for place in places) + r")\b",
            re.IGNORECASE,
        )
        self.community_patterns = {
            label: [re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE) for keyword in keywords]
            for label, keywords in COMMUNITY_KEYWORDS.items()
        }

    @staticmethod
    def _normalize_location(value: str) -> str:
        """Normalize extracted text before duplicate and exclusion checks."""
        return re.sub(r"\s+", " ", value.strip().lower())

    def _is_excluded_location(self, value: str) -> bool:
        """Return True when a disaster/event word was mislabeled as a location."""
        return self._normalize_location(value) in self.location_exclusion_words

    def extract_locations(self, text: str) -> dict[str, Any]:
        """Find unique locations with spaCy first and the local gazetteer second."""
        locations: list[dict[str, Any]] = []
        seen: set[str] = set()

        # spaCy identifies general geographic entities and facilities.
        # Because it is a general model, disaster words such as "tsunami" can
        # sometimes be mislabeled as LOC, so we filter those words before saving.
        if self.nlp is not None:
            for entity in self.nlp(text).ents:
                value = entity.text.strip()
                key = self._normalize_location(value)

                if (
                    entity.label_ in LOCATION_LABELS
                    and key
                    and key not in seen
                    and not self._is_excluded_location(value)
                ):
                    seen.add(key)
                    locations.append({
                        "text": value,
                        "label": entity.label_,
                        "start": entity.start_char,
                        "end": entity.end_char,
                        "source": "spacy_ner",
                    })

        # The gazetteer finds Sri Lankan places that the general spaCy model misses.
        for match in self.gazetteer_pattern.finditer(text):
            value = match.group(0).strip()
            key = self._normalize_location(value)
            if key not in seen and not self._is_excluded_location(value):
                seen.add(key)
                locations.append({
                    "text": value,
                    "label": "GPE",
                    "start": match.start(),
                    "end": match.end(),
                    "source": "gazetteer",
                })

        return {
            "locations": locations,
            "location_count": len(locations),
            "has_location": bool(locations),
        }

    def extract_communities(self, text: str) -> dict[str, Any]:
        """Find affected groups and return at most one match for each group type."""
        communities = []
        for label, patterns in self.community_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    communities.append({"community": label, "matched_text": match.group(0)})
                    break

        return {
            "affected_communities": communities,
            "community_count": len(communities),
            "has_community": bool(communities),
        }
