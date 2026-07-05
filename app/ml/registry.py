import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import Settings
from app.ml.model2 import MCMSModel2


logger = logging.getLogger(__name__)


class ModelRegistry:
    """Owns all loaded ML components for the lifetime of the application."""

    def __init__(self, settings: Settings) -> None:
        """Prepare empty component slots and select CUDA when it is available."""
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_errors: dict[str, str] = {}

        self.model1: Any = None
        self.tokenizer1: Any = None
        self.metadata1: dict[str, Any] | None = None
        self.model2: Any = None
        self.tokenizer2: Any = None
        self.metadata2: dict[str, Any] | None = None
        self.model3: Any = None
        self.tokenizer3: Any = None
        self.metadata3: dict[str, Any] | None = None
        self.nlp: Any = None

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        """Read model metadata or architecture configuration from a JSON file."""
        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def load_all(self) -> None:
        """Attempt to load every classifier and the spaCy language pipeline."""
        logger.info("Loading AI components on %s", self.device)
        self._load_model1()
        self._load_model2()
        self._load_model3()
        self._load_spacy()

    def _record_error(self, component: str, error: Exception) -> None:
        """Remember a startup failure so the health endpoint can explain it."""
        self.load_errors[component] = str(error)
        logger.exception("Failed to load %s", component)

    def _load_model1(self) -> None:
        """Load the crisis-type tokenizer, metadata, and trained classifier."""
        try:
            directory = self.settings.model1_dir
            self.metadata1 = self._read_json(directory / "model_metadata.json")
            self.tokenizer1 = AutoTokenizer.from_pretrained(directory)
            self.model1 = AutoModelForSequenceClassification.from_pretrained(directory).to(self.device)
            self.model1.eval()
            logger.info("Crisis-type model loaded")
        except Exception as error:
            self.model1 = self.tokenizer1 = self.metadata1 = None
            self._record_error("model1", error)

    def _load_model2(self) -> None:
        """Rebuild the custom message classifier and load its trained weights."""
        try:
            directory = self.settings.model2_dir
            architecture = self._read_json(directory / "architecture_config.json")
            self.metadata2 = self._read_json(directory / "model_metadata.json")
            self.tokenizer2 = AutoTokenizer.from_pretrained(directory)
            keys = ("model_name", "num_classes", "hidden_dim", "dropout1", "dropout2")
            self.model2 = MCMSModel2(**{key: architecture[key] for key in keys})
            weights = torch.load(directory / "model_weights.pt", map_location=self.device)
            self.model2.load_state_dict(weights)
            self.model2.to(self.device).eval()
            logger.info("Message-type model loaded")
        except Exception as error:
            self.model2 = self.tokenizer2 = self.metadata2 = None
            self._record_error("model2", error)

    def _load_model3(self) -> None:
        """Load the urgency tokenizer, metadata, and trained classifier."""
        try:
            directory = self.settings.model3_dir
            self.metadata3 = self._read_json(directory / "model_metadata.json")
            self.tokenizer3 = AutoTokenizer.from_pretrained(directory)
            self.model3 = AutoModelForSequenceClassification.from_pretrained(directory).to(self.device)
            self.model3.eval()
            logger.info("Urgency model loaded")
        except Exception as error:
            self.model3 = self.tokenizer3 = self.metadata3 = None
            self._record_error("model3", error)

    def _load_spacy(self) -> None:
        """Load the configured spaCy pipeline used for named-entity recognition."""
        try:
            import spacy

            self.nlp = spacy.load(self.settings.spacy_model)
            logger.info("spaCy model loaded")
        except Exception as error:
            self.nlp = None
            self._record_error("spacy", error)

    def status(self, component: str, loaded: bool) -> str:
        """Format one component's state for the public health response."""
        return "loaded" if loaded else f"error: {self.load_errors.get(component)}"
