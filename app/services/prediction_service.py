from typing import Any

import torch
import torch.nn.functional as functional

from app.core.config import Settings
from app.ml.registry import ModelRegistry


URGENCY_EMOJI = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}


class PredictionService:
    def __init__(self, registry: ModelRegistry, settings: Settings) -> None:
        """Use the shared models and configured tokenizer input length."""
        self.registry = registry
        self.max_length = settings.max_length

    def _encode(self, tokenizer: Any, text: str) -> dict[str, torch.Tensor]:
        """Convert text into padded token tensors accepted by a transformer model."""
        return tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

    @staticmethod
    def _scores(probabilities: torch.Tensor, metadata: dict[str, Any]) -> tuple[int, list, dict]:
        """Convert model probabilities into the winning ID, top three, and all scores."""
        prediction_id = probabilities.argmax().item()
        names = metadata["class_names"]
        scores = {
            names[index]: round(probabilities[index].item() * 100, 2)
            for index in range(metadata["num_classes"])
        }
        top_three = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
        return prediction_id, top_three, scores

    def predict_crisis_type(self, text: str) -> dict[str, Any]:
        """Run Model 1 and return the predicted crisis type with confidence scores."""
        encoded = self._encode(self.registry.tokenizer1, text)
        # Inference mode disables gradient tracking because this API does not train models.
        with torch.inference_mode():
            logits = self.registry.model1(
                input_ids=encoded["input_ids"].to(self.registry.device),
                attention_mask=encoded["attention_mask"].to(self.registry.device),
            ).logits
            probabilities = functional.softmax(logits, dim=1)[0]

        prediction_id, top_three, scores = self._scores(probabilities, self.registry.metadata1)
        return {
            "crisis_type": self.registry.metadata1["class_names"][prediction_id],
            "confidence": scores[self.registry.metadata1["class_names"][prediction_id]],
            "top_3": top_three,
            "all_scores": scores,
        }

    def predict_message_type(self, text: str) -> dict[str, Any]:
        """Run Model 2 and return the humanitarian message classification."""
        encoded = self._encode(self.registry.tokenizer2, text)
        with torch.inference_mode():
            logits = self.registry.model2(
                input_ids=encoded["input_ids"].to(self.registry.device),
                attention_mask=encoded["attention_mask"].to(self.registry.device),
            )
            probabilities = functional.softmax(logits, dim=1)[0]

        prediction_id, top_three, scores = self._scores(probabilities, self.registry.metadata2)
        return {
            "message_type": self.registry.metadata2["class_names"][prediction_id],
            "confidence": scores[self.registry.metadata2["class_names"][prediction_id]],
            "top_3": top_three,
            "all_scores": scores,
        }

    def predict_urgency(self, text: str) -> dict[str, Any]:
        """Run Model 3 and return the urgency level, emoji, and confidence scores."""
        encoded = self._encode(self.registry.tokenizer3, text)
        with torch.inference_mode():
            logits = self.registry.model3(
                input_ids=encoded["input_ids"].to(self.registry.device),
                attention_mask=encoded["attention_mask"].to(self.registry.device),
            ).logits
            probabilities = functional.softmax(logits, dim=1)[0]

        prediction_id, _, scores = self._scores(probabilities, self.registry.metadata3)
        urgency_level = self.registry.metadata3["class_names"][prediction_id]
        return {
            "urgency_level": urgency_level,
            "emoji": URGENCY_EMOJI[urgency_level],
            "confidence": scores[urgency_level],
            "all_scores": scores,
        }
