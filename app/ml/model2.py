import torch.nn as nn
from transformers import AutoModel


class MCMSModel2(nn.Module):
    """Custom HumAID classification head used when Model 2 was trained."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        hidden_dim: int,
        dropout1: float,
        dropout2: float,
    ) -> None:
        """Recreate the same encoder and custom classification head used in training."""
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout2),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def mean_pool(embeddings, attention_mask):
        """Average only real token embeddings and ignore padded token positions."""
        mask = attention_mask.unsqueeze(-1).float()
        return (embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        """Define how token tensors pass through the encoder and classifier."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(output.last_hidden_state, attention_mask)
        return self.classifier(pooled)
