import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = "MCMS - Multilingual Crisis Management System"
    app_version: str = "1.3.0"
    model1_dir: Path = BASE_DIR / "mcms_model1_final"
    model2_dir: Path = BASE_DIR / "mcms_model2_humaid"
    model3_dir: Path = BASE_DIR / "mcms_model3_final"
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    max_length: int = int(os.getenv("MODEL_MAX_LENGTH", "128"))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    )


settings = Settings()