# 🧠 MCMS AI Service — NLP Classification & Translation Pipeline

> FastAPI micro-service powering the AI backbone of the **Multilingual Crisis Management System (MCMS)**.  
> Runs three fine-tuned transformer models for crisis-type classification, humanitarian message typing, and urgency detection — with built-in multilingual translation and named-entity extraction.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-4.41-FFD21E?logo=huggingface&logoColor=black)
![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5?logo=spacy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [ML Models](#ml-models)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [API Endpoints](#api-endpoints)
- [Pipeline Flow](#pipeline-flow)
- [Multilingual Support](#multilingual-support)
- [Project Structure](#project-structure)
- [Related Repositories](#related-repositories)
- [License](#license)

---

## Overview

This service is a **standalone AI micro-service** that the Node.js backend calls over HTTP. It provides:

- **Crisis-Type Classification** — identifies what kind of disaster (flood, earthquake, storm, etc.)
- **Message-Type Classification** — categorises the humanitarian intent (request for help, infrastructure damage, rescue needed, etc.)
- **Urgency Detection** — assigns an urgency level (🟢 Low, 🟠 Medium, 🔴 High)
- **Location Extraction** — uses spaCy NER + a Sri Lankan gazetteer to find place names
- **Community Detection** — identifies affected population groups (elderly, children, displaced, etc.)
- **Multilingual Translation** — detects non-English input via Unicode script analysis and translates to English using the MyMemory API before running classification

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │            FastAPI Application              │
                         │                                             │
  POST /api/predict ────▶│  TranslationService ──▶ PipelineService     │
                         │                           │                 │
                         │            ┌──────────────┼──────────────┐  │
                         │            ▼              ▼              ▼  │
                         │      Model 1         Model 2       Model 3  │
                         │    (Crisis Type)   (Message Type)  (Urgency)│
                         │            │              │              │  │
                         │            ▼              ▼              ▼  │
                         │      PredictionService + ExtractionService  │
                         └─────────────────────────────────────────────┘
                                             │
                                    JSON response
                                             │
                                             ▼
                                    Node.js Backend
```

---

## ML Models

| Model | Directory | Architecture | Task | Classes |
|---|---|---|---|---|
| **Model 1** | `mcms_model1_final/` | Fine-tuned XLM-RoBERTa | Crisis-type classification | Flood, Earthquake, Storm, Wildfire, etc. |
| **Model 2** | `mcms_model2_humaid/` | Custom classifier (BERT-based + custom head) | Humanitarian message typing | Infrastructure Damage, Rescue, Displaced, etc. |
| **Model 3** | `mcms_model3_final/` | Fine-tuned transformer | Urgency detection | Low, Medium, High |

All models are loaded into GPU memory (if CUDA is available) at startup via the `ModelRegistry` and kept in evaluation mode for fast inference.

> **Note:** Model weight files (`.safetensors`, `.pt`) are large binary files. Make sure [Git LFS](https://git-lfs.github.com/) is installed before cloning.

---

## Key Features

| Feature | Description |
|---|---|
| **Three-Model Pipeline** | Runs crisis type, message type, and urgency classifiers in a single request |
| **Multilingual Input** | Detects 14+ languages via Unicode script analysis (Sinhala, Tamil, Arabic, Hindi, Chinese, etc.) |
| **MyMemory Translation** | Free, no-API-key translation via MyMemory REST API with automatic text chunking for long inputs |
| **Sri Lanka Place Corrections** | Post-translation fixes for commonly mistranslated Sri Lankan place names (e.g. "City of Gems" → Ratnapura) |
| **spaCy NER** | Named-entity recognition for location extraction using `en_core_web_sm` |
| **Custom Gazetteer** | Sri Lanka-specific place name lookup for locations missed by the general spaCy model |
| **Community Detection** | Pattern-based extraction of affected community groups from report text |
| **Health Endpoint** | Reports load status of every model and spaCy pipeline |
| **CUDA Support** | Automatic GPU detection and model offloading for faster inference |
| **Confidence Scores** | Returns top-3 predictions with percentage confidence for each classifier |
| **Latency Tracking** | Every response includes `latency_ms` for performance monitoring |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Framework | FastAPI 0.111 |
| ML Framework | PyTorch 2.0+ |
| Transformers | HuggingFace Transformers 4.41 |
| NER | spaCy 3.7 (`en_core_web_sm`) |
| Translation | MyMemory API (free, no key required) |
| Validation | Pydantic 2.7 |
| Server | Uvicorn 0.29 |

---

## Getting Started

### Prerequisites

- **Python** ≥ 3.11
- **pip** or **uv** for dependency management
- (Optional) NVIDIA GPU with CUDA for faster inference

### Installation

```bash
# Clone the repository
git clone https://github.com/SandaruDulneth/mcms-backend-py.git
cd mcms-backend-py

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the spaCy English model
python -m spacy download en_core_web_sm

# Start the development server
uvicorn main:app --reload --port 8000
```

The service starts on **`http://localhost:8000`** by default.

### Environment Variables (Optional)

| Variable | Default | Description |
|---|---|---|
| `SPACY_MODEL` | `en_core_web_sm` | spaCy pipeline to load for NER |
| `MODEL_MAX_LENGTH` | `128` | Maximum token length for tokenizer input |
| `CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |

---

## API Endpoints

### Prediction

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/predict` | Full pipeline — classifies, extracts locations, detects communities |
| `POST` | `/api/predict/multilingual` | Full pipeline with automatic language detection and translation |
| `POST` | `/api/predict/crisis-type` | Crisis-type classification only (Model 1) |
| `POST` | `/api/predict/message-type` | Message-type classification only (Model 2) |
| `POST` | `/api/predict/urgency` | Urgency detection only (Model 3) |

### Extraction

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/extract/locations` | spaCy NER + gazetteer location extraction |
| `POST` | `/api/extract/communities` | Affected community group detection |

### System

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/info/health` | Model load status, device info, and service health |
| `GET` | `/docs` | Interactive Swagger UI documentation |

### Example Request

```bash
curl -X POST http://localhost:8000/api/predict/multilingual \
  -H "Content-Type: application/json" \
  -d '{"text": "ගම්පහ දිස්ත්‍රික්කයේ ගංවතුරට ජනතාව දැඩි ලෙස පීඩා විඳිනවා"}'
```

### Example Response

```json
{
  "original_text": "ගම්පහ දිස්ත්‍රික්කයේ ගංවතුරට ජනතාව දැඩි ලෙස පීඩා විඳිනවා",
  "detected_language": "Sinhala",
  "language_code": "si",
  "was_translated": true,
  "translated_text": "People are severely affected by floods in Gampaha district",
  "crisis_type": {
    "crisis_type": "flood",
    "confidence": 94.32,
    "top_3": [["flood", 94.32], ["storm", 3.21], ["landslide", 1.08]]
  },
  "message_type": {
    "message_type": "displaced_people_and_evacuations",
    "confidence": 67.85
  },
  "urgency": {
    "urgency_level": "High",
    "emoji": "🔴",
    "confidence": 88.14
  },
  "location_extraction": {
    "locations": [{"text": "Gampaha", "label": "GPE", "source": "gazetteer"}],
    "location_count": 1
  },
  "summary": "Crisis: flood (94.32%) | Type: displaced_people_and_evacuations (67.85%) | Urgency: 🔴 High (88.14%) | Location: Gampaha",
  "latency_ms": 142.5
}
```

---

## Pipeline Flow

```
Input Text
    │
    ▼
┌─────────────────────────┐
│ Language Detection       │  ◀── Unicode script range analysis
│ (Sinhala, Tamil, etc.)   │
└────────────┬────────────┘
             │ non-English?
             ▼
┌─────────────────────────┐
│ MyMemory Translation     │  ◀── Free API, chunked for long text
│ + Place Name Corrections │
└────────────┬────────────┘
             │ English text
             ▼
┌─────────────────────────┐
│ Model 1: Crisis Type     │  ◀── XLM-RoBERTa classifier
├─────────────────────────┤
│ Model 2: Message Type    │  ◀── Custom BERT + classification head
├─────────────────────────┤
│ Model 3: Urgency         │  ◀── Fine-tuned transformer
├─────────────────────────┤
│ spaCy NER + Gazetteer    │  ◀── Location extraction
├─────────────────────────┤
│ Community Detection      │  ◀── Pattern-based keyword matching
└────────────┬────────────┘
             │
             ▼
      JSON Response
```

---

## Multilingual Support

The service supports **14+ languages** via Unicode script detection:

| Language | Script | Code |
|---|---|---|
| Sinhala | `U+0D80–U+0DFF` | `si` |
| Tamil | `U+0B80–U+0BFF` | `ta` |
| Arabic | `U+0600–U+06FF` | `ar` |
| Hindi | `U+0900–U+097F` (Devanagari) | `hi` |
| Bengali | `U+0980–U+09FF` | `bn` |
| Chinese | `U+4E00–U+9FFF` | `zh` |
| Japanese | `U+3040–U+309F` (Hiragana) | `ja` |
| Korean | `U+AC00–U+D7AF` | `ko` |
| French, German, Spanish, Portuguese, Russian, Urdu | Latin/Cyrillic fallback | various |

Translation is performed via the **MyMemory API** — free and requires no API key.

---

## Project Structure

```
mcms-backend-py/
├── main.py                         # Uvicorn entry point
├── requirements.txt                # Python dependencies
├── app/
│   ├── main.py                     # FastAPI app factory with lifespan
│   ├── core/
│   │   ├── config.py               # Settings dataclass (model dirs, CORS, etc.)
│   │   └── logging.py              # Logging configuration
│   ├── api/
│   │   ├── router.py               # Top-level API router
│   │   ├── dependencies.py         # FastAPI dependency injection
│   │   └── routes/
│   │       ├── predict.py          # /api/predict endpoints
│   │       ├── extract.py          # /api/extract endpoints
│   │       └── info.py             # /api/info/health endpoint
│   ├── ml/
│   │   ├── registry.py             # ModelRegistry — loads all models at startup
│   │   └── model2.py               # Custom PyTorch architecture for Model 2
│   ├── schemas/                    # Pydantic request/response models
│   ├── services/
│   │   ├── pipeline_service.py     # Orchestrates full prediction pipeline
│   │   ├── prediction_service.py   # Individual model inference (Model 1, 2, 3)
│   │   ├── extraction_service.py   # spaCy NER + gazetteer + community detection
│   │   └── translation_service.py  # MyMemory language detection & translation
│   └── data/
│       └── extraction_data.py      # Sri Lanka gazetteer, community keywords, exclusion lists
├── mcms_model1_final/              # Crisis-type classifier weights & tokenizer
├── mcms_model2_humaid/             # Message-type classifier weights & tokenizer
└── mcms_model3_final/              # Urgency classifier weights & tokenizer
```

---

## Related Repositories

| Repository | Description |
|---|---|
| [mcms-backend-ts](https://github.com/SandaruDulneth/mcms-backend-ts) | Node.js/Express REST API — report management, credibility scoring, GDACS/NewsAPI integration |
| [mcms-frontend-ts](https://github.com/SandaruDulneth/mcms-frontend-ts) | Next.js 16 dashboard, crisis map, analytics, and public report submission |

---

## License

This project is part of a Final Year Project at the University level.

