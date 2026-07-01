# MCMS FastAPI AI Backend

FastAPI service for the **Multilingual Crisis Management System (MCMS)**. This backend provides AI/NLP inference for crisis reports submitted by users or collected from external sources.

The service classifies crisis messages into disaster type, humanitarian/message type, and urgency level. It also extracts mentioned locations and affected community groups so the main MCMS backend can store clean, dashboard-friendly report data.

## Features

- Crisis/disaster type classification
- Humanitarian message type classification
- Urgency level classification
- Location extraction using spaCy NER with a Sri Lankan gazetteer fallback
- Affected community extraction using rule-based keyword matching
- Combined full-pipeline endpoint for one-call report analysis
- Health endpoint for checking model and NLP component availability
- CORS enabled for local web/backend integration

## System Role

This repository is the Python AI service in the MCMS architecture.

```text
Next.js frontend
    -> Node.js/TypeScript backend
        -> FastAPI AI backend
            -> AI predictions + NLP extraction
        -> MongoDB
```

The frontend should not call this service directly in production. The Node.js backend should call this API, validate the response, save the processed report in MongoDB, and return a clean response to the frontend.

## Models and NLP Components

| Component | Purpose | Directory / Method |
|---|---|---|
| Model 1 | Crisis/disaster type classification | `mcms_model1_final/` |
| Model 2 | Humanitarian/message type classification | `mcms_model2_humaid/` |
| Model 3 | Urgency classification | `mcms_model3_final/` |
| Location extraction | Detects locations mentioned in the text | spaCy `en_core_web_sm` + Sri Lanka gazetteer |
| Community extraction | Detects affected groups such as children, elderly, residents, farmers, etc. | Rule-based keyword matching |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Basic service information |
| `GET` | `/health` | Checks loaded model and NLP component status |
| `POST` | `/predict/crisis-type` | Predicts disaster/crisis type |
| `POST` | `/predict/message-type` | Predicts humanitarian message type |
| `POST` | `/predict/urgency` | Predicts urgency level |
| `POST` | `/extract/location` | Extracts locations from text |
| `POST` | `/extract/community` | Extracts affected communities from text |
| `POST` | `/predict/full` | Runs the full AI/NLP pipeline in one request |

## Project Structure

```text
.
├── app/
│   ├── api/                  # FastAPI routers and dependencies
│   │   └── routes/           # Info, prediction, and extraction endpoints
│   ├── core/                 # Settings and logging configuration
│   ├── data/                 # Gazetteer and community keyword data
│   ├── ml/                   # Model 2 architecture and model registry
│   ├── schemas/              # Pydantic request schemas
│   ├── services/             # Prediction, extraction, and full pipeline logic
│   └── main.py               # Application factory and lifespan setup
├── main.py                   # Lightweight Uvicorn entry point
├── requirements.txt
├── mcms_model1_final/
├── mcms_model2_humaid/
├── mcms_model3_final/
└── README.md
```

## Requirements

- Python 3.10+
- pip
- Virtual environment recommended
- CPU is supported; CUDA GPU is used automatically if available

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## Running the API

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

## Example: Full Prediction

Request:

```bash
curl -X POST "http://localhost:8000/predict/full" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Elderly residents and children trapped on rooftops as flash floods submerge Ratnapura and Kegalle. Urgent rescue needed, water levels rising fast."
  }'
```

Example response:

```json
{
  "input_text": "Elderly residents and children trapped on rooftops as flash floods submerge Ratnapura and Kegalle. Urgent rescue needed, water levels rising fast.",
  "crisis_type": {
    "crisis_type": "flood",
    "confidence": 99.72
  },
  "message_type": {
    "message_type": "requests_or_urgent_needs",
    "confidence": 91.45
  },
  "urgency": {
    "urgency_level": "High",
    "emoji": "🔴",
    "confidence": 97.49
  },
  "location_extraction": {
    "locations": [
      {
        "text": "Ratnapura",
        "label": "GPE",
        "source": "gazetteer"
      },
      {
        "text": "Kegalle",
        "label": "GPE",
        "source": "gazetteer"
      }
    ],
    "location_count": 2,
    "has_location": true
  },
  "community_extraction": {
    "affected_communities": [
      {
        "community": "elderly",
        "matched_text": "Elderly"
      },
      {
        "community": "children",
        "matched_text": "children"
      },
      {
        "community": "residents",
        "matched_text": "residents"
      }
    ],
    "community_count": 3,
    "has_community": true
  },
  "latency_ms": 813.6,
  "summary": "Crisis: flood (99.72%) | Type: requests_or_urgent_needs (91.45%) | Urgency: 🔴 High (97.49%) | Location: Ratnapura, Kegalle | Communities: elderly, children, residents"
}
```

## Request Body

Most endpoints accept the following JSON body:

```json
{
  "text": "Crisis report text goes here"
}
```

## Health Check

```bash
curl http://localhost:8000/health
```

The health endpoint reports whether the three ML models and spaCy NER component were loaded successfully.

## Integration Notes

The recommended integration pattern is:

1. User submits a crisis report through the frontend.
2. Node.js backend receives the report.
3. Node.js backend calls `POST /predict/full` on this FastAPI service.
4. Node.js backend validates the AI response.
5. Node.js backend saves the processed report in MongoDB.
6. Frontend displays disaster type, message type, urgency, extracted locations, and affected communities.

## Current Limitations

- Multilingual processing is planned but not fully implemented yet.
- Location extraction currently combines spaCy English NER with a Sri Lankan gazetteer fallback.
- Affected community extraction is rule-based and depends on the keyword list in `app/data/extraction_data.py`.
- Model performance and response latency may vary depending on hardware.
- This service is intended for development/research use and should be reviewed before production deployment.

## Future Improvements

- Improve model inference speed and startup time
- Add stronger error handling and structured logging
- Extend multilingual support for Sinhala, Tamil, and English crisis messages
- Expand the gazetteer and affected community keyword list
- Add automated tests for endpoint responses
- Add deployment configuration for production environments

## License

Add a license before publishing this repository publicly.
