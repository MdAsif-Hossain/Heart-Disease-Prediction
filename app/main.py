"""
main.py — FastAPI application entry point.

Endpoints:
    GET  /health   → liveness check
    GET  /info     → model metadata
    POST /predict  → heart disease prediction

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import HeartInput, HealthResponse, InfoResponse, PredictionOutput

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
API_VERSION = "1.0.0"
MODEL_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "heart_model.joblib")

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

TARGET_CLASSES = ["No Heart Disease", "Heart Disease"]

# ── Global model store ─────────────────────────────────────────────────────────
model_store: dict[str, Any] = {}


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    # ── STARTUP ───────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        logger.error(
            "Model file not found at %s. "
            "Please run `python model/train.py` first.",
            MODEL_PATH,
        )
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    logger.info("Loading model from %s …", MODEL_PATH)
    pipeline = joblib.load(MODEL_PATH)
    model_store["pipeline"]   = pipeline
    model_store["model_type"] = (
        f"{type(pipeline.named_steps['clf']).__name__} "
        f"(with {type(pipeline.named_steps['scaler']).__name__})"
    )
    logger.info("Model loaded successfully → %s", model_store["model_type"])

    yield  # ← application runs here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("Shutting down — clearing model cache …")
    model_store.clear()


# ── App instance ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "A machine-learning API that predicts the presence of heart disease "
        "from 13 clinical features using a Logistic Regression classifier trained "
        "on the UCI Heart Disease dataset.\n\n"
        "**Endpoints:**\n"
        "- `GET /health` — liveness check\n"
        "- `GET /info`   — model metadata\n"
        "- `POST /predict` — run a prediction\n"
    ),
    version=API_VERSION,
    contact={
        "name": "Module 17 Assignment",
        "url": "https://github.com",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ─────────────────────────────────────────────────────────────────────
def _get_pipeline():
    """Return the loaded pipeline or raise a 503 if not ready."""
    pipeline = model_store.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please check server logs.",
        )
    return pipeline


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["Monitoring"],
)
async def health_check() -> HealthResponse:
    """
    Liveness probe — confirms the API is running and the model is loaded.

    Returns a simple status message. Use this endpoint in Docker / Render
    health checks.
    """
    model_loaded = "pipeline" in model_store
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    return HealthResponse(
        status="healthy",
        message="Heart Disease Prediction API is running.",
    )


@app.get(
    "/info",
    response_model=InfoResponse,
    summary="Model Information",
    tags=["Metadata"],
)
async def model_info() -> InfoResponse:
    """
    Returns metadata about the trained model, including:
    - Model type and preprocessing steps
    - Full list of expected input features
    - Output class labels
    - API version
    """
    _get_pipeline()  # ensure model is loaded
    return InfoResponse(
        model_type=model_store.get("model_type", "Unknown"),
        features=FEATURES,
        num_features=len(FEATURES),
        target_classes=TARGET_CLASSES,
        version=API_VERSION,
        dataset="UCI Heart Disease Dataset (via Kaggle — johnsmith88)",
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Predict Heart Disease",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK,
)
async def predict(payload: HeartInput) -> PredictionOutput:
    """
    Predicts whether a patient has heart disease based on 13 clinical features.

    **Input:** JSON body with all 13 required features (see schema below).

    **Output:**
    - `heart_disease` — `true` = heart disease predicted, `false` = no heart disease
    - `confidence`    — model's confidence score (0.0 to 1.0)
    - `prediction_label` — human-readable label

    **Example curl command:**
    ```
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
           "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,
           "slope":0,"ca":0,"thal":1}'
    ```
    """
    pipeline = _get_pipeline()

    # Build input array in the exact feature order the model was trained on
    input_array = np.array([[
        payload.age,
        payload.sex,
        payload.cp,
        payload.trestbps,
        payload.chol,
        payload.fbs,
        payload.restecg,
        payload.thalach,
        payload.exang,
        payload.oldpeak,
        payload.slope,
        payload.ca,
        payload.thal,
    ]], dtype=float)

    try:
        prediction   = int(pipeline.predict(input_array)[0])          # 0 or 1
        probabilities = pipeline.predict_proba(input_array)[0]        # [p0, p1]
        confidence   = float(probabilities[prediction])
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        ) from exc

    has_disease = bool(prediction == 1)
    label       = "Heart Disease Detected" if has_disease else "No Heart Disease Detected"

    logger.info(
        "Prediction → %s  (confidence: %.2f)",
        label,
        confidence,
    )

    return PredictionOutput(
        heart_disease=has_disease,
        confidence=round(confidence, 4),
        prediction_label=label,
    )
