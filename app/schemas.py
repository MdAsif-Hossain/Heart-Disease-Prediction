"""
schemas.py — Pydantic models for request / response validation.

These models act as the "contract" between the client and the API.
FastAPI automatically validates incoming JSON against these schemas
and returns a 422 error if the data is wrong.
"""

from pydantic import BaseModel, Field


# ── Input Schema ───────────────────────────────────────────────────────────────
class HeartInput(BaseModel):
    """13 clinical features required to predict heart disease."""

    age: int = Field(
        ..., ge=1, le=120,
        description="Age of the patient in years (1–120)",
        example=63
    )
    sex: int = Field(
        ..., ge=0, le=1,
        description="Biological sex: 1 = male, 0 = female",
        example=1
    )
    cp: int = Field(
        ..., ge=0, le=3,
        description=(
            "Chest pain type: "
            "0 = typical angina, 1 = atypical angina, "
            "2 = non-anginal pain, 3 = asymptomatic"
        ),
        example=3
    )
    trestbps: int = Field(
        ..., ge=50, le=250,
        description="Resting blood pressure in mm Hg at hospital admission (50–250)",
        example=145
    )
    chol: int = Field(
        ..., ge=100, le=600,
        description="Serum cholesterol in mg/dl (100–600)",
        example=233
    )
    fbs: int = Field(
        ..., ge=0, le=1,
        description="Fasting blood sugar > 120 mg/dl: 1 = true, 0 = false",
        example=1
    )
    restecg: int = Field(
        ..., ge=0, le=2,
        description=(
            "Resting electrocardiographic results: "
            "0 = normal, 1 = ST-T wave abnormality, "
            "2 = left ventricular hypertrophy"
        ),
        example=0
    )
    thalach: int = Field(
        ..., ge=60, le=250,
        description="Maximum heart rate achieved during exercise (60–250)",
        example=150
    )
    exang: int = Field(
        ..., ge=0, le=1,
        description="Exercise-induced angina: 1 = yes, 0 = no",
        example=0
    )
    oldpeak: float = Field(
        ..., ge=0.0, le=10.0,
        description="ST depression induced by exercise relative to rest (0.0–10.0)",
        example=2.3
    )
    slope: int = Field(
        ..., ge=0, le=2,
        description=(
            "Slope of the peak exercise ST segment: "
            "0 = upsloping, 1 = flat, 2 = downsloping"
        ),
        example=0
    )
    ca: int = Field(
        ..., ge=0, le=4,
        description="Number of major vessels (0–4) colored by fluoroscopy",
        example=0
    )
    thal: int = Field(
        ..., ge=0, le=3,
        description=(
            "Thalassemia: "
            "0 = normal, 1 = fixed defect, "
            "2 = reversible defect, 3 = other"
        ),
        example=1
    )

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }


# ── Output Schemas ─────────────────────────────────────────────────────────────
class PredictionOutput(BaseModel):
    """Result returned by the /predict endpoint."""

    heart_disease: bool = Field(
        ...,
        description="True if heart disease is predicted, False otherwise"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model confidence score for the predicted class (0.0–1.0)"
    )
    prediction_label: str = Field(
        ...,
        description="Human-readable prediction label"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "heart_disease": True,
                "confidence": 0.82,
                "prediction_label": "Heart Disease Detected"
            }
        }


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(..., example="healthy")
    message: str = Field(..., example="Heart Disease Prediction API is running.")


class InfoResponse(BaseModel):
    """Response from the /info endpoint."""

    model_type: str = Field(
        ...,
        description="The type / name of the trained ML model",
        example="LogisticRegression (with StandardScaler)"
    )
    features: list[str] = Field(
        ...,
        description="Ordered list of input feature names expected by the model"
    )
    num_features: int = Field(
        ...,
        description="Total number of input features",
        example=13
    )
    target_classes: list[str] = Field(
        ...,
        description="Class labels for the prediction output",
        example=["No Heart Disease", "Heart Disease"]
    )
    version: str = Field(
        ...,
        description="API version string",
        example="1.0.0"
    )
    dataset: str = Field(
        ...,
        description="Dataset used for training",
        example="UCI Heart Disease Dataset (via Kaggle)"
    )
