# Heart Disease Prediction API

A REST API that predicts the presence or absence of heart disease from 13 clinical features, built with **FastAPI**, containerized with **Docker**, and deployed on **Render**.

- **Dataset:** [UCI Heart Disease Dataset via Kaggle (johnsmith88)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Model:** Logistic Regression with StandardScaler (scikit-learn Pipeline)
- **Live URL:** https://heart-disease-prediction-2-dbhn.onrender.com
- **Swagger UI:** https://heart-disease-prediction-2-dbhn.onrender.com/docs

---

## Project Structure

```
.
├── app/
│   ├── __init__.py          # Package marker
│   ├── main.py              # FastAPI application & endpoints
│   └── schemas.py           # Pydantic request/response models
├── model/
│   ├── train.py             # Model training script
│   └── heart_model.joblib   # Saved trained model
├── data/
│   └── heart.csv            # Dataset (download from Kaggle)
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # Local development with Docker
├── render.yaml              # Render deployment blueprint
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| API Framework | FastAPI 0.115 |
| ML Library | scikit-learn 1.6 |
| Server | Uvicorn |
| Containerization | Docker (multi-stage build) |
| Deployment | Render (Docker environment) |
| Language | Python 3.11 |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Redirects to Swagger UI |
| `GET` | `/health` | Liveness check |
| `GET` | `/info` | Model metadata |
| `POST` | `/predict` | Run a prediction |

### GET `/health`

```bash
curl https://heart-disease-prediction-2-dbhn.onrender.com/health
```

Response:
```json
{
  "status": "healthy",
  "message": "Heart Disease Prediction API is running."
}
```

### GET `/info`

```bash
curl https://heart-disease-prediction-2-dbhn.onrender.com/info
```

Response:
```json
{
  "model_type": "LogisticRegression (with StandardScaler)",
  "features": ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"],
  "num_features": 13,
  "target_classes": ["No Heart Disease", "Heart Disease"],
  "version": "1.0.0",
  "dataset": "UCI Heart Disease Dataset (via Kaggle — johnsmith88)"
}
```

### POST `/predict`

**Input — 13 clinical features:**

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `age` | int | 1–120 | Age in years |
| `sex` | int | 0–1 | 1 = male, 0 = female |
| `cp` | int | 0–3 | Chest pain type |
| `trestbps` | int | 50–250 | Resting blood pressure (mm Hg) |
| `chol` | int | 100–600 | Serum cholesterol (mg/dl) |
| `fbs` | int | 0–1 | Fasting blood sugar > 120 mg/dl |
| `restecg` | int | 0–2 | Resting ECG results |
| `thalach` | int | 60–250 | Max heart rate achieved |
| `exang` | int | 0–1 | Exercise-induced angina |
| `oldpeak` | float | 0.0–10.0 | ST depression by exercise |
| `slope` | int | 0–2 | Peak exercise ST segment slope |
| `ca` | int | 0–4 | Major vessels colored by fluoroscopy |
| `thal` | int | 0–3 | Thalassemia type |

**Example Request:**
```bash
curl -X POST https://heart-disease-prediction-2-dbhn.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

**Response:**
```json
{
  "heart_disease": true,
  "confidence": 0.8241,
  "prediction_label": "Heart Disease Detected"
}
```

---

## Run Locally with Docker

```bash
# Build the image
docker-compose build

# Start the container
docker-compose up
```

Open **http://localhost:8000/docs** in your browser.

To stop:
```bash
docker-compose down
```

---

## Run Locally without Docker

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download heart.csv from Kaggle and place it in data/

# 4. Train the model
python model/train.py

# 5. Start the server
uvicorn app.main:app --reload --port 8000
```

---

## Docker Details

The `Dockerfile` uses a **multi-stage build**:

1. **Builder stage** — installs all Python dependencies into an isolated directory
2. **Runtime stage** — copies only the installed packages and app code into a lean final image running as a non-root user

This keeps the final image small and secure.

---

## Deployment on Render

This project is deployed using Render's Docker environment via a `render.yaml` blueprint.

Steps to redeploy on your own account:

1. Push this repository to GitHub
2. Log in to [render.com](https://render.com)
3. Click **New +** → **Blueprint**
4. Connect your GitHub repo
5. Render reads `render.yaml` automatically and creates the service
6. Wait ~3–5 minutes for the build to complete

> **Note:** Render's free tier spins down after 15 minutes of inactivity. The first request after a cold start may take ~30 seconds to respond.

---

## Live Deployment

| | URL |
|--|-----|
| API Root | https://heart-disease-prediction-2-dbhn.onrender.com |
| Swagger UI | https://heart-disease-prediction-2-dbhn.onrender.com/docs |
| Health Check | https://heart-disease-prediction-2-dbhn.onrender.com/health |
| Model Info | https://heart-disease-prediction-2-dbhn.onrender.com/info |

---

## GitHub Repository

https://github.com/MdAsif-Hossain/Heart-Disease-Prediction
