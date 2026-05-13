# Heart Disease Prediction API 🫀

A production-ready REST API that predicts the **presence or absence of heart disease** from 13 clinical features, built with **FastAPI**, containerized with **Docker**, and deployed on **Render**.

> Dataset: [UCI Heart Disease Dataset via Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
> Model: Logistic Regression with StandardScaler (scikit-learn Pipeline)

---

## 🗂️ Project Structure

```
.
├── app/
│   ├── __init__.py          # Package marker
│   ├── main.py              # FastAPI application & endpoints
│   └── schemas.py           # Pydantic request/response models
├── model/
│   ├── train.py             # Model training script
│   └── heart_model.joblib   # Saved trained model (generated)
├── data/
│   └── heart.csv            # Dataset (download from Kaggle)
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # Local development with Docker
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI 0.115 |
| ML Library | scikit-learn 1.6 |
| Server | Uvicorn |
| Containerization | Docker (multi-stage build) |
| Deployment | Render (Docker environment) |
| Language | Python 3.11 |

---

## 🚀 Quick Start

### Option A — Run locally (without Docker)

```powershell
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/heart-disease-api.git
cd heart-disease-api

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset and place it at data/heart.csv
#    https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

# 5. Train the model (generates model/heart_model.joblib)
python model/train.py

# 6. Start the API server
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** in your browser → interactive Swagger UI.

---

### Option B — Run with Docker Compose (recommended)

```powershell
# Build the Docker image
docker-compose build

# Start the container
docker-compose up

# Stop the container
docker-compose down
```

Open **http://localhost:8000/docs** → Swagger UI loads automatically.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check — confirms API is running |
| `GET` | `/info` | Model metadata — type, features, version |
| `POST` | `/predict` | Run a heart disease prediction |

### GET `/health`

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Heart Disease Prediction API is running."
}
```

---

### GET `/info`

```bash
curl http://localhost:8000/info
```

**Response:**
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

---

### POST `/predict`

**Input — 13 clinical features:**

| Feature | Type | Description |
|---------|------|-------------|
| `age` | int | Age in years |
| `sex` | int | 1 = male, 0 = female |
| `cp` | int | Chest pain type (0–3) |
| `trestbps` | int | Resting blood pressure (mm Hg) |
| `chol` | int | Serum cholesterol (mg/dl) |
| `fbs` | int | Fasting blood sugar > 120 mg/dl (1/0) |
| `restecg` | int | Resting ECG results (0–2) |
| `thalach` | int | Max heart rate achieved |
| `exang` | int | Exercise-induced angina (1/0) |
| `oldpeak` | float | ST depression by exercise |
| `slope` | int | Peak exercise ST segment slope (0–2) |
| `ca` | int | Major vessels colored by fluoroscopy (0–4) |
| `thal` | int | Thalassemia type (0–3) |

**Example Request (curl):**
```bash
curl -X POST http://localhost:8000/predict \
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

## 🐳 Docker Details

The `Dockerfile` uses a **multi-stage build**:
1. **Builder stage** — installs Python dependencies cleanly
2. **Runtime stage** — lean image with only production files + a non-root user

This keeps the final image small and secure.

---

## ☁️ Deployment on Render

1. Push this repository to GitHub
2. Log in to [Render](https://render.com)
3. Click **New +** → **Web Service**
4. Connect your GitHub repo
5. Set **Environment** → **Docker**
6. Set **Branch** → `main`
7. Click **Create Web Service**
8. Wait for the build (~2–5 minutes)
9. Your live URL will be: `https://YOUR_APP_NAME.onrender.com`

**Test the deployed API:**
```bash
curl https://YOUR_APP_NAME.onrender.com/health
curl https://YOUR_APP_NAME.onrender.com/info
```

> ⚠️ Render's free tier spins down after 15 min of inactivity. The first request after a cold start may take ~30 seconds.

---

## 🔗 Live Deployment

**API URL:** `https://YOUR_APP_NAME.onrender.com`  
**Swagger UI:** `https://YOUR_APP_NAME.onrender.com/docs`

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
