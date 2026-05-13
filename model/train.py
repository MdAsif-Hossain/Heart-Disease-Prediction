"""
Heart Disease Model Training Script
=====================================
Run this script ONCE to train the model and save it.
Usage: python model/train.py

Make sure you have placed the Kaggle heart.csv in the data/ folder first.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.joblib")

# ── Feature list (must match schemas.py) ──────────────────────────────────────
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]
TARGET = "target"


def load_data(path: str) -> pd.DataFrame:
    """Load and validate the CSV dataset."""
    if not os.path.exists(path):
        print(f"\n[ERR] Dataset not found at: {path}")
        print("    Please download heart.csv from:")
        print("    https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
        print("    and place it in the data/ folder.\n")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[OK]  Dataset loaded  ->  {len(df)} rows, {len(df.columns)} columns")

    # Validate required columns
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        print(f"[ERR] Missing columns in CSV: {missing}")
        sys.exit(1)

    # Drop duplicates and rows with nulls in required columns
    before = len(df)
    df = df[FEATURES + [TARGET]].dropna().drop_duplicates()
    print(f"[OK]  After cleaning   ->  {len(df)} rows  (removed {before - len(df)} rows)")

    # Binarise target: any value > 0 → 1 (some versions have 0–4)
    df[TARGET] = (df[TARGET] > 0).astype(int)
    print(f"[OK]  Target distribution:\n{df[TARGET].value_counts().to_string()}")
    return df


def build_pipeline() -> Pipeline:
    """Return a sklearn Pipeline: StandardScaler + Logistic Regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            solver="lbfgs"
        ))
    ])


def train(df: pd.DataFrame) -> Pipeline:
    """Split data, train, evaluate, and return fitted pipeline."""
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train size: {len(X_train)}   Test size: {len(X_test)}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print("\n[REPORT] Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["No Disease", "Heart Disease"]))
    return pipeline


def save_model(pipeline: Pipeline, path: str) -> None:
    """Persist the trained pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"[SAVED] Model saved   ->  {path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    print("=" * 55)
    print("  Heart Disease Classifier - Training Script")
    print("=" * 55)

    df       = load_data(DATA_PATH)
    pipeline = train(df)
    save_model(pipeline, MODEL_PATH)

    print("\n[DONE] Model training complete! You can now run the FastAPI app.")
    print("    Start server:  uvicorn app.main:app --reload")
    print("    Swagger UI:    http://localhost:8000/docs\n")
