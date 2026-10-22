"""
ml/train.py
-----------
One-time training script for the Lab Report Risk Classifier.

Run from the `backend/` directory:
    python -m ml.train

Dataset provenance:
  Uses programmatically generated synthetic data with medically-informed
  distributions derived from published clinical reference ranges:
    - WHO haematology reference intervals
    - ADA glucose/HbA1c diagnostic thresholds
    - NCEP ATP III lipid guidelines
    - Standard biochemistry reference tables

  This is NOT derived from real patient records. It is intended for
  educational demonstration only and has NOT been clinically validated.
  Do not use predictions for diagnostic or treatment decisions.

Model:
  sklearn Pipeline: SimpleImputer(median) → RandomForestClassifier
  Labels: 0=Low risk, 1=Moderate risk, 2=High risk

To retrain on real data:
  Replace the generate_synthetic_data() call with your own DataFrame
  containing the same 10 feature columns and a 'risk_label' column
  with values 0, 1, or 2.
"""

import os
import sys
import logging
import warnings

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ml.train")

FEATURE_NAMES = [
    "hemoglobin",   # g/dL
    "wbc",          # 10^3/µL
    "platelets",    # 10^3/µL
    "glucose",      # mg/dL
    "creatinine",   # mg/dL
    "alt",          # U/L
    "ast",          # U/L
    "total_protein",# g/dL
    "hba1c",        # %
    "cholesterol",  # mg/dL
]

LABEL_NAMES = ["Low", "Moderate", "High"]


def generate_synthetic_data(n_samples: int = 12000, random_state: int = 42) -> tuple:
    """
    Generates synthetic lab data with medically-informed distributions.
    Each sample is assigned a risk label based on rule-based thresholds
    that approximate clinical decision support logic.

    Returns (X: ndarray of shape [n, 10], y: ndarray of shape [n])
    """
    rng = np.random.default_rng(random_state)
    n = n_samples

    # Generate each feature independently from realistic distributions
    # ~85% of samples are within normal range; ~15% have abnormal values to create class balance
    hemoglobin = rng.normal(14.0, 2.5, n).clip(3, 25)
    wbc        = rng.normal(7.0,  3.0, n).clip(0.5, 50)
    platelets  = rng.normal(250,  80,  n).clip(20, 900)
    glucose    = rng.normal(95,   35,  n).clip(30, 600)
    creatinine = rng.normal(1.0,  0.5, n).clip(0.3, 15)
    alt        = np.abs(rng.exponential(25, n)).clip(1, 2000)
    ast        = np.abs(rng.exponential(22, n)).clip(1, 2000)
    total_prot = rng.normal(7.2,  0.8, n).clip(3, 12)
    hba1c      = rng.normal(5.5,  1.2, n).clip(3, 15)
    cholesterol= rng.normal(185,  45,  n).clip(80, 600)

    X = np.column_stack([
        hemoglobin, wbc, platelets, glucose, creatinine,
        alt, ast, total_prot, hba1c, cholesterol
    ])

    # Rule-based labelling (mimics clinical flag logic)
    y = np.zeros(n, dtype=int)  # default: Low (0)

    # Moderate risk flags
    moderate_mask = (
        (hemoglobin < 11) | (hemoglobin > 18) |
        (wbc < 3.5) | (wbc > 11) |
        (platelets < 130) | (platelets > 400) |
        (glucose > 110) | (glucose < 60) |
        (creatinine > 1.3) |
        (alt > 45) | (ast > 45) |
        (hba1c > 5.7) |
        (cholesterol > 200)
    )
    y[moderate_mask] = 1

    # High risk flags (override moderate)
    high_mask = (
        (hemoglobin < 7) | (hemoglobin > 21) |
        (wbc < 2) | (wbc > 30) |
        (platelets < 50) | (platelets > 800) |
        (glucose > 300) | (glucose < 40) |
        (creatinine > 5) |
        (alt > 200) | (ast > 200) |
        (hba1c > 9) |
        (cholesterol > 350)
    )
    y[high_mask] = 2

    # Introduce 10% random NaN values to simulate missing OCR extractions
    mask = rng.random((n, 10)) < 0.10
    X_with_nan = X.astype(float)
    X_with_nan[mask] = np.nan

    return X_with_nan, y


def train_and_save(output_path: str = None) -> None:
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "models", "lab_risk_classifier.pkl")

    # Skip training if model already exists (avoids slow rebuild in Docker layer cache)
    if os.path.exists(output_path):
        logger.info("ML model already exists at %s — skipping training.", output_path)
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Generating synthetic training data...")
    X, y = generate_synthetic_data(n_samples=12000)
    logger.info("Data shape: %s | Label distribution: Low=%d, Moderate=%d, High=%d",
                X.shape, (y == 0).sum(), (y == 1).sum(), (y == 2).sum())

    # Build pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Stratified 5-fold CV
    logger.info("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    logger.info("CV Accuracy: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # Final train/test split for classification report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    logger.info("\nClassification Report:\n%s",
                classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    # Refit on all data before saving
    pipeline.fit(X, y)

    # Save
    model_artifact = {
        "pipeline": pipeline,
        "feature_names": FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "version": "1.0.0-lab-risk",
        "training_samples": len(X),
        "disclaimer": (
            "Trained on synthetic data. NOT clinically validated. "
            "For educational/demonstration purposes only."
        ),
    }
    joblib.dump(model_artifact, output_path)
    logger.info("Model saved to: %s", output_path)


if __name__ == "__main__":
    train_and_save()
