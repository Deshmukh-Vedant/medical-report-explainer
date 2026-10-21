"""
routes/predict.py
------------------
Provides a manual prediction endpoint so the frontend sidebar tools
(Diabetes Risk, Heart Risk) can call the ML model directly without
requiring a PDF report.

POST /predict/manual
  - Accepts any combination of the 10 lab parameters
  - Missing values are handled by the model's internal imputer
  - Returns a structured prediction result
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field, ConfigDict

from models.user import User
from routes.auth import get_current_user
from ml.extractor import FEATURE_NAMES
from ml.predictor import predict, model_available, PredictionResult

logger = logging.getLogger("predict")

router = APIRouter(tags=["ML Prediction"])


class ManualPredictRequest(BaseModel):
    """All fields are optional; missing values are imputed by the model."""
    hemoglobin: Optional[float] = Field(None, description="Hemoglobin in g/dL")
    wbc: Optional[float] = Field(None, description="White blood cell count (10^3/µL)")
    platelets: Optional[float] = Field(None, description="Platelet count (10^3/µL)")
    glucose: Optional[float] = Field(None, description="Blood glucose in mg/dL")
    creatinine: Optional[float] = Field(None, description="Serum creatinine in mg/dL")
    alt: Optional[float] = Field(None, description="ALT/SGPT in U/L")
    ast: Optional[float] = Field(None, description="AST/SGOT in U/L")
    total_protein: Optional[float] = Field(None, description="Total protein in g/dL")
    hba1c: Optional[float] = Field(None, description="HbA1c in %")
    cholesterol: Optional[float] = Field(None, description="Total cholesterol in mg/dL")


class MLPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    risk_label: str
    confidence: float
    extracted_params: dict
    missing_params: list
    model_version: str
    disclaimer: str


@router.post("/predict/manual", response_model=MLPredictionResponse)
def manual_predict(
    payload: ManualPredictRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Runs the lab risk classifier on manually entered lab values.
    Requires authentication. All fields optional — model handles missing values.
    """
    if not model_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not available. Please contact the administrator.",
        )

    params = {
        "hemoglobin": payload.hemoglobin,
        "wbc": payload.wbc,
        "platelets": payload.platelets,
        "glucose": payload.glucose,
        "creatinine": payload.creatinine,
        "alt": payload.alt,
        "ast": payload.ast,
        "total_protein": payload.total_protein,
        "hba1c": payload.hba1c,
        "cholesterol": payload.cholesterol,
    }

    try:
        result: PredictionResult = predict(params)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an unexpected error.",
        )

    return MLPredictionResponse(
        risk_label=result.risk_label,
        confidence=result.confidence,
        extracted_params=result.extracted_params,
        missing_params=result.missing_params,
        model_version=result.model_version,
        disclaimer=result.disclaimer,
    )
