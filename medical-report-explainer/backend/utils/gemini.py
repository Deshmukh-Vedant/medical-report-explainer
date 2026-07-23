"""
utils/gemini.py
-----------------
Thin wrapper around the google-generativeai SDK. Handles:
  - Configuring the Gemini client with the API key from .env
  - Sending the structured analysis prompt and parsing JSON safely
  - Sending translation prompts
  - Defensive error handling so a bad/empty AI response never crashes
    the API — callers always get a usable dict or a clear exception.
"""

import os
import json
import re
import logging

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - defensive fallback for environments without the package
    genai = None
from dotenv import load_dotenv

from utils.prompt import build_analysis_prompt, build_translation_prompt

load_dotenv()

logger = logging.getLogger("gemini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY is not set. Falling back to local report analysis."
    )
elif genai is not None:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning(
        "google-generativeai is not available. Falling back to local report analysis."
    )


class GeminiError(Exception):
    """Raised when the Gemini API call fails or returns unusable data."""
    pass


def _get_model():
    if not GEMINI_API_KEY or genai is None:
        raise GeminiError(
            "Gemini is unavailable. Falling back to local analysis."
        )
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


def _fallback_analysis(report_text: str, patient_context: str = "") -> dict:
    """Create a simple structured analysis when Gemini is unavailable."""
    normalized = (report_text or "").strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    text_lower = normalized.lower()

    detected_tests = []
    abnormal_markers = []

    def add_marker(name: str, value: str, normal_range: str, status: str, meaning: str) -> None:
        detected_tests.append({
            "test_name": name,
            "normal_range": normal_range,
            "actual_value": value,
            "status": status,
            "meaning": meaning,
            "possible_causes": ["Further evaluation may be needed."],
            "lifestyle_suggestions": ["Follow your clinician's advice and keep hydrated."],
            "diet_suggestions": ["Maintain a balanced diet and avoid excessive salt or sugar."],
            "when_to_consult_doctor": "Please follow up with your healthcare provider if symptoms persist or worsen.",
        })

    for pattern, label, range_text, meaning in [
        (r"hba1c\s*[:#]?\s*([0-9.]+)", "HbA1c", "Below 5.7%", "Shows average blood sugar over the past 2-3 months."),
        (r"cholesterol\s*[:#]?\s*([0-9.]+)", "Cholesterol", "Below 200 mg/dL", "Shows overall cholesterol status."),
        (r"glucose\s*[:#]?\s*([0-9.]+)", "Glucose", "70-140 mg/dL", "Shows current blood sugar level."),
        (r"alt\s*[:#]?\s*([0-9.]+)", "ALT", "Up to 40 U/L", "Shows liver-related activity."),
        (r"ast\s*[:#]?\s*([0-9.]+)", "AST", "Up to 40 U/L", "Shows liver-related activity."),
        (r"creatinine\s*[:#]?\s*([0-9.]+)", "Creatinine", "0.6-1.2 mg/dL", "Shows kidney function."),
        (r"hemoglobin\s*[:#]?\s*([0-9.]+)", "Hemoglobin", "13.0-17.0 g/dL", "Shows oxygen-carrying capacity of the blood."),
    ]:
        match = re.search(pattern, text_lower)
        if match:
            value = match.group(1)
            name = label
            status = "Green"
            if label == "HbA1c" and float(value) > 6.5:
                status = "Yellow"
                abnormal_markers.append(name)
            elif label == "Cholesterol" and float(value) > 200:
                status = "Yellow"
                abnormal_markers.append(name)
            elif label == "Glucose" and float(value) > 140:
                status = "Yellow"
                abnormal_markers.append(name)
            elif label == "ALT" and float(value) > 40:
                status = "Yellow"
                abnormal_markers.append(name)
            elif label == "AST" and float(value) > 40:
                status = "Yellow"
                abnormal_markers.append(name)
            elif label == "Creatinine" and float(value) > 1.2:
                status = "Yellow"
                abnormal_markers.append(name)
            elif label == "Hemoglobin" and float(value) < 13:
                status = "Yellow"
                abnormal_markers.append(name)
            add_marker(name, value, range_text, status, meaning)

    if not detected_tests:
        detected_tests.append({
            "test_name": "General report review",
            "normal_range": "N/A",
            "actual_value": "Text was provided, but no standard test values were recognized.",
            "status": "Green",
            "meaning": "The system could not detect specific numeric markers from the provided text.",
            "possible_causes": ["No obvious markers were identified."],
            "lifestyle_suggestions": ["Continue routine healthy habits and share the report with a clinician."],
            "diet_suggestions": ["Maintain balanced meals and hydration."],
            "when_to_consult_doctor": "Contact your healthcare provider if you have symptoms or questions.",
        })

    abnormal_count = len(abnormal_markers)
    score = max(45, 90 - abnormal_count * 12)
    if abnormal_count >= 3:
        risk_level = "High"
    elif abnormal_count >= 1:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    patient_summary = (
        "This summary was generated from the uploaded text because the AI service was unavailable. "
        "Please review the report carefully with a qualified healthcare professional."
    )
    overall_summary = (
        f"The report text suggests {risk_level.lower()} concern based on the values detected. "
        f"{abnormal_count} marker(s) appear outside the standard range."
    )

    return {
        "patient_summary": patient_summary,
        "overall_health_summary": overall_summary,
        "key_findings": [
            "The report was analyzed using a built-in fallback parser.",
            *([f"Detected an unusual value for {name}." for name in abnormal_markers] if abnormal_markers else []),
        ],
        "health_score": round(score, 1),
        "risk_level": risk_level,
        "detected_tests": detected_tests,
        "diet_advice": ["Eat balanced meals with vegetables, protein, and whole grains."],
        "exercise_advice": ["Aim for light activity such as walking if cleared by your clinician."],
        "doctor_recommendation": "Please share this report with your doctor for a proper medical interpretation.",
        "red_flags": ["Any worsening symptoms or severe pain should be evaluated promptly."],
        "emergency_signs": ["Seek immediate emergency care for chest pain, trouble breathing, severe weakness, fainting, or sudden confusion."],
    }


def _extract_json(raw_text: str) -> dict:
    """
    Gemini sometimes wraps JSON in ```json ... ``` fences despite
    instructions not to. This strips any fencing/whitespace and
    safely parses the result.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Try to salvage the largest {...} block in the text as a fallback
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise GeminiError(f"Gemini did not return valid JSON: {exc}") from exc


def analyze_report(report_text: str, patient_context: str = "") -> dict:
    """
    Sends extracted report text to Gemini and returns the parsed
    structured analysis dict matching schemas.report.ReportAnalysis.

    Falls back to a local parser when the Gemini API is unavailable or
    returns unusable data.
    """
    if not report_text or not report_text.strip():
        raise GeminiError("No text was extracted from the uploaded report.")

    if len(report_text.strip()) < 80:
        logger.warning("Report text is too short for normal analysis; using fallback analysis.")
        return _fallback_analysis(report_text, patient_context)

    try:
        model = _get_model()
        prompt = build_analysis_prompt(report_text, patient_context)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 4096,
            },
        )
    except GeminiError:
        logger.warning("Gemini unavailable; using built-in fallback analysis.")
        return _fallback_analysis(report_text, patient_context)
    except Exception as exc:  # network / SDK errors
        logger.exception("Gemini API call failed")
        logger.warning("Using built-in fallback analysis after Gemini failure.")
        return _fallback_analysis(report_text, patient_context)

    if not response or not getattr(response, "text", None):
        logger.warning("Gemini returned an empty response; using fallback analysis.")
        return _fallback_analysis(report_text, patient_context)

    try:
        return _extract_json(response.text)
    except GeminiError as exc:
        logger.warning("Gemini returned unusable JSON: %s", exc)
        return _fallback_analysis(report_text, patient_context)


def translate_text(text: str, target_language: str) -> str:
    """
    Translates a block of explanation text into the target language
    ("mr" for Marathi). Returns plain translated text (not JSON).
    """
    if target_language == "en":
        return text  # no-op, already English

    model = _get_model()
    prompt = build_translation_prompt(text, target_language)

    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 2048},
        )
    except Exception as exc:
        logger.exception("Gemini translation call failed")
        raise GeminiError(f"Gemini translation request failed: {exc}") from exc

    if not response or not getattr(response, "text", None):
        raise GeminiError("Gemini returned an empty translation response.")

    return response.text.strip()
