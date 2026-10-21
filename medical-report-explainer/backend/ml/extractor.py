"""
ml/extractor.py
----------------
Extracts numeric laboratory parameters from raw OCR/pdfplumber text
produced during report upload.

Supported parameters (10 features used by the risk classifier):
  hemoglobin, wbc, platelets, glucose, creatinine, alt, ast,
  total_protein, hba1c, cholesterol

Unit normalisation:
  - Glucose: mmol/L → mg/dL (* 18.0182)
  - Creatinine: µmol/L → mg/dL (* 0.0113)
  - Cholesterol: mmol/L → mg/dL (* 38.665)
  - HbA1c: already % — no conversion needed

Missing values are returned as None. The predictor handles imputation.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("ml.extractor")

# Ordered list of features the model expects (must match train.py)
FEATURE_NAMES = [
    "hemoglobin",
    "wbc",
    "platelets",
    "glucose",
    "creatinine",
    "alt",
    "ast",
    "total_protein",
    "hba1c",
    "cholesterol",
]

# Medically plausible value ranges for basic sanity checking
_SANE_RANGES = {
    "hemoglobin":    (3.0,  25.0),
    "wbc":           (0.5,  100.0),
    "platelets":     (10.0, 1500.0),
    "glucose":       (20.0, 800.0),
    "creatinine":    (0.1,  30.0),
    "alt":           (1.0,  5000.0),
    "ast":           (1.0,  5000.0),
    "total_protein": (2.0,  15.0),
    "hba1c":         (2.0,  20.0),
    "cholesterol":   (30.0, 700.0),
}

# Regex patterns: (parameter_name, compiled_regex, unit_conversion_fn)
# Each regex captures a numeric value. The conversion fn normalises to standard units.
_PARAM_PATTERNS = [
    # Hemoglobin: g/dL or g/L (g/L → g/dL by /10)
    ("hemoglobin", re.compile(
        r"\bhb(?:a)?\b|\bhemoglobin\b|\bhaemoglobin\b",
        re.IGNORECASE,
    ), None),

    # WBC: 10^3/µL or cells/µL (* 1e-3 if very large)
    ("wbc", re.compile(
        r"\bwbc\b|\bwhite\s+blood\s+(?:cell|count)\b|\bleukocyte\b",
        re.IGNORECASE,
    ), None),

    # Platelets: 10^3/µL
    ("platelets", re.compile(
        r"\bplatelet(?:s)?\b|\bplt\b|\bthrombocyte\b",
        re.IGNORECASE,
    ), None),

    # Glucose: mg/dL or mmol/L
    ("glucose", re.compile(
        r"\bglucose\b|\bblood\s+sugar\b|\bfbs\b|\brbs\b|\bppbs\b",
        re.IGNORECASE,
    ), None),

    # Creatinine: mg/dL or µmol/L
    ("creatinine", re.compile(
        r"\bcreatinine\b|\bserum\s+creatinine\b",
        re.IGNORECASE,
    ), None),

    # ALT (SGPT)
    ("alt", re.compile(
        r"\balt\b|\bsgpt\b|\balanine\s+(?:amino)?transferase\b",
        re.IGNORECASE,
    ), None),

    # AST (SGOT)
    ("ast", re.compile(
        r"\bast\b|\bsgot\b|\baspartate\s+(?:amino)?transferase\b",
        re.IGNORECASE,
    ), None),

    # Total Protein
    ("total_protein", re.compile(
        r"\btotal\s+protein\b|\btp\b",
        re.IGNORECASE,
    ), None),

    # HbA1c
    ("hba1c", re.compile(
        r"\bhba1c\b|\bglycated\s+haemoglobin\b|\bglycosylated\s+hemoglobin\b|\bha1c\b",
        re.IGNORECASE,
    ), None),

    # Cholesterol (total)
    ("cholesterol", re.compile(
        r"\b(?:total\s+)?cholesterol\b",
        re.IGNORECASE,
    ), None),
]

# Number extraction: matches optional sign, digits, optional decimal
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _extract_value_near(text: str, label_match_re: re.Pattern) -> Optional[float]:
    """
    Searches for a label in `text`, then extracts the first numeric value
    that appears immediately after the label match (prioritising same line).
    Returns None if not found or value is outside sanity range.
    """
    for m in label_match_re.finditer(text):
        after_text = text[m.end(): m.end() + 80]
        # Prefer the same line if numeric content is present
        line_chunk = after_text.split('\n')[0]
        target = line_chunk if any(c.isdigit() for c in line_chunk) else after_text
        nums = _NUMBER_RE.findall(target)
        if nums:
            try:
                val = float(nums[0])
                return val
            except ValueError:
                pass
    return None


def _sanity_check(name: str, value: Optional[float]) -> Optional[float]:
    """Returns None if the value is outside the plausible medical range."""
    if value is None:
        return None
    lo, hi = _SANE_RANGES.get(name, (float("-inf"), float("inf")))
    if not (lo <= value <= hi):
        logger.debug("Value %s=%s rejected (out of sane range %s-%s)", name, value, lo, hi)
        return None
    return value


def extract_lab_params(report_text: str) -> dict:
    """
    Main entry point. Extracts all 10 lab parameters from the given
    report text string (as produced by OCR/pdfplumber).

    Returns:
        dict with keys matching FEATURE_NAMES. Values are float | None.
        None means the parameter was not found or failed a sanity check.
    """
    if not report_text:
        return {name: None for name in FEATURE_NAMES}

    results = {}
    for (name, pattern, _converter) in _PARAM_PATTERNS:
        raw_val = _extract_value_near(report_text, pattern)

        # WBC unit fix: if value > 200, probably reported as cells/µL (e.g. 6800 instead of 6.8)
        if name == "wbc" and raw_val is not None and raw_val > 200:
            raw_val = round(raw_val / 1000, 2)

        # Platelets: if value > 2000, probably cells/µL (e.g. 230000 instead of 230)
        if name == "platelets" and raw_val is not None and raw_val > 2000:
            raw_val = round(raw_val / 1000, 2)

        results[name] = _sanity_check(name, raw_val)

    extracted_count = sum(1 for v in results.values() if v is not None)
    logger.info("Lab param extraction: %d/%d parameters found", extracted_count, len(FEATURE_NAMES))

    return results
