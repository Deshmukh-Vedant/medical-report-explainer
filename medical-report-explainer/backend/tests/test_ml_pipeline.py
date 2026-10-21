"""
tests/test_ml_pipeline.py
--------------------------
Unit and regression tests for:
1. Lab parameter regex extraction (ml.extractor)
2. Value normalization (WBC/platelets scale)
3. Sanity range checks
4. ML model inference and confidence scores (ml.predictor)
"""

import unittest
from ml.extractor import extract_lab_params, FEATURE_NAMES, _sanity_check
from ml.predictor import predict, model_available, PredictionResult


class MLPipelineTests(unittest.TestCase):

    def test_model_is_available(self):
        """Verifies the trained lab_risk_classifier.pkl is present and loadable."""
        self.assertTrue(model_available(), "Trained ML model artifact should exist")

    def test_extract_lab_params_comprehensive(self):
        """Tests regex extraction from realistic laboratory report text."""
        sample_report = """
        METROPOLIS CLINICAL LABORATORIES
        Patient: John Doe   Age: 45   Gender: Male
        -------------------------------------------
        COMPLETE BLOOD COUNT:
        Hemoglobin: 14.2 g/dL       (Ref: 13.0 - 17.0)
        WBC Count: 6800 /uL         (Ref: 4000 - 11000)
        Platelet Count: 230000 /uL  (Ref: 150000 - 450000)

        BIOCHEMISTRY & LIPID PANEL:
        Fasting Blood Glucose: 98 mg/dL  (Ref: 70 - 100)
        Serum Creatinine: 0.9 mg/dL      (Ref: 0.6 - 1.2)
        Total Cholesterol: 185 mg/dL     (Ref: < 200)
        HbA1c: 5.4 %                     (Ref: < 5.7)
        SGPT (ALT): 28 U/L               (Ref: < 45)
        SGOT (AST): 24 U/L               (Ref: < 40)
        Total Protein: 7.1 g/dL          (Ref: 6.0 - 8.3)
        """
        extracted = extract_lab_params(sample_report)

        self.assertAlmostEqual(extracted["hemoglobin"], 14.2, places=1)
        # 6800 cells/uL scaled to 6.8 (10^3/uL)
        self.assertAlmostEqual(extracted["wbc"], 6.8, places=1)
        # 230000 scaled to 230.0 (10^3/uL)
        self.assertAlmostEqual(extracted["platelets"], 230.0, places=1)
        self.assertAlmostEqual(extracted["glucose"], 98.0, places=1)
        self.assertAlmostEqual(extracted["creatinine"], 0.9, places=1)
        self.assertAlmostEqual(extracted["cholesterol"], 185.0, places=1)
        self.assertAlmostEqual(extracted["hba1c"], 5.4, places=1)
        self.assertAlmostEqual(extracted["alt"], 28.0, places=1)
        self.assertAlmostEqual(extracted["ast"], 24.0, places=1)
        self.assertAlmostEqual(extracted["total_protein"], 7.1, places=1)

    def test_sanity_check_rejects_implausible_values(self):
        """Ensures extreme/erroneous values are rejected."""
        self.assertIsNone(_sanity_check("glucose", 5.0))      # Too low for life
        self.assertIsNone(_sanity_check("glucose", 9500.0))   # Impossibly high
        self.assertIsNotNone(_sanity_check("glucose", 120.0)) # Valid

    def test_predict_normal_profile(self):
        """A normal lab panel should classify as Low risk with reasonable confidence."""
        normal_params = {
            "hemoglobin": 14.0,
            "wbc": 6.5,
            "platelets": 250.0,
            "glucose": 92.0,
            "creatinine": 0.9,
            "alt": 22.0,
            "ast": 20.0,
            "total_protein": 7.2,
            "hba1c": 5.2,
            "cholesterol": 175.0,
        }
        res: PredictionResult = predict(normal_params)
        self.assertEqual(res.risk_label, "Low")
        self.assertGreaterEqual(res.confidence, 0.5)
        self.assertEqual(len(res.missing_params), 0)

    def test_predict_high_risk_profile(self):
        """Critical diabetic / renal markers should classify as High risk."""
        high_risk_params = {
            "glucose": 380.0,
            "hba1c": 11.5,
            "creatinine": 6.2,
        }
        res: PredictionResult = predict(high_risk_params)
        self.assertEqual(res.risk_label, "High")
        self.assertIn("hemoglobin", res.missing_params)
        self.assertIn("cholesterol", res.missing_params)


if __name__ == "__main__":
    unittest.main()
