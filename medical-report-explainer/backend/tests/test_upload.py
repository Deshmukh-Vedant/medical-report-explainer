"""
tests/test_upload.py
--------------------
Tests medical report PDF upload, validation, extraction,
and subsequent AI/fallback analysis.
"""

import io
import unittest
from datetime import datetime
import fitz
from fastapi.testclient import TestClient

from main import app
from models.database import init_db


class UploadWorkflowTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)
        cls.email = f"upload_test_{int(datetime.utcnow().timestamp())}@example.com"
        cls.password = "UploadPassword123!"

        # Register and get auth token
        reg_res = cls.client.post("/register", json={
            "name": "Alex Mercer",
            "email": cls.email,
            "password": cls.password,
            "confirm_password": cls.password,
        })
        assert reg_res.status_code == 201
        cls.token = reg_res.json()["access_token"]
        cls.headers = {"Authorization": f"Bearer {cls.token}"}

    def _create_sample_pdf(self, content_text: str) -> bytes:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), content_text)
        pdf_bytes = doc.write()
        doc.close()
        return pdf_bytes

    def test_01_upload_valid_pdf(self):
        pdf_content = (
            "DIAGNOSTIC PATHOLOGY LABORATORY\n"
            "Patient: Alex Mercer   Age: 38\n"
            "Hemoglobin: 14.5 g/dL (Normal: 13.0-17.0)\n"
            "Fasting Glucose: 102 mg/dL (Normal: 70-100)\n"
            "Total Cholesterol: 195 mg/dL (Normal: < 200)\n"
            "Serum Creatinine: 1.0 mg/dL (Normal: 0.6-1.2)\n"
        )
        pdf_bytes = self._create_sample_pdf(pdf_content)

        files = {"file": ("blood_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        res = self.client.post("/upload?hospital_name=City+Hospital", files=files, headers=self.headers)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn("report_id", data)
        self.__class__.report_id = data["report_id"]

    def test_02_upload_rejects_non_pdf(self):
        files = {"file": ("malicious.exe", io.BytesIO(b"fake executable"), "application/octet-stream")}
        res = self.client.post("/upload", files=files, headers=self.headers)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Only PDF files are accepted", res.json()["detail"])

    def test_03_upload_rejects_empty_file(self):
        files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
        res = self.client.post("/upload", files=files, headers=self.headers)
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json()["detail"].lower())

    def test_04_analyze_uploaded_report(self):
        res = self.client.post(f"/analyze/{self.report_id}", headers=self.headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("patient_summary", data)
        self.assertIn("health_score", data)
        self.assertIn("risk_level", data)
        # Check that ML prediction is present
        if data.get("ml_prediction"):
            self.assertIn("risk_label", data["ml_prediction"])
            self.assertIn("confidence", data["ml_prediction"])

    def test_05_view_analyzed_report_in_history(self):
        hist_res = self.client.get("/history", headers=self.headers)
        self.assertEqual(hist_res.status_code, 200)
        reports = hist_res.json()
        self.assertGreater(len(reports), 0)
        found = any(r["id"] == self.report_id for r in reports)
        self.assertTrue(found, "Uploaded and analyzed report must appear in user history")


if __name__ == "__main__":
    unittest.main()
