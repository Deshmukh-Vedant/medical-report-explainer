import unittest

from models.report import Report
from routes.analysis import build_fallback_analysis


class FallbackAnalysisTests(unittest.TestCase):
    def test_build_fallback_analysis_uses_existing_report_fields(self):
        report = Report(
            summary="Stored summary text",
            health_score=78,
            risk_level="Moderate",
        )

        analysis = build_fallback_analysis(report)

        self.assertEqual(analysis.patient_summary, "Stored summary text")
        self.assertEqual(analysis.overall_health_summary, "Stored summary text")
        self.assertEqual(analysis.health_score, 78.0)
        self.assertEqual(analysis.risk_level, "Moderate")

    def test_fallback_analysis_does_not_default_to_perfect_score(self):
        text = "HbA1c: 8.2%\nCholesterol: 240 mg/dL\nALT: 55 U/L"
        analysis = __import__('routes.analysis', fromlist=['']).build_fallback_analysis(
            Report(
                summary=text,
                health_score=None,
                risk_level=None,
            )
        )

        self.assertLess(analysis.health_score, 90)
        self.assertIn(analysis.risk_level, {"Moderate", "High"})


if __name__ == "__main__":
    unittest.main()
