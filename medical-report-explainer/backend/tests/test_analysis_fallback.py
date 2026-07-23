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


if __name__ == "__main__":
    unittest.main()
