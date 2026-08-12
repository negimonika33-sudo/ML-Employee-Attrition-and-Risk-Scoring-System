"""End-to-end checks for the supplied data, trained model, and deliverables."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from workforce_assistant import (  # noqa: E402
    add_engineered_features,
    categorize_probability,
    load_dataset,
    load_model_bundle,
    model_features,
)
from workforce_assistant.features import ENGINEERED_FEATURES  # noqa: E402


class MachineLearningProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, cls.quality = load_dataset(
            PROJECT_DIR / "data" / "Palo Alto Networks.csv"
        )
        cls.bundle = load_model_bundle(PROJECT_DIR / "artifacts" / "model_bundle.joblib")

    def test_dataset_quality_and_employee_ids(self) -> None:
        self.assertEqual(self.quality["rows"], 1470)
        self.assertEqual(self.quality["columns"], 31)
        self.assertEqual(self.quality["missing_cells"], 0)
        self.assertEqual(self.quality["duplicate_rows"], 0)
        self.assertEqual(self.data.iloc[0]["EmployeeID"], "EMP-0001")
        self.assertEqual(self.data.iloc[-1]["EmployeeID"], "EMP-1470")
        self.assertEqual(self.data["EmployeeID"].nunique(), 1470)

    def test_all_required_engineered_features(self) -> None:
        featured = add_engineered_features(self.data)
        for feature in ENGINEERED_FEATURES:
            self.assertIn(feature, featured.columns)
            self.assertTrue(np.isfinite(featured[feature]).all())
        self.assertEqual(featured["WorkloadStressFlag"].isin([0, 1]).all(), True)

    def test_sensitive_and_target_fields_are_not_model_inputs(self) -> None:
        inputs = model_features(self.data)
        for excluded in ["Attrition", "AttritionStatus", "EmployeeID", "Gender", "MaritalStatus"]:
            self.assertNotIn(excluded, inputs.columns)

    def test_all_required_models_and_metrics_exist(self) -> None:
        metrics = self.bundle["metrics"]
        self.assertEqual(
            set(metrics["Model"]),
            {"Logistic Regression", "Random Forest", "Gradient Boosting"},
        )
        for metric in ["Accuracy", "Precision", "Recall", "F1Score", "ROCAUC"]:
            self.assertIn(metric, metrics.columns)
            self.assertTrue(metrics[metric].between(0, 1).all())

    def test_stratified_split_sizes(self) -> None:
        self.assertEqual(self.bundle["training_rows"], 1176)
        self.assertEqual(self.bundle["testing_rows"], 294)
        expected_training_cases = round(int(self.data["Attrition"].sum()) * 0.8)
        self.assertLessEqual(
            abs(self.bundle["positive_training_rows"] - expected_training_cases), 1
        )

    def test_every_employee_has_probability_and_category(self) -> None:
        scores = self.bundle["risk_scores"]
        self.assertEqual(len(scores), 1470)
        self.assertTrue(scores["AttritionProbability"].between(0, 1).all())
        self.assertEqual(set(scores["RiskCategory"]), {"Low", "Medium", "High"})
        self.assertFalse(scores["ReasonCodes"].isna().any())

    def test_risk_threshold_boundaries(self) -> None:
        self.assertEqual(categorize_probability(0.2999), "Low")
        self.assertEqual(categorize_probability(0.30), "Medium")
        self.assertEqual(categorize_probability(0.60), "Medium")
        self.assertEqual(categorize_probability(0.6001), "High")

    def test_model_can_score_a_new_scenario(self) -> None:
        scenario = self.data.iloc[[0]].copy()
        scenario.loc[:, "OverTime"] = "No"
        scenario.loc[:, "JobSatisfaction"] = 4
        probability = float(
            self.bundle["model"].predict_proba(model_features(scenario))[0, 1]
        )
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_feature_importance_is_available(self) -> None:
        importance = self.bundle["feature_importance"]
        self.assertFalse(importance.empty)
        self.assertIn("Feature", importance.columns)
        self.assertIn("Importance", importance.columns)
        self.assertAlmostEqual(float(importance["Importance"].sum()), 1.0, places=6)

    def test_submission_reports_exist(self) -> None:
        research = PROJECT_DIR / "reports" / "research_paper.md"
        summary = PROJECT_DIR / "reports" / "executive_summary.md"
        self.assertTrue(research.exists())
        self.assertTrue(summary.exists())
        research_text = research.read_text(encoding="utf-8")
        self.assertIn("Exploratory data analysis", research_text)
        self.assertIn("Attrition rate", research_text)
        self.assertIn("Model evaluation", research_text)
        self.assertIn("Government Stakeholders", summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
