import unittest

import numpy as np
import pandas as pd

import app


class AnalysisHelpersTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "measurement": [1.0, 2.0, 2.0, 3.0, 100.0, np.nan, 2.0],
                "group": ["a", "a", "b", "b", "c", "c", "a"],
            }
        )

    def test_iqr_outlier_detection(self):
        counts = app.outlier_counts(self.frame)
        self.assertEqual(int(counts["measurement"]), 1)

    def test_cleaning_fills_missing_removes_duplicates_and_caps(self):
        cleaned = app.clean_dataset(
            self.frame,
            missing_action="Fill numeric with median and categories with mode",
            remove_duplicates=True,
            outlier_action="Cap values at IQR limits",
        )
        self.assertEqual(int(cleaned.isna().sum().sum()), 0)
        self.assertEqual(int(cleaned.duplicated().sum()), 0)
        self.assertLess(float(cleaned["measurement"].max()), 100.0)

    def test_cleaning_can_drop_rows_and_outliers(self):
        cleaned = app.clean_dataset(
            self.frame,
            missing_action="Drop rows with missing values",
            outlier_action="Remove rows with IQR outliers",
        )
        self.assertEqual(len(cleaned), 5)
        self.assertNotIn(100.0, cleaned["measurement"].values)

    def test_insights_and_report_are_informative(self):
        observations = app.automatic_insights(self.frame)
        self.assertTrue(any("missing" in item.lower() for item in observations))
        report = app.make_report(self.frame, "test.csv")
        self.assertIn("DataLens Analysis Report", report)
        self.assertIn("test.csv", report)
        self.assertIn("measurement", report)

    def test_ml_pipeline_handles_numeric_and_categorical_features(self):
        frame = pd.DataFrame(
            {
                "number": [1, 2, 3, 4, 5, 6],
                "category": ["x", "y", "x", "y", "x", "y"],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        model, numeric, categorical = app.model_pipeline(frame, "target", "classification")
        self.assertEqual(numeric, ["number"])
        self.assertEqual(categorical, ["category"])
        model.fit(frame.drop(columns="target"), frame["target"])
        self.assertEqual(len(model.predict(frame.drop(columns="target"))), len(frame))


if __name__ == "__main__":
    unittest.main()
