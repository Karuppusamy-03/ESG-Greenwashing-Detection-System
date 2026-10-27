"""Risk prediction must distinguish three states and never invent a score.

States:
1. No eligible evidence                 -> NOT_PREDICTED, no score, no category.
2. Evidence found, NLI not completed    -> NOT_PREDICTED (unassessed), no score,
                                           no category. Not LOW, not support,
                                           not contradiction.
3. NLI completed (counts, incl. zero)   -> HEURISTIC, scored as before.

All inputs are synthetic and fictional. No real company evidence is used.
"""

import unittest

import numpy as np
import pandas as pd

from explainability import explain_risk_predictions
from risk_model import FEATURE_COLUMNS
from risk_predictor import (
    NLI_UNAVAILABLE_REASON,
    NO_EVIDENCE_REASON,
    predict_risk_from_features,
)


def synthetic_row(claim_id, **overrides):
    row = {column: None for column in FEATURE_COLUMNS}
    row.update({
        "claim_id": claim_id,
        "claim": f"Synthetic claim {claim_id}.",
        "similarity": 0.6,
        "evidence_available": 1,
        "percentage_difference_missing": 0,
    })
    row.update(overrides)
    return row


NO_NLI = {
    "nli_entailment": None,
    "nli_contradiction": None,
    "nli_neutral": None,
}
COMPLETED_ZERO = {
    "nli_entailment": 0,
    "nli_contradiction": 0,
    "nli_neutral": 0,
}
CONTRADICTION = {
    "nli_entailment": 0,
    "nli_contradiction": 1,
    "nli_neutral": 0,
}


class ThreeStateTests(unittest.TestCase):
    def test_no_eligible_evidence_is_unassessed_with_reason(self):
        output = predict_risk_from_features(pd.DataFrame([
            synthetic_row(1, evidence_available=0, **NO_NLI),
        ]))
        row = output.iloc[0]
        self.assertEqual(row["risk_score_status"], "NOT_PREDICTED")
        self.assertTrue(pd.isna(row["risk_score"]))
        self.assertTrue(pd.isna(row["predicted_risk_class"]))
        self.assertEqual(row["demonstration_risk_category"], "NOT_PREDICTED")
        self.assertEqual(row["risk_unavailable_reason"], NO_EVIDENCE_REASON)

    def test_evidence_without_nli_is_unassessed_not_low(self):
        output = predict_risk_from_features(pd.DataFrame([
            synthetic_row(1, **NO_NLI),
        ]))
        row = output.iloc[0]
        self.assertEqual(row["risk_score_status"], "NOT_PREDICTED")
        self.assertTrue(pd.isna(row["risk_score"]))
        self.assertTrue(pd.isna(row["predicted_risk_class"]))
        self.assertEqual(row["demonstration_risk_category"], "NOT_PREDICTED")
        self.assertNotEqual(row["demonstration_risk_category"], "LOW")
        self.assertEqual(row["risk_unavailable_reason"], NLI_UNAVAILABLE_REASON)

    def test_completed_nli_with_zero_counts_is_still_scored(self):
        """A finished NLI run that found nothing is a real outcome."""
        output = predict_risk_from_features(pd.DataFrame([
            synthetic_row(1, **COMPLETED_ZERO),
        ]))
        row = output.iloc[0]
        self.assertEqual(row["risk_score_status"], "HEURISTIC")
        self.assertFalse(pd.isna(row["risk_score"]))
        self.assertEqual(row["demonstration_risk_category"], "LOW")
        self.assertEqual(row["risk_unavailable_reason"], "")

    def test_valid_contradiction_prediction_still_works(self):
        output = predict_risk_from_features(pd.DataFrame([
            synthetic_row(1, **CONTRADICTION),
        ]))
        row = output.iloc[0]
        self.assertEqual(row["risk_score_status"], "HEURISTIC")
        self.assertEqual(row["demonstration_risk_category"], "HIGH")
        self.assertEqual(int(row["predicted_risk_class"]), 1)
        self.assertEqual(row["risk_unavailable_reason"], "")

    def test_missing_nli_is_not_support_contradiction_or_low_risk(self):
        """Unavailable NLI must not be read as any NLI outcome."""
        output = predict_risk_from_features(pd.DataFrame([
            synthetic_row(1, **NO_NLI),
        ]))
        row = output.iloc[0]
        self.assertEqual(row["risk_score_status"], "NOT_PREDICTED")
        self.assertNotIn(row["demonstration_risk_category"], {"LOW", "MEDIUM", "HIGH"})


class MixedBatchTests(unittest.TestCase):
    def build_batch(self):
        return pd.DataFrame([
            synthetic_row(1, evidence_available=0, **NO_NLI),
            synthetic_row(2, **NO_NLI),
            synthetic_row(3, **COMPLETED_ZERO),
            synthetic_row(4, **CONTRADICTION),
            synthetic_row(5, **NO_NLI),
        ])

    def test_mixed_batch_produces_no_conversion_errors(self):
        output = predict_risk_from_features(self.build_batch())
        self.assertEqual(len(output), 5)
        self.assertEqual(
            output["risk_score_status"].tolist(),
            ["NOT_PREDICTED", "NOT_PREDICTED", "HEURISTIC", "HEURISTIC", "NOT_PREDICTED"],
        )

    def test_unassessed_rows_have_no_score_or_class(self):
        output = predict_risk_from_features(self.build_batch())
        unassessed = output[output["risk_score_status"] == "NOT_PREDICTED"]
        self.assertTrue(unassessed["risk_score"].isna().all())
        self.assertTrue(unassessed["predicted_risk_class"].isna().all())
        self.assertTrue(
            (unassessed["demonstration_risk_category"] == "NOT_PREDICTED").all()
        )

    def test_aggregates_exclude_unassessed_rows(self):
        output = predict_risk_from_features(self.build_batch())
        scored = output[output["risk_score"].notna()]
        self.assertEqual(len(scored), 2)
        categories = output["demonstration_risk_category"].value_counts()
        self.assertEqual(categories.get("LOW", 0), 1)
        self.assertEqual(categories.get("HIGH", 0), 1)
        self.assertEqual(categories.get("NOT_PREDICTED", 0), 3)
        self.assertEqual(
            scored["risk_score"].astype(float).mean(),
            np.mean([
                float(scored.iloc[0]["risk_score"]),
                float(scored.iloc[1]["risk_score"]),
            ]),
        )

    def test_reasons_distinguish_the_two_unassessed_states(self):
        output = predict_risk_from_features(self.build_batch())
        self.assertEqual(output.loc[0, "risk_unavailable_reason"], NO_EVIDENCE_REASON)
        self.assertEqual(output.loc[1, "risk_unavailable_reason"], NLI_UNAVAILABLE_REASON)
        self.assertEqual(output.loc[4, "risk_unavailable_reason"], NLI_UNAVAILABLE_REASON)
        self.assertEqual(output.loc[2, "risk_unavailable_reason"], "")
        self.assertEqual(output.loc[3, "risk_unavailable_reason"], "")

    def test_explainer_skips_unassessed_rows_without_crashing(self):
        """Explanations must cover only scored claims, never cast NA."""
        output = predict_risk_from_features(self.build_batch())
        explanations = explain_risk_predictions(output, None, "Synthetic Co")
        self.assertEqual(len(explanations), 2)
        self.assertEqual({item["claim_id"] for item in explanations}, {3, 4})
        for item in explanations:
            self.assertIsInstance(item["risk_score"], float)

    def test_explainer_with_only_unassessed_rows_returns_empty(self):
        output = predict_risk_from_features(pd.DataFrame([
            synthetic_row(1, **NO_NLI),
        ]))
        self.assertEqual(explain_risk_predictions(output, None, "Synthetic Co"), [])


class UnavailableCategoryChartTests(unittest.TestCase):
    """The chart builder must never count an unassessed row as LOW."""

    def test_category_counts_match_scored_rows_only(self):
        output = predict_risk_from_features(pd.DataFrame([
            synthetic_row(1, **NO_NLI),
            synthetic_row(2, **COMPLETED_ZERO),
        ]))
        chart_categories = [
            category
            for category in output["demonstration_risk_category"].tolist()
            if category in {"LOW", "MEDIUM", "HIGH"}
        ]
        self.assertEqual(chart_categories, ["LOW"])


if __name__ == "__main__":
    unittest.main()
