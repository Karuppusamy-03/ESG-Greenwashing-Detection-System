"""Regression tests: a failed NLI run must never be scored as a completed outcome.

Defect (reproduced before the fix): contradiction_detector marked failed
classifier outputs as UNAVAILABLE but left scoring_eligible=True (inherited from
the pair). feature_builder then counted them as zero-count completed NLI runs,
and the risk heuristic scored the claim in the LOW band with status HEURISTIC.
Evidence without a usable NLI result must stay NOT_PREDICTED.
"""

import unittest
from unittest import mock

import pandas as pd

import contradiction_detector
from feature_builder import build_features_from_results
from risk_predictor import NLI_UNAVAILABLE_REASON, predict_risk_from_features

CLAIM = "We committed to renewable energy across our operations."
EVIDENCE = "The company committed to renewable energy across its operations."
SOURCE_URL = "https://independent.example/report.pdf"


def _matching(claim=CLAIM):
    return [{
        "claim": claim,
        "matches": [{
            "matched_evidence": EVIDENCE,
            "similarity": 0.8,
            "match_label": "potentially_relevant",
            "source_url": SOURCE_URL,
            "scoring_eligible": True,
        }],
    }]


class UnavailableNliIsNotScoredTests(unittest.TestCase):
    def test_unavailable_only_is_not_predicted(self):
        nli = [{
            "claim": CLAIM,
            "evidence": EVIDENCE,
            "source_url": SOURCE_URL,
            "nli_result": "UNAVAILABLE",
            "scoring_eligible": False,
            "nli_failed": True,
        }]
        features = build_features_from_results(_matching(), nli)
        row = predict_risk_from_features(features).iloc[0]

        self.assertEqual(row["risk_score_status"], "NOT_PREDICTED")
        self.assertTrue(pd.isna(row["risk_score"]))
        self.assertEqual(row["demonstration_risk_category"], "NOT_PREDICTED")
        self.assertEqual(row["risk_unavailable_reason"], NLI_UNAVAILABLE_REASON)
        # Evidence was found; only the NLI step failed.
        self.assertEqual(int(row["evidence_available"]), 1)

    def test_unavailable_flag_without_scoring_eligible_key_is_not_counted(self):
        # Older saved results may lack the scoring_eligible key entirely.
        nli = [{
            "claim": CLAIM,
            "evidence": EVIDENCE,
            "source_url": SOURCE_URL,
            "nli_result": "UNAVAILABLE",
        }]
        features = build_features_from_results(_matching(), nli)
        self.assertTrue(pd.isna(features.iloc[0]["nli_entailment"]))
        row = predict_risk_from_features(features).iloc[0]
        self.assertEqual(row["risk_score_status"], "NOT_PREDICTED")

    def test_completed_pair_still_counts_when_another_pair_failed(self):
        nli = [
            {
                "claim": CLAIM,
                "evidence": EVIDENCE,
                "source_url": SOURCE_URL,
                "nli_result": "ENTAILMENT",
                "scoring_eligible": True,
            },
            {
                "claim": CLAIM,
                "evidence": "Another statement.",
                "source_url": SOURCE_URL,
                "nli_result": "UNAVAILABLE",
                "scoring_eligible": False,
                "nli_failed": True,
            },
        ]
        features = build_features_from_results(_matching(), nli)
        row = features.iloc[0]
        self.assertEqual(int(row["nli_entailment"]), 1)
        self.assertEqual(int(row["nli_contradiction"]), 0)
        self.assertEqual(int(row["nli_neutral"]), 0)
        scored = predict_risk_from_features(features).iloc[0]
        self.assertEqual(scored["risk_score_status"], "HEURISTIC")

    def test_zero_count_completed_run_is_still_scored(self):
        # A real completed NLI run with zero counts is an outcome and stays scored.
        nli = [{
            "claim": CLAIM,
            "evidence": EVIDENCE,
            "source_url": SOURCE_URL,
            "nli_result": "NEUTRAL",
            "scoring_eligible": True,
        }]
        features = build_features_from_results(_matching(), nli)
        self.assertEqual(int(features.iloc[0]["nli_neutral"]), 1)
        scored = predict_risk_from_features(features).iloc[0]
        self.assertEqual(scored["risk_score_status"], "HEURISTIC")


class EligibilityExclusionStillWorksTests(unittest.TestCase):
    def test_incompatible_pair_still_removes_evidence(self):
        # Eligibility exclusions (not NLI failures) must keep removing evidence.
        nli = [{
            "claim": CLAIM,
            "evidence": EVIDENCE,
            "source_url": SOURCE_URL,
            "nli_result": "UNAVAILABLE",
            "scoring_eligible": False,
            "compatibility_status": "INCOMPATIBLE",
        }]
        row = build_features_from_results(_matching(), nli).iloc[0]
        self.assertEqual(int(row["evidence_available"]), 0)


class DetectorMarksFailedOutputsTests(unittest.TestCase):
    def _run_detector(self, classifier_outputs):
        data = {
            "results": [{
                "claim": CLAIM,
                "matches": [{
                    "matched_evidence": EVIDENCE,
                    "similarity": 0.8,
                    "match_label": "potentially_relevant",
                    "source_url": SOURCE_URL,
                }],
            }],
        }

        def fake_classifier(inputs, **kwargs):
            return classifier_outputs[: len(inputs)]

        with mock.patch.object(
            contradiction_detector,
            "_get_nli_classifier",
            return_value=fake_classifier,
        ):
            return contradiction_detector.detect_online_contradictions(data)

    def test_missing_label_is_marked_not_scoring_eligible(self):
        results = self._run_detector([[{"score": 0.9}]])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["nli_result"], "UNAVAILABLE")
        self.assertIs(results[0]["scoring_eligible"], False)

    def test_unrecognized_structure_is_marked_not_scoring_eligible(self):
        results = self._run_detector(["not-a-dict"])
        self.assertEqual(results[0]["nli_result"], "UNAVAILABLE")
        self.assertIs(results[0]["scoring_eligible"], False)

    def test_unknown_label_is_marked_not_scoring_eligible(self):
        results = self._run_detector([[{"label": "something-else", "score": 0.4}]])
        self.assertEqual(results[0]["nli_result"], "UNAVAILABLE")
        self.assertIs(results[0]["scoring_eligible"], False)

    def test_failed_detector_output_end_to_end_is_not_predicted(self):
        nli = self._run_detector([[{"score": 0.9}]])
        features = build_features_from_results(_matching(), nli)
        row = predict_risk_from_features(features).iloc[0]
        self.assertEqual(row["risk_score_status"], "NOT_PREDICTED")
        self.assertEqual(row["risk_unavailable_reason"], NLI_UNAVAILABLE_REASON)

    def test_valid_entailment_is_scoring_eligible(self):
        results = self._run_detector([[{"label": "entailment", "score": 0.93}]])
        self.assertEqual(results[0]["nli_result"], "ENTAILMENT")
        self.assertIs(results[0]["scoring_eligible"], True)


if __name__ == "__main__":
    unittest.main()


from risk_predictor import NLI_PARTIAL_REASON  # noqa: E402

URL = "https://independent.example/report.pdf"
EV_A = "The company committed to renewable energy across its operations."
EV_B = "Renewable electricity supplied 60% of operations in 2024."


def _two_match_claim(claim=CLAIM):
    return [{
        "claim": claim,
        "matches": [
            {"matched_evidence": EV_A, "similarity": 0.8,
             "match_label": "potentially_relevant", "source_url": URL,
             "scoring_eligible": True},
            {"matched_evidence": EV_B, "similarity": 0.6,
             "match_label": "potentially_relevant", "source_url": URL,
             "scoring_eligible": True},
        ],
    }]


def _nli(evidence, label="ENTAILMENT", claim=CLAIM):
    return {"claim": claim, "evidence": evidence, "source_url": URL,
            "nli_result": label, "scoring_eligible": True}


class NliCoveragePolicyTests(unittest.TestCase):
    def test_complete_coverage_is_scored_on_all_matches(self):
        features = build_features_from_results(
            _two_match_claim(),
            [_nli(EV_A, "ENTAILMENT"), _nli(EV_B, "NEUTRAL")],
        )
        row = features.iloc[0]
        self.assertEqual(int(row["nli_pairs_eligible"]), 2)
        self.assertEqual(int(row["nli_pairs_completed"]), 2)
        self.assertEqual(int(row["nli_coverage_complete"]), 1)
        self.assertEqual(int(row["nli_entailment"]), 1)
        self.assertEqual(int(row["nli_neutral"]), 1)
        scored = predict_risk_from_features(features).iloc[0]
        self.assertEqual(scored["risk_score_status"], "HEURISTIC")

    def test_partial_coverage_is_not_scored_and_keeps_evidence(self):
        features = build_features_from_results(
            _two_match_claim(),
            [_nli(EV_A, "CONTRADICTION")],  # EV_B was never evaluated
        )
        row = features.iloc[0]
        self.assertEqual(int(row["evidence_available"]), 1)
        self.assertEqual(int(row["nli_pairs_eligible"]), 2)
        self.assertEqual(int(row["nli_pairs_completed"]), 1)
        self.assertEqual(int(row["nli_coverage_complete"]), 0)
        # Similarity still reflects all eligible matches (max over both).
        self.assertAlmostEqual(float(row["similarity"]), 0.8)
        predicted = predict_risk_from_features(features).iloc[0]
        self.assertEqual(predicted["risk_score_status"], "NOT_PREDICTED")
        self.assertTrue(pd.isna(predicted["risk_score"]))
        self.assertTrue(pd.isna(predicted["predicted_risk_class"]))
        self.assertEqual(predicted["risk_unavailable_reason"], NLI_PARTIAL_REASON)
        self.assertEqual(predicted["demonstration_risk_category"], "NOT_PREDICTED")

    def test_zero_coverage_is_unavailable_not_partial(self):
        features = build_features_from_results(_two_match_claim(), [])
        row = features.iloc[0]
        self.assertEqual(int(row["evidence_available"]), 1)
        self.assertEqual(int(row["nli_pairs_completed"]), 0)
        self.assertEqual(int(row["nli_coverage_complete"]), 0)
        self.assertTrue(pd.isna(row["nli_entailment"]))
        predicted = predict_risk_from_features(features).iloc[0]
        self.assertEqual(predicted["risk_score_status"], "NOT_PREDICTED")
        self.assertEqual(predicted["risk_unavailable_reason"], NLI_UNAVAILABLE_REASON)

    def test_nli_row_for_unmatched_evidence_does_not_inflate_coverage(self):
        other = "An unrelated sentence the matcher never selected."
        features = build_features_from_results(
            _two_match_claim(),
            [_nli(EV_A, "ENTAILMENT"), _nli(other, "ENTAILMENT")],
        )
        row = features.iloc[0]
        self.assertEqual(int(row["nli_pairs_completed"]), 1)
        self.assertEqual(int(row["nli_coverage_complete"]), 0)
        self.assertEqual(int(row["nli_entailment"]), 1)  # only the matched pair
        self.assertEqual(
            predict_risk_from_features(features).iloc[0]["risk_score_status"],
            "NOT_PREDICTED",
        )

    def test_failed_pair_counts_as_incomplete_coverage(self):
        failed = _nli(EV_B, "UNAVAILABLE")
        failed.update({"scoring_eligible": False, "nli_failed": True})
        features = build_features_from_results(
            _two_match_claim(), [_nli(EV_A, "NEUTRAL"), failed]
        )
        row = features.iloc[0]
        self.assertEqual(int(row["nli_coverage_complete"]), 0)
        self.assertEqual(int(row["evidence_available"]), 1)
        self.assertEqual(
            predict_risk_from_features(features).iloc[0]["risk_score_status"],
            "NOT_PREDICTED",
        )

    def test_legacy_rows_without_coverage_column_keep_previous_behaviour(self):
        legacy = pd.DataFrame([{
            "claim_id": 1, "similarity": 0.8, "evidence_available": 1,
            "nli_entailment": 0, "nli_contradiction": 0, "nli_neutral": 1,
            "percentage_difference": None, "percentage_difference_missing": 1,
            "historical_change": None,
        }])
        row = predict_risk_from_features(legacy).iloc[0]
        self.assertEqual(row["risk_score_status"], "HEURISTIC")

    def test_partial_coverage_with_both_pairs_completed_in_other_order(self):
        features = build_features_from_results(
            _two_match_claim(), [_nli(EV_B, "NEUTRAL"), _nli(EV_A, "ENTAILMENT")]
        )
        self.assertEqual(int(features.iloc[0]["nli_coverage_complete"]), 1)


class LegacySavedFailedRowsTests(unittest.TestCase):
    """Analyses saved before the D1 fix store failed rows with scoring_eligible=True."""

    def test_legacy_failed_row_with_eligible_flag_is_not_counted(self):
        legacy_failed = {
            "claim": CLAIM, "evidence": EVIDENCE, "source_url": SOURCE_URL,
            "nli_result": "UNAVAILABLE", "scoring_eligible": True,
        }
        features = build_features_from_results(_matching(), [legacy_failed])
        row = features.iloc[0]
        self.assertEqual(int(row["evidence_available"]), 1)
        self.assertTrue(pd.isna(row["nli_entailment"]))
        self.assertEqual(
            predict_risk_from_features(features).iloc[0]["risk_score_status"],
            "NOT_PREDICTED",
        )
