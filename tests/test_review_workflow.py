import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation import prepare_review_packet, import_review_packet
from risk_model import FEATURE_COLUMNS


def _analysis():
    claim = "We achieved 80% renewable electricity in 2024."
    evidence = "The company generated 80% of its electricity from renewable sources in 2024."
    analysis_id = "analysis-test"
    return {
        "analysis_id": analysis_id,
        "company": "Example Corp",
        "company_claims": [{
            "statement": claim,
            "source_role": "company_source",
            "source_title": "2024 Sustainability Report",
            "source_url": "https://company.example/report",
        }],
        "matching_results": [{
            "claim": claim,
            "matches": [
                {
                    "matched_evidence": evidence,
                    "source_title": "Energy Authority Record",
                    "source_url": "https://authority.example/record",
                    "publisher": "Energy Authority",
                    "source_role": "external_regulatory_evidence",
                    "similarity": 0.9,
                },
                {
                    "matched_evidence": "Unverified source statement.",
                    "source_title": "Unknown",
                    "source_url": "https://unknown.example/page",
                    "publisher": "Unknown",
                    "source_role": "other_unverified",
                },
            ],
        }],
        "nli_results": [{
            "claim": claim,
            "evidence": evidence,
            "source_url": "https://authority.example/record",
            "source_role": "external_regulatory_evidence",
            "raw_nli_label": "entailment",
            "confidence": 0.8,
        }],
        "features": [{
            "claim_id": 1,
            "claim": claim,
            "evidence_available": 1,
            "similarity": 0.9,
            "nli_entailment": 1,
            "nli_contradiction": 0,
            "nli_neutral": 0,
            "percentage_difference": None,
            "percentage_difference_missing": 1,
            "historical_change": None,
            "predicted_risk_class": 1,
            "risk_score": 80,
        }],
    }


class ReviewWorkflowTests(unittest.TestCase):
    def _paths(self, directory):
        analysis_path = Path(directory) / "analysis.json"
        analysis_path.write_text(
            json.dumps(_analysis()),
            encoding="utf-8",
        )
        return analysis_path

    def test_prepares_blind_nli_packet_without_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            analysis = self._paths(directory)
            output = Path(directory) / "nli_review.csv"
            report = prepare_review_packet.prepare_review_packet(
                analysis,
                output,
                "nli",
            )
            packet = pd.read_csv(output).fillna("")
            self.assertEqual(len(packet), 1)
            self.assertIn("pending_review", packet["eligibility_decision"].tolist())
            self.assertNotIn("model_prediction", packet.columns)
            self.assertNotIn("confidence", packet.columns)
            self.assertEqual(report["contains_model_predictions_or_features"], False)

    def test_prepares_blind_risk_packet_without_features(self):
        with tempfile.TemporaryDirectory() as directory:
            analysis = self._paths(directory)
            output = Path(directory) / "risk_review.csv"
            prepare_review_packet.prepare_review_packet(
                analysis,
                output,
                "risk",
            )
            packet = pd.read_csv(output).fillna("")
            self.assertEqual(len(packet), 1)
            self.assertEqual(
                packet.loc[0, "evidence_availability"],
                "candidate_pending_review",
            )
            self.assertFalse(set(FEATURE_COLUMNS) & set(packet.columns))
            self.assertNotIn("risk_score", packet.columns)

    def test_refuses_to_overwrite_review_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            analysis = self._paths(directory)
            output = Path(directory) / "nli_review.csv"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_review_packet.prepare_review_packet(
                    analysis,
                    output,
                    "nli",
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

    def test_imports_adjudicated_nli_labels_with_saved_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            analysis_path = self._paths(directory)
            packet_path = Path(directory) / "reviewed.csv"
            output_path = Path(directory) / "nli_labels.csv"
            prepare_review_packet.prepare_review_packet(
                analysis_path,
                packet_path,
                "nli",
            )
            packet = pd.read_csv(packet_path).fillna("")
            packet.loc[0, "evidence_availability"] = "available_eligible"
            packet.loc[0, "eligibility_decision"] = "eligible"
            for field in (
                "metric_match",
                "unit_match",
                "period_match",
                "scope_match",
                "organizational_boundary_match",
                "methodology_match",
                "performance_status_match",
            ):
                packet.loc[0, field] = "compatible"
            packet.loc[0, "baseline_match"] = "not_applicable"
            for field, value in (
                ("claim_baseline_year", "not applicable"),
                ("evidence_baseline_year", "not applicable"),
                ("claim_organizational_boundary", "global operations"),
                ("evidence_organizational_boundary", "global operations"),
                ("claim_methodology", "GHG Protocol; location-based"),
                ("evidence_methodology", "GHG Protocol; location-based"),
                ("claim_performance_status", "reported"),
                ("evidence_performance_status", "reported"),
            ):
                packet.loc[0, field] = value
            packet.loc[0, "reviewer_1_id"] = "reviewer-1"
            packet.loc[0, "reviewer_1_label"] = "entailment"
            packet.loc[0, "reviewer_1_notes"] = "Independent review."
            packet.loc[0, "reviewer_2_id"] = "reviewer-2"
            packet.loc[0, "reviewer_2_label"] = "entailment"
            packet.loc[0, "reviewer_2_notes"] = "Independent review."
            packet.loc[0, "adjudicated_label"] = "entailment"
            packet.loc[0, "adjudicator_id"] = "adjudicator-1"
            packet.loc[0, "review_status"] = "adjudicated"
            packet.loc[0, "review_notes"] = "Claim-evidence relation reviewed."
            packet.to_csv(packet_path, index=False)

            result = import_review_packet.import_nli_reviews(
                packet_path,
                analysis_path,
                output_path,
            )
            imported = pd.read_csv(output_path)
            self.assertEqual(result["model_predictions_available"], 1)
            self.assertEqual(imported.loc[0, "model_prediction"], "entailment")
            self.assertEqual(imported.loc[0, "adjudicated_label"], "entailment")

    def test_imports_risk_labels_with_actual_saved_features(self):
        with tempfile.TemporaryDirectory() as directory:
            analysis_path = self._paths(directory)
            packet_path = Path(directory) / "reviewed.csv"
            output_path = Path(directory) / "risk_labels.csv"
            prepare_review_packet.prepare_review_packet(
                analysis_path,
                packet_path,
                "risk",
            )
            packet = pd.read_csv(packet_path).fillna("")
            packet.loc[0, "evidence_availability"] = "available_eligible"
            packet.loc[0, "compatibility_decision"] = "compatible"
            for field in (
                "metric_match",
                "unit_match",
                "period_match",
                "scope_match",
                "organizational_boundary_match",
                "methodology_match",
                "performance_status_match",
            ):
                packet.loc[0, field] = "compatible"
            packet.loc[0, "baseline_match"] = "not_applicable"
            for field, value in (
                ("claim_baseline_year", "not applicable"),
                ("evidence_baseline_year", "not applicable"),
                ("claim_organizational_boundary", "global operations"),
                ("evidence_organizational_boundary", "global operations"),
                ("claim_methodology", "GHG Protocol; location-based"),
                ("evidence_methodology", "GHG Protocol; location-based"),
                ("claim_performance_status", "reported"),
                ("evidence_performance_status", "reported"),
            ):
                packet.loc[0, field] = value
            packet.loc[0, "support_label"] = "entailment"
            packet.loc[0, "reviewer_1_id"] = "reviewer-1"
            packet.loc[0, "reviewer_1_risk_label"] = "no_review_concern"
            packet.loc[0, "reviewer_1_support_label"] = "entailment"
            packet.loc[0, "reviewer_1_notes"] = "Independent review."
            packet.loc[0, "reviewer_2_id"] = "reviewer-2"
            packet.loc[0, "reviewer_2_risk_label"] = "no_review_concern"
            packet.loc[0, "reviewer_2_support_label"] = "entailment"
            packet.loc[0, "reviewer_2_notes"] = "Independent review."
            packet.loc[0, "label"] = "no_review_concern"
            packet.loc[0, "adjudicator_id"] = "adjudicator-1"
            packet.loc[0, "review_status"] = "adjudicated"
            packet.loc[0, "review_notes"] = "Claim and evidence reviewed."
            packet.to_csv(packet_path, index=False)

            result = import_review_packet.import_risk_reviews(
                packet_path,
                analysis_path,
                output_path,
            )
            imported = pd.read_csv(output_path)
            self.assertTrue(result["validated"])
            self.assertEqual(imported.loc[0, "similarity"], 0.9)
            self.assertEqual(imported.loc[0, "evidence_available"], 1)
            self.assertEqual(imported.loc[0, "label"], "no_review_concern")

    def test_import_rejects_model_outputs_in_blind_nli_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            analysis_path = self._paths(directory)
            packet_path = Path(directory) / "reviewed.csv"
            output_path = Path(directory) / "nli_labels.csv"
            prepare_review_packet.prepare_review_packet(
                analysis_path,
                packet_path,
                "nli",
            )
            packet = pd.read_csv(packet_path)
            packet["model_prediction"] = "entailment"
            packet.to_csv(packet_path, index=False)
            with self.assertRaisesRegex(ValueError, "must not contain model outputs"):
                import_review_packet.import_nli_reviews(
                    packet_path,
                    analysis_path,
                    output_path,
                )


if __name__ == "__main__":
    unittest.main()
