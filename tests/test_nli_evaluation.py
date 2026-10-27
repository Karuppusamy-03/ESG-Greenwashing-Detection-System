import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation import evaluate_nli


def _nli_rows(groups_per_label=10):
    rows = []
    for label in evaluate_nli.LABELS:
        for index in range(groups_per_label):
            company = f"{label}-{index}"
            rows.append({
                "pair_id": f"pair-{label}-{index}",
                "analysis_id": f"analysis-{company}",
                "company": company,
                "company_group": company,
                "report_id": f"report-{company}",
                "split_group": company,
                "claim_group": f"claim-group-{company}",
                "claim": f"Company {company} reported a measured result.",
                "claim_source_title": "Company sustainability report",
                "claim_source_url": f"https://company.example/{company}",
                "claim_source_role": "company_source",
                "claim_reporting_period": "2024",
                "claim_metric": "Renewable electricity",
                "claim_value": "80",
                "claim_unit": "%",
                "claim_scope": "Operations",
                "claim_baseline_year": "not applicable",
                "claim_organizational_boundary": "global operations",
                "claim_methodology": "GHG Protocol; location-based",
                "claim_performance_status": "reported",
                "evidence_source_title": "Independent energy record",
                "evidence_publisher": "Example Authority",
                "evidence_source_url": f"https://authority.example/{company}",
                "evidence_source_role": "external_regulatory_evidence",
                "evidence_excerpt": f"Reviewed excerpt for {company}.",
                "evidence_reporting_period": "2024",
                "evidence_metric": "Renewable electricity",
                "evidence_value": "80",
                "evidence_unit": "%",
                "evidence_scope": "Operations",
                "evidence_baseline_year": "not applicable",
                "evidence_organizational_boundary": "global operations",
                "evidence_methodology": "GHG Protocol; location-based",
                "evidence_performance_status": "reported",
                "evidence_availability": "available_eligible",
                "eligibility_decision": "eligible",
                "metric_match": "compatible",
                "unit_match": "compatible",
                "period_match": "compatible",
                "scope_match": "compatible",
                "baseline_match": "not_applicable",
                "organizational_boundary_match": "compatible",
                "methodology_match": "compatible",
                "performance_status_match": "compatible",
                "reviewer_1_id": "reviewer-1",
                "reviewer_1_label": label,
                "reviewer_1_notes": "Independent review rationale.",
                "reviewer_2_id": "reviewer-2",
                "reviewer_2_label": label,
                "reviewer_2_notes": "Independent review rationale.",
                "adjudicated_label": label,
                "adjudicator_id": "adjudicator-1",
                "review_status": "adjudicated",
                "review_notes": "Evidence and claim reviewed independently.",
                "adjudication_notes": "",
                "uncertainty_reason": "",
                "model_name": "test-model",
                "model_prediction": label,
                "model_confidence": 0.75,
            })
    return pd.DataFrame(rows)


class NliEvaluationTests(unittest.TestCase):
    def _load(self, frame):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nli.csv"
            frame.to_csv(path, index=False)
            return evaluate_nli.load_nli_annotations(path)

    def test_requires_adjudicated_eligible_compatible_pairs(self):
        frame = _nli_rows()
        frame.loc[0, "evidence_availability"] = "missing"
        with self.assertRaisesRegex(ValueError, "missing evidence is not"):
            self._load(frame)

        frame = _nli_rows()
        frame.loc[0, "scope_match"] = "incompatible"
        with self.assertRaisesRegex(ValueError, "scope_match"):
            self._load(frame)

        frame = _nli_rows()
        frame.loc[0, "methodology_match"] = "incompatible"
        with self.assertRaisesRegex(ValueError, "methodology_match"):
            self._load(frame)

        frame = _nli_rows()
        frame.loc[0, "organizational_boundary_match"] = "not_applicable"
        with self.assertRaisesRegex(ValueError, "every applicable measurement"):
            self._load(frame)

        frame = _nli_rows()
        frame.loc[0, "baseline_match"] = "not_applicable"
        frame.loc[0, "claim_baseline_year"] = "2015"
        frame.loc[0, "evidence_baseline_year"] = "2020"
        with self.assertRaisesRegex(ValueError, "baseline_match"):
            self._load(frame)

        frame = _nli_rows()
        frame.loc[0, "review_status"] = "pending"
        with self.assertRaisesRegex(ValueError, "review_status='adjudicated'"):
            self._load(frame)

    def test_rejects_non_independent_sources_and_invalid_urls(self):
        frame = _nli_rows()
        frame.loc[0, "evidence_source_role"] = "company_source"
        with self.assertRaisesRegex(ValueError, "eligible external evidence"):
            self._load(frame)

        frame = _nli_rows()
        frame.loc[0, "evidence_source_url"] = "file:///private/source"
        with self.assertRaisesRegex(ValueError, "absolute HTTP"):
            self._load(frame)

    def test_requires_independent_reviewer_and_adjudicator_ids(self):
        frame = _nli_rows()
        frame.loc[0, "reviewer_2_id"] = frame.loc[0, "reviewer_1_id"]
        with self.assertRaisesRegex(ValueError, "distinct reviewers"):
            self._load(frame)

        frame = _nli_rows()
        frame.loc[0, "adjudicator_id"] = frame.loc[0, "reviewer_1_id"]
        with self.assertRaisesRegex(ValueError, "distinct from both reviewers"):
            self._load(frame)

    def test_rejects_duplicate_pairs_and_company_group_leakage(self):
        frame = _nli_rows()
        frame.loc[1, "claim"] = frame.loc[0, "claim"]
        frame.loc[1, "evidence_excerpt"] = frame.loc[0, "evidence_excerpt"]
        frame.loc[1, "evidence_source_url"] = frame.loc[0, "evidence_source_url"]
        with self.assertRaisesRegex(ValueError, "Duplicate claim-evidence"):
            self._load(frame)

        frame = _nli_rows()
        frame.loc[1, "company_group"] = frame.loc[0, "company_group"]
        with self.assertRaisesRegex(ValueError, "company_group"):
            self._load(frame)

    def test_grouped_splits_and_held_out_metrics_are_reproducible(self):
        frame = self._load(_nli_rows())
        first = evaluate_nli._group_splits(frame)
        second = evaluate_nli._group_splits(frame)
        for name in first:
            self.assertEqual(
                first[name]["split_group"].tolist(),
                second[name]["split_group"].tolist(),
            )
            self.assertEqual(set(first[name]["adjudicated_label"]), set(
                evaluate_nli.LABELS
            ))
        report = evaluate_nli.evaluate_dataset(frame)
        self.assertEqual(report["test_metrics"]["accuracy"], 1.0)
        self.assertEqual(report["reviewer_agreement"]["exact_agreement"], 1.0)
        self.assertEqual(
            report["test_metrics"]["confusion_matrix"]["labels"],
            list(evaluate_nli.LABELS),
        )
        self.assertEqual(
            report["test_confidence_intervals"]["status"],
            "not_reported",
        )

    def test_unavailable_model_outputs_do_not_become_zero_score_metrics(self):
        frame = _nli_rows()
        frame["model_prediction"] = "unavailable"
        report = evaluate_nli.evaluate_dataset(self._load(frame))
        self.assertIsNone(report["test_metrics"])
        self.assertIsNone(report["majority_class_baseline"])
        self.assertIn("No test pair", report["test_metrics_status"])

    def test_missing_dataset_returns_error_without_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            self.assertEqual(evaluate_nli.main([str(missing)]), 1)


if __name__ == "__main__":
    unittest.main()
