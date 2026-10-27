import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from evaluation import evaluate
from risk_model import FEATURE_COLUMNS


def _rows(groups_per_label=10):
    rows = []
    for label, is_concern in (
        (evaluate.NEGATIVE_LABEL, 0),
        (evaluate.POSITIVE_LABEL, 1),
    ):
        for index in range(groups_per_label):
            split_group = f"{label}-report-{index // 2}"
            rows.append({
                "annotation_id": f"annotation-{label}-{index}",
                "analysis_id": f"analysis-{label}-{index}",
                "claim_id": f"claim-{label}-{index}",
                "company": f"{split_group} Corp",
                "report_id": f"report-{label}-{index // 2}",
                "claim": f"Reviewed environmental claim {label} {index}",
                "claim_group": f"{label}-{index}",
                "split_group": split_group,
                "claim_source_url": (
                    f"https://company.example/report/{index // 2}"
                ),
                "claim_source_role": "company_source",
                "claim_baseline_year": "not applicable",
                "claim_organizational_boundary": "global operations",
                "claim_methodology": "GHG Protocol; location-based",
                "claim_performance_status": "reported",
                "evidence_source_url": f"https://example.test/report/{index}",
                "evidence_source_urls": (
                    f"https://example.test/report/{index}"
                ),
                "evidence_source_role": "independent_third_party_source",
                "evidence_baseline_year": "not applicable",
                "evidence_organizational_boundary": "global operations",
                "evidence_methodology": "GHG Protocol; location-based",
                "evidence_performance_status": "reported",
                "evidence_excerpt": "Reviewed evidence excerpt.",
                "evidence_availability": "available_eligible",
                "compatibility_decision": "compatible",
                "metric_match": "compatible",
                "unit_match": "compatible",
                "period_match": "compatible",
                "scope_match": "compatible",
                "baseline_match": "not_applicable",
                "organizational_boundary_match": "compatible",
                "methodology_match": "compatible",
                "performance_status_match": "compatible",
                "support_label": (
                    "contradiction" if is_concern else "entailment"
                ),
                "reviewer_1_id": "reviewer-1",
                "reviewer_1_risk_label": label,
                "reviewer_1_support_label": (
                    "contradiction" if is_concern else "entailment"
                ),
                "reviewer_1_notes": "Independent review rationale.",
                "reviewer_2_id": "reviewer-2",
                "reviewer_2_risk_label": label,
                "reviewer_2_support_label": (
                    "contradiction" if is_concern else "entailment"
                ),
                "reviewer_2_notes": "Independent review rationale.",
                "adjudicator_id": "adjudicator-1",
                "review_status": "adjudicated",
                "adjudication_notes": "",
                "review_notes": "Synthetic fixture for evaluator software tests.",
                "label": label,
                "similarity": 0.8 if is_concern else 0.6,
                "evidence_available": 1,
                "nli_entailment": 0 if is_concern else 1,
                "nli_contradiction": is_concern,
                "nli_neutral": 0,
                "percentage_difference": 0,
                "percentage_difference_missing": 1,
                "historical_change": 0,
            })
    return pd.DataFrame(rows)


class EvaluationTests(unittest.TestCase):
    def _load(self, frame):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            frame.to_csv(path, index=False)
            return evaluate.load_evaluation_data(path)

    def test_requires_provenance_and_real_feature_columns(self):
        frame = _rows().drop(columns=["evidence_source_url"])
        with self.assertRaisesRegex(ValueError, "evidence_source_url"):
            self._load(frame)

    def test_rejects_quantitative_labels_with_unmatched_dimensions(self):
        frame = _rows()
        frame.loc[0, "methodology_match"] = "not_applicable"
        with self.assertRaisesRegex(ValueError, "every applicable measurement"):
            self._load(frame)

        frame = _rows()
        frame.loc[0, "baseline_match"] = "compatible"
        frame.loc[0, "claim_baseline_year"] = "2015"
        frame.loc[0, "evidence_baseline_year"] = "2020"
        with self.assertRaisesRegex(ValueError, "baseline_match"):
            self._load(frame)

    def test_requires_report_group_metadata(self):
        frame = _rows().drop(columns=["split_group"])
        with self.assertRaisesRegex(ValueError, "split_group"):
            self._load(frame)

    def test_rejects_non_http_evidence_source(self):
        frame = _rows()
        frame.loc[0, "evidence_source_url"] = "not-a-url"
        with self.assertRaisesRegex(ValueError, "absolute HTTP"):
            self._load(frame)

    def test_numeric_database_identifiers_are_not_treated_as_blank(self):
        frame = _rows()
        frame["claim_id"] = range(1, len(frame) + 1)
        frame["report_id"] = range(1, len(frame) + 1)
        loaded = self._load(frame)
        self.assertEqual(loaded.loc[0, "claim_id"], "1")
        self.assertEqual(loaded.loc[0, "report_id"], "1")

    def test_rejects_exact_duplicate_claim_in_different_groups(self):
        frame = _rows()
        frame.loc[1, "claim"] = frame.loc[0, "claim"].upper()
        frame.loc[1, "claim_group"] = "different-group"
        frame.loc[1, "label"] = frame.loc[0, "label"]
        with self.assertRaisesRegex(ValueError, "duplicate claims"):
            self._load(frame)

    def test_rejects_conflicting_labels_within_claim_group(self):
        frame = _rows()
        frame.loc[10, "claim_group"] = frame.loc[0, "claim_group"]
        frame.loc[10, "split_group"] = frame.loc[0, "split_group"]
        with self.assertRaisesRegex(ValueError, "one adjudicated label"):
            self._load(frame)

    def test_rejects_claim_group_spanning_report_groups(self):
        frame = _rows()
        frame.loc[1, "claim_group"] = frame.loc[0, "claim_group"]
        frame.loc[1, "split_group"] = "different-report"
        with self.assertRaisesRegex(ValueError, "one split_group"):
            self._load(frame)

    def test_missing_evidence_is_not_an_evaluable_risk_label(self):
        frame = _rows()
        frame.loc[0, "evidence_available"] = 0
        with self.assertRaisesRegex(
            ValueError,
            "missing evidence is not a negative label",
        ):
            self._load(frame)

    def test_rejects_unadjudicated_labels_and_unverified_sources(self):
        frame = _rows()
        frame.loc[0, "review_status"] = "pending"
        with self.assertRaisesRegex(ValueError, "Only rows with review_status"):
            self._load(frame)

        frame = _rows()
        frame.loc[0, "evidence_source_role"] = "external_source_unverified"
        with self.assertRaisesRegex(ValueError, "eligible external evidence"):
            self._load(frame)

    def test_requires_independent_reviewer_and_adjudicator_ids(self):
        frame = _rows()
        frame.loc[0, "reviewer_2_id"] = frame.loc[0, "reviewer_1_id"]
        with self.assertRaisesRegex(ValueError, "distinct reviewers"):
            self._load(frame)

        frame = _rows()
        frame.loc[0, "adjudicator_id"] = frame.loc[0, "reviewer_1_id"]
        with self.assertRaisesRegex(ValueError, "distinct from both reviewers"):
            self._load(frame)

    def test_rejects_a_company_split_across_evaluation_folds(self):
        frame = _rows()
        frame.loc[1, "company"] = frame.loc[0, "company"]
        frame.loc[1, "split_group"] = "another-report"
        with self.assertRaisesRegex(ValueError, "All reports for a company"):
            self._load(frame)

    def test_group_splits_are_reproducible_and_disjoint(self):
        frame = self._load(_rows())
        first = evaluate._stratified_group_splits(frame)
        second = evaluate._stratified_group_splits(frame)
        for split_name in first:
            self.assertEqual(
                first[split_name]["claim_group"].tolist(),
                second[split_name]["claim_group"].tolist(),
            )
        split_group_sets = {
            name: set(split["split_group"])
            for name, split in first.items()
        }
        self.assertFalse(
            split_group_sets["train"] & split_group_sets["validation"]
        )
        self.assertFalse(split_group_sets["train"] & split_group_sets["test"])
        self.assertFalse(
            split_group_sets["validation"] & split_group_sets["test"]
        )
        for split in first.values():
            self.assertEqual(set(split["label"]), evaluate.ALLOWED_LABELS)
        self.assertEqual(
            sum(len(split) for split in first.values()),
            len(frame),
        )

    def test_labels_do_not_generate_features_or_heuristic_scores(self):
        frame = self._load(_rows())
        changed_labels = frame.copy()
        changed_labels["label"] = changed_labels["label"].map({
            evaluate.POSITIVE_LABEL: evaluate.NEGATIVE_LABEL,
            evaluate.NEGATIVE_LABEL: evaluate.POSITIVE_LABEL,
        })
        self.assertEqual(
            evaluate._risk_scores(frame),
            evaluate._risk_scores(changed_labels),
        )
        self.assertEqual(
            set(FEATURE_COLUMNS),
            set(frame.columns) & set(FEATURE_COLUMNS),
        )

    def test_report_uses_held_out_test_and_majority_baseline(self):
        report = evaluate.evaluate_dataset(self._load(_rows()))
        self.assertEqual(
            report["evaluation_type"],
            "fixed heuristic score with validation-selected cutoff; "
            "no learned model was fit",
        )
        self.assertIn(report["validation_threshold"], evaluate.THRESHOLDS)
        self.assertEqual(
            report["test_metrics"]["confusion_matrix"]["labels"],
            [evaluate.NEGATIVE_LABEL, evaluate.POSITIVE_LABEL],
        )
        self.assertIn(
            "accuracy",
            report["majority_class_baseline"]["test_metrics"],
        )
        self.assertEqual(
            set(report["split_summary"]),
            {"train", "validation", "test"},
        )
        self.assertEqual(
            report["test_confidence_intervals"]["status"],
            "not_reported",
        )
        self.assertIn("not a ground-truth", report["prediction_target"])
        self.assertEqual(
            report["reviewer_agreement"]["risk_label"]["exact_agreement"],
            1.0,
        )

    def test_missing_dataset_exits_without_producing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            with patch("sys.stderr"):
                self.assertEqual(evaluate.main([str(missing)]), 1)


if __name__ == "__main__":
    unittest.main()
