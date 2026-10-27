"""Mechanics tests for the two-reviewer pair labelling workflow.

All analyses here are SYNTHETIC fixtures written to temporary directories. They
test validation and arithmetic only. None of their labels or metrics is a
research result.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation import pair_review as pr

URL_C = "https://company.example/report.html"
URL_E1 = "https://independent.example/a.pdf"
URL_E2 = "https://independent.example/b.pdf"


def synthetic_analysis(analysis_id="syn0001", company="Synthetic Co", pairs=None):
    """Synthetic saved-analysis record. Not real company data."""
    claims = [
        {"statement": "We cut Scope 1 emissions 10% in 2024.", "source_role": "company_source",
         "source_url": URL_C, "source_title": "Synthetic report", "publication_date": "2025-03-01"},
        {"statement": "We source 50% renewable power.", "source_role": "company_source",
         "source_url": URL_C, "source_title": "Synthetic report", "publication_date": "2025-03-01"},
    ]
    matches_1 = [
        {"matched_evidence": "Scope 1 emissions fell 10% in 2024 per the verifier.",
         "source_url": URL_E1, "source_role": "external_source", "source_title": "Verifier A",
         "publisher": "Verifier A", "publication_date": "2025-06-01", "similarity": 0.8,
         "match_label": "potentially_relevant"},
        {"matched_evidence": "Scope 1 emissions rose 5% in 2024.",
         "source_url": URL_E2, "source_role": "external_source", "source_title": "Verifier B",
         "publisher": "Verifier B", "publication_date": "2025-06-02", "similarity": 0.6,
         "match_label": "potentially_relevant"},
    ]
    matching = [
        {"claim": claims[0]["statement"], "matches": matches_1},
        {"claim": claims[1]["statement"], "matches": [
            {"matched_evidence": "Renewables were 20% of supply.", "source_url": URL_E1,
             "source_role": "external_source", "source_title": "Verifier A",
             "publisher": "Verifier A", "similarity": 0.7, "match_label": "potentially_relevant"},
        ]},
    ]
    nli = [
        {"claim": claims[0]["statement"], "evidence": "Scope 1 emissions fell 10% in 2024 per the verifier.",
         "source_url": URL_E1, "nli_result": "ENTAILMENT", "scoring_eligible": True},
        {"claim": claims[0]["statement"], "evidence": "Scope 1 emissions rose 5% in 2024.",
         "source_url": URL_E2, "nli_result": "CONTRADICTION", "scoring_eligible": True},
        {"claim": claims[1]["statement"], "evidence": "Renewables were 20% of supply.",
         "source_url": URL_E1, "nli_result": "UNAVAILABLE", "scoring_eligible": False},
    ]
    return {"analysis_id": analysis_id, "company": company, "company_claims": claims,
            "matching_results": matching, "nli_results": nli}


def write_json(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    return path


def label_returned(src: Path, dst: Path, reviewer, labels: dict, rationale="synthetic rationale"):
    rows = read_rows(src)
    for row in rows:
        if row["pair_id"] in labels:
            row["label"] = labels[row["pair_id"]]
            row["reviewer_id"] = reviewer
            row["rationale"] = rationale
    return write_rows(dst, pr.PACKET_COLUMNS, rows)


class PairBuildTests(unittest.TestCase):
    def test_accepts_eligible_pairs_and_keeps_provenance(self):
        pairs, rejected = pr.build_pairs(synthetic_analysis())
        self.assertEqual(len(pairs), 3)
        first = pairs[0]
        self.assertEqual(first["evidence"], "Scope 1 emissions fell 10% in 2024 per the verifier.")
        self.assertEqual(first["evidence_source_url"], URL_E1)
        self.assertEqual(first["evidence_publication_date"], "2025-06-01")
        self.assertEqual(first["claim_source_url"], URL_C)
        self.assertEqual(rejected["accepted"], 3)

    def test_model_status_is_taken_from_saved_nli(self):
        pairs, _ = pr.build_pairs(synthetic_analysis())
        statuses = {p["evidence"]: (p["model_nli_status"], p["model_nli_result"]) for p in pairs}
        self.assertEqual(statuses["Scope 1 emissions fell 10% in 2024 per the verifier."],
                         ("completed", "ENTAILMENT"))
        self.assertEqual(statuses["Renewables were 20% of supply."], ("unavailable", "UNAVAILABLE"))

    def test_rejects_bad_url_role_duplicate_and_ambiguous_claim(self):
        analysis = synthetic_analysis()
        analysis["matching_results"][0]["matches"].append(
            {"matched_evidence": "Bad url evidence.", "source_url": "ftp://x", "source_role": "external_source"})
        analysis["matching_results"][0]["matches"].append(
            {"matched_evidence": "Wrong role evidence.", "source_url": URL_E1, "source_role": "company_source"})
        analysis["matching_results"][0]["matches"].append(
            dict(analysis["matching_results"][0]["matches"][0]))  # exact duplicate
        pairs, rejected = pr.build_pairs(analysis)
        self.assertEqual(rejected.get("evidence_url_missing_or_invalid"), 1)
        self.assertEqual(rejected.get("evidence_role_not_eligible"), 1)
        self.assertEqual(rejected.get("duplicate_pair"), 1)
        self.assertEqual(len(pairs), 3)

    def test_ambiguous_claim_source_rejects_all_pairs_for_that_claim(self):
        analysis = synthetic_analysis()
        analysis["company_claims"].append(dict(analysis["company_claims"][0]))
        pairs, rejected = pr.build_pairs(analysis)
        self.assertEqual(rejected.get("claim_source_missing_or_ambiguous"), 1)
        self.assertEqual(len(pairs), 1)  # only the second claim's pair survives


class PacketBlindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.analysis = write_json(self.dir / "analysis.json", synthetic_analysis())
        self.out = self.dir / "review"

    def tearDown(self):
        self.tmp.cleanup()

    def test_packets_contain_no_model_output_and_are_identical_in_pairs(self):
        pr.make_packets(self.analysis, self.out, "reviewer_one", "reviewer_two")
        a = read_rows(self.out / "packets" / "packet_a_reviewer_one.csv")
        b = read_rows(self.out / "packets" / "packet_b_reviewer_two.csv")
        self.assertEqual(set(a[0].keys()), set(pr.PACKET_COLUMNS))
        for forbidden in ("model_nli_status", "model_nli_result", "nli_result", "similarity"):
            self.assertNotIn(forbidden, a[0])
        self.assertEqual({r["pair_id"] for r in a}, {r["pair_id"] for r in b})
        self.assertTrue(all(r["label"] == "" for r in a))
        self.assertTrue((self.out / "coordinator_only" / "coordinator_manifest.csv").exists())

    def test_refuses_to_overwrite_and_requires_two_different_reviewers(self):
        pr.make_packets(self.analysis, self.out, "reviewer_one", "reviewer_two")
        with self.assertRaises(pr.ReviewError):
            pr.make_packets(self.analysis, self.out, "reviewer_one", "reviewer_two")
        with self.assertRaises(pr.ReviewError):
            pr.make_packets(self.analysis, self.dir / "other", "same", "SAME")

    def test_analysis_without_id_is_rejected(self):
        broken = synthetic_analysis()
        del broken["analysis_id"]
        path = write_json(self.dir / "broken.json", broken)
        with self.assertRaises(pr.ReviewError):
            pr.make_packets(path, self.dir / "x", "a", "b")


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        analysis = write_json(self.dir / "analysis.json", synthetic_analysis())
        self.out = self.dir / "review"
        pr.make_packets(analysis, self.out, "reviewer_one", "reviewer_two")
        self.packet = self.out / "packets" / "packet_a_reviewer_one.csv"
        self.ids = [r["pair_id"] for r in read_rows(self.packet)]

    def tearDown(self):
        self.tmp.cleanup()

    def _returned(self, labels, rationale="x", name="ret.csv"):
        return label_returned(self.packet, self.dir / name, "reviewer_one", labels, rationale)

    def test_accepts_complete_and_partial_returns(self):
        counts = pr.validate_returned(self.packet, self._returned({self.ids[0]: "SUPPORTING"}))
        self.assertEqual(counts["complete"], 1)
        self.assertEqual(counts["unlabelled"], 2)

    def test_rejects_invalid_label(self):
        with self.assertRaises(pr.ReviewError):
            pr.validate_returned(self.packet, self._returned({self.ids[0]: "GREENWASHING"}))

    def test_rejects_contradicting_or_uncertain_without_rationale(self):
        with self.assertRaises(pr.ReviewError):
            pr.validate_returned(self.packet, self._returned({self.ids[0]: "CONTRADICTING"}, rationale=""))
        with self.assertRaises(pr.ReviewError):
            pr.validate_returned(self.packet, self._returned({self.ids[0]: "UNCERTAIN"}, rationale=""))

    def test_rejects_altered_evidence_text(self):
        rows = read_rows(self.packet)
        rows[0]["evidence"] = "A different, more favourable sentence."
        rows[0]["label"] = "SUPPORTING"
        rows[0]["reviewer_id"] = "reviewer_one"
        path = write_rows(self.dir / "altered.csv", pr.PACKET_COLUMNS, rows)
        with self.assertRaisesRegex(pr.ReviewError, "Source text was changed"):
            pr.validate_returned(self.packet, path)

    def test_rejects_altered_url(self):
        rows = read_rows(self.packet)
        rows[1]["evidence_source_url"] = "https://elsewhere.example/x"
        path = write_rows(self.dir / "url.csv", pr.PACKET_COLUMNS, rows)
        with self.assertRaisesRegex(pr.ReviewError, "Source text was changed"):
            pr.validate_returned(self.packet, path)

    def test_rejects_added_missing_and_duplicate_rows(self):
        rows = read_rows(self.packet)
        extra = dict(rows[0]); extra["pair_id"] = "f" * 64
        with self.assertRaisesRegex(pr.ReviewError, "not in this packet"):
            pr.validate_returned(self.packet, write_rows(self.dir / "extra.csv", pr.PACKET_COLUMNS, rows + [extra]))
        with self.assertRaisesRegex(pr.ReviewError, "missing"):
            pr.validate_returned(self.packet, write_rows(self.dir / "miss.csv", pr.PACKET_COLUMNS, rows[:2]))
        with self.assertRaisesRegex(pr.ReviewError, "Duplicate"):
            pr.validate_returned(self.packet, write_rows(self.dir / "dup.csv", pr.PACKET_COLUMNS, rows + [rows[0]]))

    def test_rejects_changed_columns_and_missing_reviewer_id(self):
        rows = read_rows(self.packet)
        rows[0]["label"] = "UNRELATED"
        bad = write_rows(self.dir / "noid.csv", pr.PACKET_COLUMNS, rows)
        with self.assertRaisesRegex(pr.ReviewError, "no reviewer_id"):
            pr.validate_returned(self.packet, bad)
        cols = [c for c in pr.PACKET_COLUMNS if c != "rationale"]
        for r in rows:
            r.pop("rationale", None)
        renamed = write_rows(self.dir / "cols.csv", cols, rows)
        with self.assertRaisesRegex(pr.ReviewError, "columns were changed"):
            pr.validate_returned(self.packet, renamed)


class ReconcileFinalizeEvaluateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        analysis = write_json(self.dir / "analysis.json", synthetic_analysis())
        self.out = self.dir / "review"
        pr.make_packets(analysis, self.out, "reviewer_one", "reviewer_two")
        self.pa = self.out / "packets" / "packet_a_reviewer_one.csv"
        self.pb = self.out / "packets" / "packet_b_reviewer_two.csv"
        self.manifest = self.out / "coordinator_only" / "coordinator_manifest.csv"
        self.ids = [r["pair_id"] for r in read_rows(self.pa)]

    def tearDown(self):
        self.tmp.cleanup()

    def _reconcile(self, labels_a, labels_b, rationale_a="ra", rationale_b="rb"):
        ra = label_returned(self.pa, self.dir / "ra.csv", "reviewer_one", labels_a, rationale_a)
        rb = label_returned(self.pb, self.dir / "rb.csv", "reviewer_two", labels_b, rationale_b)
        out = self.dir / "recon"
        return pr.reconcile(self.pa, ra, self.pb, rb, self.manifest, out), out

    def test_agreement_queue_and_consensus(self):
        a = {self.ids[0]: "SUPPORTING", self.ids[1]: "CONTRADICTING", self.ids[2]: "UNRELATED"}
        b = {self.ids[0]: "SUPPORTING", self.ids[1]: "UNRELATED", self.ids[2]: "UNRELATED"}
        report, out = self._reconcile(a, b)
        self.assertEqual(report["pairs_labelled_by_both"], 3)
        self.assertEqual(report["agreed_pairs"], 2)
        self.assertEqual(report["disagreements_for_adjudication"], 1)
        self.assertAlmostEqual(report["percent_agreement"], 2 / 3, places=4)
        queue = read_rows(out / "adjudication_queue.csv")
        self.assertEqual(queue[0]["label_a"], "CONTRADICTING")
        self.assertEqual(queue[0]["label_b"], "UNRELATED")
        self.assertEqual(queue[0]["adjudicated_label"], "")
        self.assertEqual(len(read_rows(out / "consensus_labels.csv")), 2)

    def test_unfinished_pairs_stay_out_of_consensus(self):
        report, out = self._reconcile({self.ids[0]: "SUPPORTING"}, {})
        self.assertEqual(report["pairs_labelled_by_both"], 0)
        self.assertEqual(report["unfinished"]["pairs_labelled_by_a_only"], 1)
        self.assertEqual(read_rows(out / "consensus_labels.csv"), [])

    def test_same_reviewer_identity_is_rejected(self):
        ra = label_returned(self.pa, self.dir / "ra.csv", "same_person", {self.ids[0]: "UNRELATED"})
        rb = label_returned(self.pb, self.dir / "rb.csv", "same_person", {self.ids[0]: "UNRELATED"})
        with self.assertRaisesRegex(pr.ReviewError, "independent"):
            pr.reconcile(self.pa, ra, self.pb, rb, self.manifest, self.dir / "recon")

    def test_finalize_refuses_open_disagreement_and_same_adjudicator(self):
        report, out = self._reconcile(
            {i: "SUPPORTING" for i in self.ids}, {i: "UNRELATED" for i in self.ids})
        queue = out / "adjudication_queue.csv"
        with self.assertRaisesRegex(pr.ReviewError, "no adjudicated_label"):
            pr.finalize(queue, out / "consensus_labels.csv", self.manifest, self.dir / "final.csv")
        rows = read_rows(queue)
        for r in rows:
            r["adjudicated_label"] = "UNCERTAIN"
            r["adjudicator_id"] = "reviewer_one"  # one of the original reviewers
            r["adjudication_rationale"] = "x"
        bad = write_rows(self.dir / "adj_bad.csv", list(rows[0].keys()), rows)
        with self.assertRaisesRegex(pr.ReviewError, "third person"):
            pr.finalize(bad, out / "consensus_labels.csv", self.manifest, self.dir / "final2.csv")

    def test_finalize_accepts_valid_adjudication(self):
        report, out = self._reconcile(
            {i: "SUPPORTING" for i in self.ids}, {i: "UNRELATED" for i in self.ids})
        rows = read_rows(out / "adjudication_queue.csv")
        for r in rows:
            r["adjudicated_label"] = "UNCERTAIN"
            r["adjudicator_id"] = "adjudicator_three"
            r["adjudication_rationale"] = "insufficient context in test fixture"
        good = write_rows(self.dir / "adj_ok.csv", list(rows[0].keys()), rows)
        summary = pr.finalize(good, out / "consensus_labels.csv", self.manifest, self.dir / "final.csv")
        self.assertEqual(summary["final_labels"], 3)
        final = read_rows(self.dir / "final.csv")
        self.assertTrue(all(r["label_source"] == "adjudicated" for r in final))

    def test_evaluate_marks_small_samples_insufficient_and_scores_only_completed_nli(self):
        final = self.dir / "final_small.csv"
        write_rows(final, ("pair_id", "final_label", "label_source", "reviewer_a_id", "reviewer_b_id"), [
            {"pair_id": self.ids[0], "final_label": "SUPPORTING", "label_source": "agreed",
             "reviewer_a_id": "a", "reviewer_b_id": "b"},
            {"pair_id": self.ids[1], "final_label": "CONTRADICTING", "label_source": "agreed",
             "reviewer_a_id": "a", "reviewer_b_id": "b"},
            {"pair_id": self.ids[2], "final_label": "UNRELATED", "label_source": "agreed",
             "reviewer_a_id": "a", "reviewer_b_id": "b"},
        ])
        result = pr.evaluate(final, self.manifest)
        self.assertEqual(result["reporting_status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["sample_size_final_labels"], 3)
        self.assertEqual(result["nli_coverage"]["unavailable"], 1)
        self.assertEqual(result["pairs_scored_against_nli"], 2)
        self.assertIn("company-level greenwashing accuracy (no company-level ground truth)",
                      result["not_computed"])

    def test_evaluate_arithmetic_on_hand_checked_case(self):
        # Hand-checked mechanics: manifest has (ENTAILMENT, CONTRADICTION, unavailable).
        # Labels: pair0 SUPPORTING (NLI ENTAILMENT -> TP), pair1 SUPPORTING (NLI CONTRADICTION -> FN for SUPPORTING).
        final = self.dir / "final_arith.csv"
        write_rows(final, ("pair_id", "final_label", "label_source", "reviewer_a_id", "reviewer_b_id"), [
            {"pair_id": self.ids[0], "final_label": "SUPPORTING", "label_source": "agreed",
             "reviewer_a_id": "a", "reviewer_b_id": "b"},
            {"pair_id": self.ids[1], "final_label": "SUPPORTING", "label_source": "agreed",
             "reviewer_a_id": "a", "reviewer_b_id": "b"},
        ])
        result = pr.evaluate(final, self.manifest)
        s = result["metrics_supporting_vs_nli_entailment"]
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (1, 0, 1))
        self.assertEqual(s["precision"], 1.0)
        self.assertEqual(s["recall"], 0.5)

    def test_kappa_helper(self):
        self.assertAlmostEqual(pr.cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]), 1.0)
        self.assertIsNone(pr.cohen_kappa([], []))


class SafetyRegressionTests(unittest.TestCase):
    """Regression tests for defects found in the Windows readiness audit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.analysis = write_json(self.dir / "a.json", synthetic_analysis())
        self.run = self.dir / "run"
        pr.make_packets(self.analysis, self.run, "reviewer_one", "reviewer_two")
        self.pa = self.run / "packets" / "packet_a_reviewer_one.csv"
        self.pb = self.run / "packets" / "packet_b_reviewer_two.csv"
        self.manifest = self.run / "coordinator_only" / "coordinator_manifest.csv"
        self.ids = [r["pair_id"] for r in read_rows(self.pa)]

    def tearDown(self):
        self.tmp.cleanup()

    def _disagree_and_reconcile(self):
        ra = label_returned(self.pa, self.dir / "ra.csv", "reviewer_one",
                            {i: "SUPPORTING" for i in self.ids}, "r")
        rb = label_returned(self.pb, self.dir / "rb.csv", "reviewer_two",
                            {i: "UNRELATED" for i in self.ids}, "r")
        out = self.run / "recon"
        pr.reconcile(self.pa, ra, self.pb, rb, self.manifest, out)
        return out

    def _adjudicated_copy(self, out, name, mutate=None):
        rows = read_rows(out / "adjudication_queue.csv")
        for r in rows:
            r["adjudicated_label"] = "UNCERTAIN"
            r["adjudicator_id"] = "adjudicator_three"
            r["adjudication_rationale"] = "synthetic"
        if mutate:
            rows = mutate(rows)
        return write_rows(self.dir / name, list(read_rows(out / "adjudication_queue.csv")[0].keys()), rows)

    def test_manifest_is_not_in_reviewer_packet_folder(self):
        packets = sorted(p.name for p in (self.run / "packets").iterdir())
        self.assertNotIn("coordinator_manifest.csv", packets)
        self.assertTrue(self.manifest.exists())

    def test_deleted_disagreement_row_is_refused(self):
        out = self._disagree_and_reconcile()
        bad = self._adjudicated_copy(out, "del.csv", mutate=lambda rows: rows[1:])
        with self.assertRaisesRegex(pr.ReviewError, "deleted"):
            pr.finalize(bad, out / "consensus_labels.csv", self.manifest, self.dir / "f1.csv")

    def test_added_or_duplicated_adjudication_row_is_refused(self):
        out = self._disagree_and_reconcile()
        dup = self._adjudicated_copy(out, "dup.csv", mutate=lambda rows: rows + [rows[0]])
        with self.assertRaisesRegex(pr.ReviewError, "duplicate"):
            pr.finalize(dup, out / "consensus_labels.csv", self.manifest, self.dir / "f2.csv")

    def test_finalize_reports_pairs_excluded_for_missing_second_label(self):
        ra = label_returned(self.pa, self.dir / "ra2.csv", "reviewer_one",
                            {i: "SUPPORTING" for i in self.ids})
        rb = label_returned(self.pb, self.dir / "rb2.csv", "reviewer_two",
                            {self.ids[0]: "SUPPORTING"})
        out = self.run / "recon2"
        pr.reconcile(self.pa, ra, self.pb, rb, self.manifest, out)
        # No disagreements here, so the queue is empty and finalize can run with it.
        summary = pr.finalize(out / "adjudication_queue.csv", out / "consensus_labels.csv",
                              self.manifest, self.dir / "f3.csv")
        self.assertEqual(summary["pairs_not_labelled_by_both_reviewers_excluded"], 2)
        self.assertEqual(summary["final_labels"], 1)

    def test_output_is_ascii_for_non_ascii_company_name(self):
        import contextlib
        import io
        analysis = write_json(self.dir / "u.json", synthetic_analysis(
            analysis_id="uni0001", company="Soci\u00e9t\u00e9 Synth\u00e9tique"))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = pr.main(["make-packets", "--analysis", str(analysis), "--out-dir",
                            str(self.dir / "uni_run"), "--reviewer-a", "x1", "--reviewer-b", "x2"])
        self.assertEqual(code, 0)
        self.assertTrue(buffer.getvalue().isascii())

    def test_evaluate_aggregates_runs_and_rejects_pairs_repeated_across_runs(self):
        other = write_json(self.dir / "b.json", synthetic_analysis(analysis_id="syn0002", company="Second Co"))
        run2 = self.dir / "run2"
        pr.make_packets(other, run2, "reviewer_one", "reviewer_two")
        fields = ("pair_id", "final_label", "label_source", "reviewer_a_id", "reviewer_b_id")
        for root, manifest_path in ((self.run, self.manifest),
                                    (run2, run2 / "coordinator_only" / "coordinator_manifest.csv")):
            ids = [r["pair_id"] for r in read_rows(manifest_path)]
            write_rows(root / "final_labels.csv", fields, [
                {"pair_id": i, "final_label": "UNRELATED", "label_source": "agreed",
                 "reviewer_a_id": "a", "reviewer_b_id": "b"} for i in ids])
        result = pr.evaluate(out_path=None, run_dirs=[self.run, run2])
        self.assertEqual(result["runs"], 2)
        self.assertEqual(result["sample_size_final_labels"], 6)
        self.assertEqual(result["companies"], 2)
        # The same run twice repeats every pair, which must be rejected.
        with self.assertRaisesRegex(pr.ReviewError, "more than one run"):
            pr.evaluate(out_path=None, run_dirs=[self.run, self.run])


class ProducerConsumerIntegrationTests(unittest.TestCase):
    """Real matcher and NLI-detector code, with a fake embedding model and fake classifier.

    The text is synthetic. The purpose is to check that the saved-analysis shape
    written by the matcher and detector is the shape the packet builder reads.
    """

    def test_app_shaped_analysis_round_trips_into_packets(self):
        import re
        import torch
        from contradiction_detector import detect_online_contradictions
        from online_evidence_matcher import match_online_evidence

        company = "Synthetic Co"
        claim = "Synthetic Co committed to renewable energy across its operations."
        evidence = "Synthetic Co committed to renewable energy across its operations, the verifier notes."

        def vec(text):
            v = torch.zeros(64)
            for tok in re.findall(r"[a-z]+", text.lower()):
                v[sum(map(ord, tok)) % 64] += 1.0
            return v

        class FakeEmbedder:
            def encode(self, texts, convert_to_tensor=True, normalize_embeddings=True):
                return torch.nn.functional.normalize(torch.stack([vec(t) for t in texts]), dim=1)

        statements = [{
            "statement": evidence, "source_role": "external_source",
            "source_url": URL_E1, "source_title": "Verifier A", "publisher": "Verifier A",
            "publication_date": "2025-06-01", "entity_context": company,
            "source_domain": "independent.example", "source_type": "verifier report",
        }]
        results = match_online_evidence([claim], statements, FakeEmbedder())
        self.assertTrue(results and results[0]["matches"], "matcher returned no candidate")

        class FakeClassifier:
            def __call__(self, inputs, **kwargs):
                return [{"label": "ENTAILMENT", "score": 0.9} for _ in inputs]

        from unittest import mock
        with mock.patch.object(contradiction_detector_module(), "_get_nli_classifier",
                               return_value=FakeClassifier()):
            nli = detect_online_contradictions({"company": company, "results": results})

        analysis = {
            "analysis_id": "realshape0001",
            "company": company,
            "company_claims": [{
                "statement": claim, "source_role": "company_source", "source_url": URL_C,
                "source_title": "Company report", "publication_date": "2025-03-01",
            }],
            "matching_results": results,
            "nli_results": nli,
        }
        path = write_json(self.tmpdir_path() / "shape.json", analysis)
        loaded = pr.load_analysis(path)
        pairs, rejected = pr.build_pairs(loaded)
        self.assertEqual(len(pairs), 1, rejected)
        pair = pairs[0]
        self.assertEqual(pair["claim"], claim)
        self.assertEqual(pair["evidence"], evidence)
        self.assertEqual(pair["evidence_source_url"], URL_E1)
        self.assertEqual(pair["model_nli_status"], "completed")
        self.assertEqual(pair["model_nli_result"], "ENTAILMENT")

    def tmpdir_path(self):
        if not hasattr(self, "_tmp"):
            self._tmp = tempfile.TemporaryDirectory()
            self.addCleanup(self._tmp.cleanup)
        return Path(self._tmp.name)


def contradiction_detector_module():
    import contradiction_detector
    return contradiction_detector


class InspectTests(unittest.TestCase):
    def test_inspect_reports_readiness_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = write_json(tmp / "a.json", synthetic_analysis())
            before = sorted(p.name for p in tmp.iterdir())
            summary = pr.inspect_analysis(path)
            self.assertEqual(sorted(p.name for p in tmp.iterdir()), before)
            self.assertTrue(summary["ready_for_packets"])
            self.assertEqual(summary["pairs_accepted_for_review"], 3)
            self.assertEqual(summary["pairs_by_model_nli_status"].get("unavailable"), 1)

    def test_inspect_flags_analysis_without_company_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            analysis = synthetic_analysis()
            analysis["company_claims"] = []
            path = write_json(Path(tmp) / "empty.json", analysis)
            summary = pr.inspect_analysis(path)
            self.assertFalse(summary["ready_for_packets"])


if __name__ == "__main__":
    unittest.main()
