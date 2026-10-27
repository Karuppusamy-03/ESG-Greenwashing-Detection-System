import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from contradiction_detector import detect_online_contradictions
from database import (
    initialize_database,
    save_company_predictions,
    save_timeline_records,
)
from explainability import explain_risk_predictions
from feature_builder import build_features_from_results
from online_claim_extractor import (
    extract_embedded_assurance_statements,
    extract_environmental_statements,
    identify_source_role,
)
from online_evidence_matcher import match_online_evidence
from risk_model import FEATURE_COLUMNS, train_risk_model
from risk_predictor import predict_risk_from_features
from timeline_analyzer import (
    analyze_timeline,
    build_timeline_change_rows,
    build_trend_points,
    compare_measurement_compatibility,
    extract_timeline_records,
    get_comparable_metrics,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
# Synthetic, committed saved-analysis fixture (fictional company). The
# user's real saved analyses live in the gitignored output/ folder and are
# covered only by the opt-in local-data checks in local_checks/.
SYNTHETIC_SAVED_ANALYSIS_PATH = (
    PROJECT_DIR / "tests" / "fixtures" / "saved_analysis_synthetic.json"
)


class FakeSimilarityMatrix:
    def __init__(self, scores):
        self.scores = scores

    def __getitem__(self, index):
        return FakeScoreRow(self.scores[index])


class FakeScoreRow:
    def __init__(self, scores):
        self.scores = scores

    def tolist(self):
        return self.scores


class FakeEmbeddingModel:
    def encode(
        self,
        texts,
        convert_to_tensor=False,
        normalize_embeddings=False,
    ):
        if convert_to_tensor and normalize_embeddings:
            return texts
        raise AssertionError("Matcher must request normalized embeddings.")


class FakeNLIClassifier:
    def __init__(self, predictions):
        self.predictions = iter(predictions)

    def __call__(self, evidence, text_pair=None, truncation=False,
                 batch_size=None):
        if isinstance(evidence, list):
            outputs = []
            for item in evidence:
                if not item.get("text") or not item.get("text_pair"):
                    raise AssertionError("NLI input pair was incomplete.")
                outputs.append([next(self.predictions)])
            return [output[0] for output in outputs]
        if not evidence or not text_pair or not truncation:
            raise AssertionError("NLI input pair was incomplete.")
        return [next(self.predictions)]


class OfflineWorkflowTests(unittest.TestCase):
    def test_collected_evidence_analyzer_retains_but_excludes_rejected_sources(self):
        import analyze_collected_evidence

        documents = [
            {
                "status": "success",
                "title": "Acme company sustainability report",
                "source_type": "Official company source",
                "source_domain": "www.acme.example",
                "source_url": "https://www.acme.example/report",
                "final_url": "https://www.acme.example/report",
                "text": (
                    "Acme reports renewable energy use across its "
                    "operations and describes its environmental progress."
                ),
            },
            {
                "status": "success",
                "title": "Rehosted Acme Sustainability Report",
                "source_type": "Other / unverified",
                "source_domain": "archive.org",
                "source_url": "https://archive.org/acme-report",
                "final_url": "https://archive.org/acme-report",
                "text": (
                    "Acme reports renewable energy use across its "
                    "operations and describes its environmental progress."
                ),
            },
            {
                "status": "success",
                "title": "Independent Acme climate assessment",
                "source_type": "Independent third-party source",
                "source_domain": "research.example",
                "source_url": "https://research.example/acme",
                "final_url": "https://research.example/acme",
                "independence_verified": True,
                "text": (
                    "The independent assessment of Acme reports renewable "
                    "energy use and emissions reduction across operations."
                ),
            },
            {
                "status": "failed",
                "title": "Blocked regulatory record",
                "source_type": "Government / regulatory source",
                "source_domain": "epa.gov",
                "source_url": "https://epa.gov/acme",
                "final_url": "https://epa.gov/acme",
                "http_status": 403,
                "error": "HTTP 403 Forbidden",
                "text": "",
            },
        ]

        records = analyze_collected_evidence.extract_environmental_sentences(
            documents
        )
        rejected_by_role = {
            record["source_role"]: record
            for record in records
            if record["candidate_status"] == "rejected"
        }
        eligible = [
            record
            for record in records
            if record["candidate_status"] == "eligible"
        ]

        self.assertEqual(
            {
                record["source_role"]
                for record in eligible
            },
            set(),
        )
        self.assertEqual(
            rejected_by_role["company_source"]["url"],
            "https://www.acme.example/report",
        )
        self.assertIn(
            "not independent",
            rejected_by_role["company_source"]["rejection_reason"],
        )
        self.assertEqual(
            rejected_by_role["rehosted_report_unverified"]["url"],
            "https://archive.org/acme-report",
        )
        self.assertIn(
            "rehosting platform",
            rejected_by_role["rehosted_report_unverified"][
                "rejection_reason"
            ],
        )
        self.assertEqual(
            rejected_by_role["external_source_unverified"]["url"],
            "https://research.example/acme",
        )
        self.assertIn(
            "not classified",
            rejected_by_role["external_source_unverified"][
                "rejection_reason"
            ],
        )
        self.assertEqual(
            records[-1]["candidate_status"],
            "rejected",
        )
        self.assertIn("failed", records[-1]["rejection_reason"])
        self.assertIn("HTTP 403", records[-1]["rejection_reason"])
        self.assertEqual(
            records[-1]["original_url"],
            "https://epa.gov/acme",
        )

        with patch.object(
            analyze_collected_evidence,
            "match_evidence",
            return_value=[],
        ) as matcher:
            result = analyze_collected_evidence.analyze_claims(
                ["Acme reports renewable energy use."],
                records,
            )

        matcher.assert_not_called()
        self.assertEqual(result, [])

    def test_collection_saves_do_not_overwrite_same_second_results(self):
        import automatic_collector
        from datetime import datetime, timezone

        fixed_time = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

        class FixedDateTime:
            @staticmethod
            def now(tz=None):
                return fixed_time

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(
                    automatic_collector,
                    "OUTPUT_DIR",
                    Path(temporary_directory),
                ),
                patch.object(
                    automatic_collector,
                    "datetime",
                    FixedDateTime,
                ),
            ):
                first_path = automatic_collector.save_documents(
                    [{"marker": "first"}],
                    "Acme",
                )
                second_path = automatic_collector.save_documents(
                    [{"marker": "second"}],
                    "Acme",
                )

            self.assertNotEqual(first_path, second_path)
            self.assertEqual(
                json.loads(Path(first_path).read_text(encoding="utf-8"))[
                    "documents"
                ],
                [{"marker": "first"}],
            )
            self.assertEqual(
                json.loads(Path(second_path).read_text(encoding="utf-8"))[
                    "documents"
                ],
                [{"marker": "second"}],
            )

    def test_claim_extraction_separates_company_and_external_sources(self):
        company_document = {
            "status": "success",
            "text": (
                "Acme reports that renewable electricity supplied "
                "65 percent of its operations in 2024."
            ),
            "source_type": "Official company source",
            "source_domain": "www.acme.example",
            "source_url": "https://acme.example/sustainability",
            "final_url": "https://acme.example/sustainability",
            "title": "Acme Sustainability Report",
            "publisher": "Acme",
            "publication_date": "2025-02-01",
        }
        regulator_document = {
            **company_document,
            "source_type": "Government source",
            "source_domain": "epa.gov",
            "source_url": "https://epa.gov/report",
            "final_url": "https://epa.gov/report",
            "title": "Agency Report",
            "publisher": "Environmental Agency",
        }

        company_claims = extract_environmental_statements(company_document)
        external_statements = extract_environmental_statements(
            regulator_document
        )

        self.assertEqual(identify_source_role(company_document), "company_source")
        self.assertEqual(
            identify_source_role(regulator_document),
            "external_regulatory_evidence",
        )
        self.assertEqual(len(company_claims), 1)
        self.assertEqual(len(external_statements), 1)
        self.assertEqual(
            company_claims[0]["source_role"],
            "company_source",
        )
        self.assertEqual(
            external_statements[0]["source_role"],
            "external_regulatory_evidence",
        )
        for statement in (company_claims[0], external_statements[0]):
            self.assertTrue(statement["source_title"])
            self.assertTrue(statement["publisher"])
            self.assertTrue(statement["source_url"])
            self.assertEqual(statement["publication_date"], "2025-02-01")

        inaccessible = {
            **company_document,
            "status": "failed",
            "text": "",
        }
        self.assertEqual(extract_environmental_statements(inaccessible), [])
        self.assertEqual(extract_environmental_statements({}), [])

    def test_claim_extraction_ignores_navigation_and_page_headings(self):
        document = {
            "status": "success",
            "text": (
                "Sustainability strategy - Acme\n"
                "Mode\nLight\nDark\nFuture of Mobility\n"
                "Investor Relations\nFinancial Information\n"
                "Environmental Compliance Reports | Acme\n"
                "We prioritize renewable energy throughout operations.\n"
                "Our environmental strategy protects biodiversity near "
                "manufacturing facilities. "
                "Scope 1 emissions fell by 12 percent to 8,500 tonnes "
                "CO2e in 2024."
            ),
            "source_type": "Official company source",
            "source_domain": "www.acme.example",
            "source_url": "https://acme.example/esg",
            "final_url": "https://acme.example/esg",
            "title": "Sustainability strategy - Acme",
        }

        statements = extract_environmental_statements(document)

        self.assertEqual(
            [item["statement"] for item in statements],
            [
                "We prioritize renewable energy throughout operations.",
                "Our environmental strategy protects biodiversity near "
                "manufacturing facilities.",
                "Scope 1 emissions fell by 12 percent to 8,500 tonnes "
                "CO2e in 2024.",
            ],
        )

    def test_claim_extraction_removes_rehosted_page_chrome(self):
        document = {
            "status": "success",
            "text": (
                "Acme ESG and Sustainability Report 2024 | PDF | "
                "Sustainability Skip to main content Open navigation menu "
                "en Change Language, English Upload Sign in Sign in "
                "0 ratings 0% found this document useful (0 votes) "
                "854 views 28 pages Acme ESG and Sustainability Report 2024 "
                "Acme is committed to reducing operational emissions."
            ),
            "source_type": "Other / unverified",
            "source_domain": "www.scribd.com",
            "source_url": "https://www.scribd.com/document/example",
            "final_url": "https://www.scribd.com/document/example",
            "title": "Acme ESG and Sustainability Report 2024 | PDF | Sustainability",
        }

        statements = extract_environmental_statements(document)

        self.assertEqual(len(statements), 1)
        self.assertEqual(
            statements[0]["statement"],
            "Acme is committed to reducing operational emissions.",
        )
        self.assertEqual(
            statements[0]["source_role"],
            "rehosted_report_unverified",
        )

    def test_claim_extraction_ignores_environmental_question_headings(self):
        document = {
            "status": "success",
            "text": (
                "What is the carbon footprint of a Tesla? "
                "Tesla reports a carbon footprint of 135 g CO2e per mile "
                "for Model 3 and Model Y vehicles."
            ),
            "source_type": "Other / unverified",
            "source_domain": "research.example",
            "source_url": "https://research.example/tesla-footprint",
            "final_url": "https://research.example/tesla-footprint",
            "title": "Tesla lifecycle emissions",
        }

        statements = extract_environmental_statements(document)

        self.assertEqual(len(statements), 1)
        self.assertEqual(
            statements[0]["statement"],
            "Tesla reports a carbon footprint of 135 g CO2e per mile "
            "for Model 3 and Model Y vehicles.",
        )

    def test_claim_extraction_removes_leading_pdf_page_number_sequences(self):
        document = {
            "status": "success",
            "text": (
                "04 05 06 07 15 20 25 30 32 40 54 59 64 74 87 94 "
                "97 101 102 Samsung Electronics is committed to "
                "reducing greenhouse gas emissions."
            ),
            "source_type": "Official company source",
            "source_domain": "www.samsung.com",
            "source_url": "https://www.samsung.com/report",
            "final_url": "https://www.samsung.com/report",
            "title": "Samsung Sustainability Report",
        }

        statements = extract_environmental_statements(document)

        self.assertEqual(len(statements), 1)
        self.assertEqual(
            statements[0]["statement"],
            "Samsung Electronics is committed to reducing greenhouse gas "
            "emissions.",
        )

    def test_claim_extraction_removes_contents_page_numbers_from_pdf(self):
        from online_claim_extractor import extract_environmental_statements

        document = {
            "status": "success",
            "title": "Apple 2025 Environmental Progress Report",
            "source_type": "Official company source",
            "source_domain": "www.apple.com",
            "source_url": "https://www.apple.com/environment/report.pdf",
            "final_url": "https://www.apple.com/environment/report.pdf",
            "text": (
                "Contents\n"
                "Introduction\n"
                "3\n"
                "Reflections from Lisa Jackson\n"
                "4\n"
                "Report highlights\n"
                "5\n"
                "Goals and progress\n"
                "Apple 2030\n"
                "9\n"
                "Approach\n"
                "13\n"
                "Environmental Initiatives\n"
                "and Advocacy\n"
                "worldwide\n"
                "Ernst & Young LLP Use of Proceeds Examination\n"
                "98\n"
                "2025 Environmental Progress Report\n"
                "At Apple, we are constantly innovating to make\n"
                "the world's best technology while reducing\n"
                "our impact on the environment.\n"
                "Scope 1 emissions were 125\n"
                "metric tons CO2e in 2024."
            ),
        }

        statements = extract_environmental_statements(document)

        self.assertEqual(len(statements), 2)
        self.assertEqual(
            statements[0]["statement"],
            "At Apple, we are constantly innovating to make the world's "
            "best technology while reducing our impact on the environment.",
        )
        self.assertEqual(
            statements[1]["statement"],
            "Scope 1 emissions were 125 metric tons CO2e in 2024.",
        )

    def test_unrelated_epa_apple_food_page_is_not_company_evidence(self):
        import app

        document = {
            "status": "success",
            "text": (
                "This guide reviews energy use and greenhouse gas emissions "
                "from canned apple products and preserved vegetables."
            ),
            "source_type": "Government / regulatory source",
            "source_domain": "www.epa.gov",
            "source_url": "https://www.epa.gov/example/canned-fruit",
            "final_url": "https://www.epa.gov/example/canned-fruit",
            "title": "Canned and Preserved Fruits and Vegetables",
            "publisher": "U.S. Environmental Protection Agency",
            "publication_date": "2025-01-01",
        }

        self.assertFalse(app.document_mentions_company("Apple", document))
        company_claims, external, statements = app.collect_candidate_statements(
            "Apple",
            [document],
        )
        self.assertEqual(company_claims, [])
        self.assertEqual(external, [])
        self.assertEqual(statements, [])

    def test_fictional_apple_reference_is_not_company_evidence(self):
        import app

        document = {
            "status": "success",
            "text": (
                "Help Apple and her friends learn to reduce food waste "
                "and protect the environment."
            ),
            "source_type": "Government / regulatory source",
            "source_domain": "www.epa.gov",
            "source_url": "https://www.epa.gov/students/apple-activity",
            "final_url": "https://www.epa.gov/students/apple-activity",
            "title": "Games and Activities about the Environment",
        }

        self.assertFalse(app.document_mentions_company("Apple", document))
        self.assertEqual(
            app.collect_candidate_statements("Apple", [document]),
            ([], [], []),
        )

    def test_external_title_alone_cannot_establish_company_relevance(self):
        import app

        document = {
            "status": "success",
            "text": (
                "The report reviews greenhouse gas emissions and energy "
                "consumption across the national economy."
            ),
            "source_type": "Government / regulatory source",
            "source_domain": "www.epa.gov",
            "source_url": "https://www.epa.gov/example/company-report",
            "final_url": "https://www.epa.gov/example/company-report",
            "title": "Acme Environmental Report",
        }

        self.assertFalse(app.document_mentions_company("Acme", document))
        self.assertEqual(
            app.collect_candidate_statements("Acme", [document]),
            ([], [], []),
        )

    def test_candidate_statements_deduplicate_across_sources_and_keep_links(self):
        import app

        first = {
            "status": "success",
            "text": (
                "Acme reports renewable electricity supplied "
                "65 percent of its operations in 2024."
            ),
            "source_type": "Official company source",
            "source_domain": "acme.example",
            "source_url": "https://acme.example/report",
            "final_url": "https://acme.example/report",
            "title": "Acme Report",
        }
        second = {
            **first,
            "source_domain": "www.acme.example",
            "source_url": "https://www.acme.example/report",
            "final_url": "https://www.acme.example/report",
            "title": "Acme Environmental Report",
        }

        claims, external, statements = app.collect_candidate_statements(
            "Acme",
            [first, second],
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(external, [])
        self.assertEqual(len(statements), 1)
        self.assertEqual(
            statements[0]["additional_sources"][0]["source_url"],
            "https://www.acme.example/report",
        )

    def test_relevant_independent_apple_statement_keeps_source_provenance(self):
        import app

        document = {
            "status": "success",
            "text": (
                "Apple Inc. reported greenhouse gas emissions associated "
                "with its supplier operations in the regulatory filing."
            ),
            "source_type": "Government / regulatory source",
            "source_domain": "www.sec.gov",
            "source_url": "https://www.sec.gov/Archives/example",
            "final_url": "https://www.sec.gov/Archives/example",
            "title": "Apple Inc. Regulatory Filing",
            "publisher": "U.S. Securities and Exchange Commission",
            "publication_date": "2025-02-03",
        }

        self.assertTrue(app.document_mentions_company("Apple", document))
        claims, external, _ = app.collect_candidate_statements(
            "Apple",
            [document],
        )
        self.assertEqual(claims, [])
        self.assertEqual(len(external), 1)
        self.assertEqual(
            external[0]["source_role"],
            "external_regulatory_evidence",
        )
        self.assertEqual(
            external[0]["source_url"],
            "https://www.sec.gov/Archives/example",
        )
        self.assertEqual(
            external[0]["source_title"],
            "Apple Inc. Regulatory Filing",
        )
        self.assertEqual(
            external[0]["publisher"],
            "U.S. Securities and Exchange Commission",
        )
        self.assertEqual(external[0]["publication_date"], "2025-02-03")

    def test_embedded_assurance_statement_keeps_issuer_and_report_link(self):
        import app
        from analyze_collected_evidence import extract_environmental_sentences

        assurance_text = (
            "Acme Inc. reported greenhouse gas emissions decreased by "
            "60 percent in fiscal year 2024.\n"
            "INDEPENDENT ASSURANCE STATEMENT\n"
            "Apex Companies, LLC (Apex) was engaged by Acme Inc. (Acme) "
            "to conduct an independent assurance of selected environmental "
            "information.\n"
            "The assurance covered greenhouse gas emissions and carbon "
            "footprint data for fiscal year 2024.\n"
            "Apex Companies, LLC signed this statement on April 15th, 2026.\n"
            "Our Conclusion\n"
            "Based on our assurance procedures, the Scope 1, Scope 2, and "
            "Scope 3 greenhouse gas emissions assertions shown above are "
            "materially correct.\n"
            "Apex Companies, LLC Page 2 of 2"
        )
        document = {
            "status": "success",
            "text": assurance_text,
            "source_type": "Official company source",
            "source_domain": "www.acme.example",
            "source_url": "https://www.acme.example/report.pdf",
            "final_url": "https://www.acme.example/report.pdf",
            "title": "Acme Environmental Report",
        }

        claims, external, statements = app.collect_candidate_statements(
            "Acme",
            [document],
        )

        self.assertEqual(len(claims), 1)
        self.assertFalse(
            any(
                "materially correct" in item["statement"].casefold()
                for item in claims
            )
        )
        self.assertTrue(external)
        self.assertTrue(
            all(
                item["source_role"] == "independent_third_party_source"
                and item["publisher"] == "Apex Companies, LLC"
                and item["source_url"]
                == "https://www.acme.example/report.pdf"
                and item["publication_date"] == "2026-04-15"
                and item["candidate_status"] == "eligible_external_candidate"
                for item in external
            )
        )
        self.assertTrue(
            all(
                item["source_role"] != "company_source"
                for item in statements
                if item["publisher"] == "Apex Companies, LLC"
            )
        )
        extracted_records = extract_environmental_sentences(
            [document],
            entity_name="Acme",
        )
        self.assertTrue(
            any(
                record["source_role"] == "independent_third_party_source"
                and record["candidate_status"] == "eligible"
                and record["url"] == document["source_url"]
                for record in extracted_records
            )
        )
        self.assertTrue(
            extract_embedded_assurance_statements(
                {
                    **document,
                    "text": (
                        "Acme reports greenhouse gas emissions in its "
                        "annual report without an assurance section."
                    ),
                },
                "Acme",
            )
            == []
        )
        self.assertEqual(
            extract_embedded_assurance_statements(
                {
                    **document,
                    "text": (
                        "INDEPENDENT ASSURANCE STATEMENT\n"
                        "Trinity reviewed Acme greenhouse gas emissions "
                        "data in fiscal year 2024."
                    ),
                },
                "Acme",
            ),
            [],
        )

    def test_deloitte_accountant_review_report_preserves_scope_and_blocks_baseline(self):
        import app

        report_text = (
            "1.11 Independent accountant’s review report\n"
            "Deloitte & Touche LLP\n"
            "To the Board of Directors of Microsoft Corporation\n"
            "We have reviewed management of Microsoft Corporation’s assertion "
            "that the specified environmental information in Section 1 of "
            "the Fact Sheet for the fiscal year ended June 30, 2024 is "
            "presented in accordance with the reporting criteria.\n"
            "Further, information relating to periods prior to the year-ended "
            "June 30, 2024 or forward-looking statements, targets, goals, and "
            "progress against goals, was not subject to our review and we do "
            "not express a conclusion or any form of assurance on such "
            "information.\n"
            "Based on our review, we are not aware of any material "
            "modifications to management’s assertion in order for it to be "
            "fairly stated.\n"
            "May 29, 2025"
        )
        document = {
            "status": "success",
            "title": "2025 Microsoft Environmental Data Fact Sheet",
            "source_type": "Official company source",
            "source_domain": "cdn-dynmedia-1.microsoft.com",
            "source_url": "https://cdn-dynmedia-1.microsoft.com/fact-sheet.pdf",
            "final_url": "https://cdn-dynmedia-1.microsoft.com/fact-sheet.pdf",
            "text": report_text,
        }

        claims, external, statements = app.collect_candidate_statements(
            "Microsoft",
            [document],
        )
        assurance = [
            item for item in external
            if item.get("assurance_report")
        ]

        self.assertGreaterEqual(len(assurance), 1)
        record = next(
            item for item in assurance
            if "We have reviewed management" in item["statement"]
        )
        self.assertEqual(record["publisher"], "Deloitte & Touche LLP")
        self.assertEqual(
            record["source_role"],
            "independent_third_party_source",
        )
        self.assertEqual(
            record["candidate_status"],
            "eligible_external_candidate",
        )
        self.assertEqual(record["source_url"], document["source_url"])
        self.assertEqual(
            record["covered_fiscal_period"],
            "fiscal year ended June 30, 2024",
        )
        self.assertEqual(record["publication_date"], "2025-05-29")
        exclusions = " ".join(record["scope_exclusions"])
        self.assertIn("periods prior", exclusions)
        self.assertIn("targets, goals", exclusions)
        self.assertIn("Deloitte & Touche LLP", record["original_text"])
        self.assertIn("May 29, 2025", record["original_text"])
        self.assertTrue(record["assurance_scope"])
        self.assertFalse(
            any(
                item.get("source_role") == "company_source"
                and "Deloitte & Touche LLP" in item.get("statement", "")
                for item in statements
            )
        )

        claim = (
            "Our value chain or Scope 3 emissions increased by 26% "
            "from our 2020 baseline."
        )
        status, reason = compare_measurement_compatibility(
            claim,
            record["statement"],
            record,
        )
        self.assertEqual(status, "INCONCLUSIVE")
        self.assertIn("2020", reason)
        self.assertIn("2024", reason)

        with patch(
            "sentence_transformers.util.cos_sim",
            return_value=FakeSimilarityMatrix([[0.90]]),
        ):
            matching_results = match_online_evidence(
                [claim],
                [record],
                FakeEmbeddingModel(),
            )
        matched_pair = matching_results[0]
        match = matched_pair["matches"][0]
        self.assertEqual(
            match["covered_fiscal_period"],
            "fiscal year ended June 30, 2024",
        )
        self.assertEqual(match["scope_exclusions"], record["scope_exclusions"])
        self.assertEqual(match["original_text"], record["original_text"])

        with patch(
            "contradiction_detector._get_nli_classifier",
            side_effect=AssertionError("Excluded assurance reached NLI."),
        ):
            nli_results = detect_online_contradictions({
                "company": "Microsoft",
                "results": [matched_pair],
            })

        self.assertEqual(len(nli_results), 1)
        self.assertEqual(nli_results[0]["nli_result"], "UNAVAILABLE")
        self.assertFalse(nli_results[0]["scoring_eligible"])
        features = build_features_from_results(
            [matched_pair],
            nli_results,
        )
        self.assertEqual(features.loc[0, "evidence_available"], 0)

        unrelated_reports = [
            (
                "1.11 Independent accountant’s report\n"
                "Deloitte & Touche LLP reviewed Microsoft’s emissions."
            ),
            (
                "1.11 Independent accountant’s review report\n"
                "Ernst & Young LLP\n"
                "To the Board of Directors of Microsoft Corporation\n"
                "We have reviewed management of Microsoft Corporation’s "
                "environmental assertion for the fiscal year ended June 30, "
                "2024. Deloitte & Touche LLP was mentioned in a separate "
                "advisory context.\n"
                "Based on our review, it is fairly stated."
            ),
            (
                "1.11 Independent accountant’s review report\n"
                "Ernst & Young LLP\n"
                "To the Board of Directors of Microsoft Corporation\n"
                "We have reviewed management of Microsoft Corporation’s "
                "environmental assertion for the fiscal year ended June 30, "
                "2024.\n"
                "Based on our review, it is fairly stated."
            ),
        ]
        for unrelated_text in unrelated_reports:
            with self.subTest(text=unrelated_text[:60]):
                self.assertEqual(
                    extract_embedded_assurance_statements(
                        {
                            **document,
                            "text": unrelated_text,
                        },
                        "Microsoft",
                    ),
                    [],
                )

    def test_assurance_date_and_publisher_stop_at_attestation(self):
        assurance = {
            "status": "success",
            "source_url": "https://www.acme.example/report.pdf",
            "final_url": "https://www.acme.example/report.pdf",
            "text": (
                "INDEPENDENT ASSURANCE STATEMENT\n"
                "Apex Companies, LLC (Apex) was engaged by Acme Inc. "
                "(Acme) to review selected environmental information.\n"
                "Apex verified Acme greenhouse gas emissions of "
                "15.28 million tonnes of CO2e.\n"
                "Attestation:\n"
                "Apex Companies, LLC\n"
                "April 15th, 2026\n"
                "Fraunhofer IZM independently reviewed Acme product "
                "emissions using modeled data.\n"
                "Berlin, March 5, 2026"
            ),
        }

        statements = extract_embedded_assurance_statements(
            assurance,
            "Acme",
        )

        self.assertTrue(statements)
        self.assertTrue(
            all(item["publisher"] == "Apex Companies, LLC" for item in statements)
        )
        self.assertTrue(
            all(item["publication_date"] == "2026-04-15" for item in statements)
        )
        self.assertFalse(
            any("Fraunhofer" in item["statement"] for item in statements)
        )

        trinity = {
            "status": "success",
            "source_url": "https://www.acme.example/report.pdf",
            "text": (
                "INDEPENDENT REVIEW STATEMENT\n"
                "Trinity Consultants, Inc. reviewed Acme "
                "greenhouse gas emissions data for fiscal year 2025.\n"
                "Trinity Consultants, Incorporated\n"
                "Dallas Texas Corporate Headquarters\n"
                "April 1, 2026"
            ),
        }
        trinity_statements = extract_embedded_assurance_statements(
            trinity,
            "Acme",
        )
        self.assertTrue(trinity_statements)
        self.assertTrue(
            all(
                item["publication_date"] == "2026-04-01"
                for item in trinity_statements
            )
        )

    def test_embedded_assurance_table_preserves_measurement_and_caveats(self):
        assurance = {
            "status": "success",
            "source_url": "https://www.acme.example/report.pdf",
            "final_url": "https://www.acme.example/report.pdf",
            "title": "Acme Environmental Report",
            "text": (
                "INDEPENDENT ASSURANCE STATEMENT\n"
                "Apex Companies, LLC (Apex) was engaged by Acme Inc. "
                "(Acme) to conduct independent assurance.\n"
                "Apex reviewed environmental data for the fiscal year "
                "2025 reporting period (January 1 through December 31, "
                "2025).\n"
                "Parameter\nQuantity\nUnits\nBoundary / Protocol\n"
                "Comprehensive Carbon Footprint Note 5\n"
                "15,280,000\nt C O 2 e\n"
                "Worldwide occupied properties / WRI/WBCSD GHG Protocol\n"
                "4. Not verified by Apex. Verified by a non-Apex "
                "third-party provider contracted to Acme.\n"
                "5. Comprehensive Carbon Footprint = Scope 1 + Scope 2 "
                "+ Scope 3.\n"
                "Our Conclusion\nThe assertions were reviewed."
            ),
        }

        statements = extract_embedded_assurance_statements(
            assurance,
            "Acme",
        )

        table_statement = next(
            item for item in statements
            if (
                item["source_type"].startswith(
                    "Independent assurance measurement table"
                )
            )
        )
        self.assertIn(
            "fiscal year 2025 reporting period",
            table_statement["statement"],
        )
        self.assertIn(
            "Comprehensive Carbon Footprint Note 5",
            table_statement["statement"],
        )
        self.assertIn(
            "Worldwide occupied properties",
            table_statement["statement"],
        )
        self.assertIn(
            "not verified by apex",
            table_statement["statement"].casefold(),
        )
        self.assertIn(
            "Scope 1 + Scope 2 + Scope 3",
            table_statement["statement"],
        )
        self.assertEqual(
            table_statement["source_role"],
            "independent_third_party_source",
        )
        self.assertEqual(
            table_statement["source_url"],
            assurance["source_url"],
        )

    def test_standalone_extractor_separates_assurance_from_company_claims(self):
        import online_claim_extractor

        with tempfile.TemporaryDirectory() as temporary_directory:
            collected_dir = Path(temporary_directory) / "collected"
            output_dir = Path(temporary_directory) / "output"
            collected_dir.mkdir()
            output_dir.mkdir()
            collection = {
                "search_name": "Acme",
                "documents": [{
                    "status": "success",
                    "title": "Acme Environmental Report",
                    "source_type": "Official company source",
                    "source_domain": "www.acme.example",
                    "source_url": "https://www.acme.example/report.pdf",
                    "final_url": "https://www.acme.example/report.pdf",
                    "text": (
                        "Acme reported greenhouse gas emissions decreased "
                        "60 percent from a 2015 baseline.\n"
                        "INDEPENDENT ASSURANCE STATEMENT\n"
                        "Apex Companies, LLC (Apex) was engaged by Acme "
                        "Inc. (Acme) to conduct independent assurance.\n"
                        "Apex reviewed environmental data for the fiscal "
                        "year 2025 reporting period.\n"
                        "Parameter\nQuantity\nUnits\nBoundary / Protocol\n"
                        "Comprehensive Carbon Footprint Note 5\n"
                        "15,280,000\nt C O 2 e\n"
                        "Worldwide occupied properties / GHG Protocol\n"
                        "Our Conclusion\n"
                        "The Scope 1 emissions assertions were reviewed."
                    ),
                }],
            }
            input_path = collected_dir / "acme_collection.json"
            input_path.write_text(
                json.dumps(collection),
                encoding="utf-8",
            )

            with (
                patch.object(
                    online_claim_extractor,
                    "COLLECTED_DIR",
                    collected_dir,
                ),
                patch.object(
                    online_claim_extractor,
                    "OUTPUT_DIR",
                    output_dir,
                ),
            ):
                online_claim_extractor.main()

            output_path = output_dir / "acme_collection_statements.json"
            with output_path.open("r", encoding="utf-8") as file:
                result = json.load(file)

        company_statements = [
            item for item in result["statements"]
            if item["source_role"] == "company_source"
        ]
        assurance_statements = [
            item for item in result["statements"]
            if item["source_role"] == "independent_third_party_source"
        ]
        self.assertEqual(len(company_statements), 1)
        self.assertFalse(
            any(
                "assertions were reviewed" in item["statement"].casefold()
                for item in company_statements
            )
        )
        self.assertTrue(
            any(
                "15,280,000" in item["statement"]
                and item["entity_context"] == "Acme"
                for item in assurance_statements
            )
        )

    def test_unverified_third_party_source_is_retained_but_not_a_candidate(self):
        import app

        document = {
            "status": "success",
            "text": (
                "Acme reports renewable electricity use at its facilities "
                "and describes its environmental initiatives."
            ),
            "source_type": "Other / unverified",
            "source_domain": "news.example",
            "source_url": "https://news.example/acme-environment",
            "final_url": "https://news.example/acme-environment",
            "title": "Acme environmental coverage",
            "publisher": "News Example",
        }

        claims, external, statements = app.collect_candidate_statements(
            "Acme",
            [document],
        )

        self.assertEqual(claims, [])
        self.assertEqual(external, [])
        self.assertEqual(len(statements), 1)
        self.assertEqual(
            statements[0]["source_role"],
            "external_source_unverified",
        )
        self.assertEqual(
            statements[0]["statement_kind"],
            "The source is not classified as regulatory, configured, or "
            "explicitly attributed independent evidence.",
        )
        self.assertEqual(
            statements[0]["candidate_status"],
            "rejected_external_candidate",
        )
        self.assertEqual(
            statements[0]["candidate_rejection_reason"],
            statements[0]["statement_kind"],
        )
        diagnostics = app.build_pipeline_diagnostics(
            {},
            [document],
            [document],
            statements,
            external,
            [],
            [],
        )
        rejected_candidate = diagnostics["pipeline"]["evidence"][
            "rejected_candidates"
        ][0]
        self.assertEqual(
            rejected_candidate["source_url"],
            "https://news.example/acme-environment",
        )
        self.assertEqual(
            rejected_candidate["reason"],
            statements[0]["candidate_rejection_reason"],
        )
        self.assertEqual(
            statements[0]["source_url"],
            "https://news.example/acme-environment",
        )

    def test_configured_environmental_publisher_is_verified_by_domain(self):
        import web_search

        url = "https://www.ipcc.ch/report/climate-change/"
        document = {
            "source_type": web_search.classify_source(
                url,
                company_name="Acme",
            ),
            "source_domain": "www.ipcc.ch",
            "source_url": url,
            "final_url": url,
            "publisher": "Intergovernmental Panel on Climate Change",
        }

        self.assertEqual(
            document["source_type"],
            "Configured environmental source",
        )
        self.assertEqual(
            identify_source_role(document),
            "configured_environmental_source",
        )

    def test_named_third_party_publisher_is_not_verified_by_name_alone(self):
        import web_search

        url = (
            "https://theplanetbrief.com/esg/"
            "apple-environmental-progress-report-2026/"
        )
        document = {
            "source_type": web_search.classify_source(
                url,
                company_name="Apple",
            ),
            "source_domain": "theplanetbrief.com",
            "source_url": url,
            "final_url": url,
            "publisher": "The Planet Brief",
        }

        self.assertEqual(document["source_type"], "Other / unverified")
        self.assertEqual(
            identify_source_role(document),
            "external_source_unverified",
        )

    def test_company_owned_publisher_remains_company_source(self):
        import web_search

        url = "https://www.apple.com/environment/pdf/report.pdf"
        document = {
            "source_type": web_search.classify_source(
                url,
                company_name="Apple",
            ),
            "source_domain": "www.apple.com",
            "source_url": url,
            "final_url": url,
            "publisher": "Apple",
        }

        self.assertEqual(
            document["source_type"],
            "Official company source",
        )
        self.assertEqual(identify_source_role(document), "company_source")

    def test_third_party_article_repeating_company_claim_is_not_evidence(self):
        import app
        import web_search

        url = (
            "https://theplanetbrief.com/esg/"
            "apple-environmental-progress-report-2026/"
        )
        document = {
            "status": "success",
            "title": "Apple Environmental Progress Report 2026",
            "text": (
                "Apple reported 15.3 million tonnes of gross greenhouse "
                "gas emissions for 2025, more than 60% below its 2015 "
                "footprint. The figures come from Apple's 2026 "
                "Environmental Progress Report."
            ),
            "source_type": web_search.classify_source(
                url,
                company_name="Apple",
            ),
            "source_domain": "theplanetbrief.com",
            "source_url": url,
            "final_url": url,
            "publisher": "The Planet Brief",
            "publication_date": "2026-08-08T13:30:17.000Z",
        }

        claims, external, statements = app.collect_candidate_statements(
            "Apple",
            [document],
        )

        self.assertEqual(claims, [])
        self.assertEqual(external, [])
        self.assertTrue(statements)
        repeated_claim = next(
            item
            for item in statements
            if "more than 60%" in item["statement"]
        )
        self.assertEqual(
            repeated_claim["source_role"],
            "external_source_unverified",
        )
        self.assertEqual(
            repeated_claim["candidate_status"],
            "rejected_external_candidate",
        )
        self.assertEqual(repeated_claim["publisher"], "The Planet Brief")
        self.assertEqual(repeated_claim["source_url"], url)
        self.assertEqual(
            repeated_claim["publication_date"],
            "2026-08-08T13:30:17.000Z",
        )
        self.assertIn(
            "not classified",
            repeated_claim["candidate_rejection_reason"],
        )

    def test_domain_name_company_guess_is_not_reported_as_verified_ownership(self):
        import app

        document = {
            "status": "success",
            "text": (
                "Acme reports renewable electricity use and its "
                "environmental progress."
            ),
            "source_type": "Potential company source (domain-name match)",
            "source_domain": "acme.example",
            "source_url": "https://acme.example/sustainability",
            "title": "Acme sustainability",
        }

        claims, external, _ = app.collect_candidate_statements(
            "Acme",
            [document],
        )

        self.assertEqual(external, [])
        self.assertEqual(len(claims), 1)
        self.assertEqual(
            claims[0]["source_role"],
            "potential_company_source_unverified",
        )
        self.assertEqual(
            claims[0]["statement_kind"],
            "Possible company-source candidate; not verified",
        )

    def test_source_label_or_flag_does_not_establish_independence(self):
        import app

        documents = [
            {
                "status": "success",
                "title": "Acme emissions assessment",
                "source_type": "Independent third-party source",
                "source_domain": "research.example",
                "source_url": "https://research.example/acme-review",
                "final_url": "https://research.example/acme-review",
                "text": (
                    "Acme Scope 1 emissions were 80 t CO2e in 2022 for "
                    "global operations under the GHG Protocol."
                ),
            },
            {
                "status": "success",
                "title": "Acme emissions assessment",
                "source_type": "Other / unverified",
                "source_domain": "research.example",
                "source_url": "https://research.example/acme-review",
                "final_url": "https://research.example/acme-review",
                "independence_verified": True,
                "text": (
                    "Acme Scope 1 emissions were 80 t CO2e in 2022 for "
                    "global operations under the GHG Protocol."
                ),
            },
        ]

        for document in documents:
            with self.subTest(document=document):
                claims, external, statements = (
                    app.collect_candidate_statements("Acme", [document])
                )

                self.assertEqual(claims, [])
                self.assertEqual(external, [])
                self.assertEqual(len(statements), 1)
                self.assertEqual(
                    statements[0]["source_role"],
                    "external_source_unverified",
                )
                self.assertEqual(
                    statements[0]["candidate_status"],
                    "rejected_external_candidate",
                )

    def test_company_entity_filter_isolated_for_different_names(self):
        import app

        acme_document = {
            "status": "success",
            "title": "Acme Environmental Report",
            "text": (
                "Acme reports greenhouse gas emissions from its "
                "operations in 2024."
            ),
            "source_type": "Official company source",
            "source_domain": "acme.example",
            "source_url": "https://acme.example/report",
        }
        globex_document = {
            **acme_document,
            "title": "Globex Environmental Report",
            "text": (
                "Globex reports greenhouse gas emissions from its "
                "operations in 2024."
            ),
            "source_domain": "globex.example",
            "source_url": "https://globex.example/report",
        }

        self.assertTrue(app.document_mentions_company("Acme", acme_document))
        self.assertFalse(
            app.document_mentions_company("Acme", globex_document)
        )
        self.assertTrue(
            app.document_mentions_company("Globex", globex_document)
        )
        self.assertFalse(
            app.document_mentions_company("Globex", acme_document)
        )

    def test_redirected_nonregulatory_domain_is_not_marked_regulatory(self):
        document = {
            "source_type": "Government / regulatory source",
            "source_domain": "www.sec.gov",
            "source_url": "https://www.sec.gov/filing",
            "final_url": "https://mirror.example/filing",
        }

        self.assertEqual(
            identify_source_role(document),
            "external_source_unverified",
        )

    def test_unconfigured_government_domain_is_not_regulatory_evidence(self):
        import web_search

        url = "https://county.gov/environment/report"
        document = {
            "source_type": web_search.classify_source(url),
            "source_domain": "county.gov",
            "source_url": url,
            "final_url": url,
        }

        self.assertEqual(
            document["source_type"],
            "Government source / regulatory status unverified",
        )
        self.assertEqual(
            identify_source_role(document),
            "external_source_unverified",
        )

    def test_pending_search_candidate_requires_page_entity_attribution(self):
        import app
        search_result = {
            "title": "Environmental disclosure index",
            "url": "https://research.example/disclosure-index",
            "snippet": "A review of corporate climate disclosures.",
            "search_query": '"Acme Holdings" environmental disclosure',
            "source_type": "Other / unverified",
            "attribution_status": "pending_page_content",
            "attribution_basis": (
                "Search query names Acme Holdings; page text must confirm."
            ),
        }

        rejected_document = {
            **search_result,
            "source_url": search_result["url"],
            "final_url": "https://research.example/final-index",
            "status": "success",
            "text": "A general overview of corporate climate disclosures.",
        }
        self.assertFalse(
            app.document_mentions_company("Acme Holdings", rejected_document)
        )

        diagnostics = app.build_pipeline_diagnostics(
            {"diagnostics": {}},
            [rejected_document],
            [],
            [],
            [],
            [],
            [],
        )
        rejection = diagnostics["pipeline"]["relevance"][
            "rejected_documents"
        ][0]
        self.assertEqual(
            rejection["reason"],
            "The page content did not confirm the requested entity; "
            "search metadata attribution was pending.",
        )
        self.assertEqual(
            rejection["search_query"],
            search_result["search_query"],
        )
        self.assertEqual(
            rejection["original_source_url"],
            search_result["url"],
        )
        self.assertEqual(
            rejection["source_url"],
            rejected_document["final_url"],
        )

        relevant_document = {
            **rejected_document,
            "text": (
                "Acme Holdings reports Scope 1 emissions in its "
                "environmental disclosure."
            ),
        }
        self.assertTrue(
            app.document_mentions_company("Acme Holdings", relevant_document)
        )
        incidental_mention = {
            **rejected_document,
            "text": (
                "Acme Holdings is listed in the directory. "
                "This page provides general climate emissions data."
            ),
        }
        self.assertFalse(
            app.document_mentions_company(
                "Acme Holdings",
                incidental_mention,
            )
        )
        subsidiary_document = {
            **rejected_document,
            "title": "Acme Services Environmental Report",
            "text": (
                "Acme Services reports Scope 1 emissions in this "
                "environmental report."
            ),
        }
        self.assertFalse(
            app.document_mentions_company(
                "Acme Holdings",
                subsidiary_document,
            )
        )

    def test_collection_document_preserves_pending_attribution_metadata(self):
        import automatic_collector

        result = {
            "title": "Environmental disclosure index",
            "snippet": "A review of climate disclosures.",
            "search_query": '"Acme Holdings" climate emissions',
            "source_type": "Other / unverified",
            "attribution_status": "pending_page_content",
            "attribution_basis": "Page content must confirm entity.",
        }

        document = automatic_collector.create_document(
            "https://research.example/disclosure",
            result,
        )

        self.assertEqual(document["search_query"], result["search_query"])
        self.assertEqual(document["attribution_status"], "pending_page_content")
        self.assertEqual(document["attribution_basis"], result["attribution_basis"])


    def test_risk_predictor_handles_missing_optional_feature_columns(self):
        features = pd.DataFrame([
            {
                "claim_id": 1,
                "claim": "Alpha claim",
                "similarity": 0.82,
                "evidence_available": 1,
                "nli_entailment": 0,
                "nli_contradiction": 0,
                "nli_neutral": 1,
                "historical_change": None,
            }
        ])

        result = predict_risk_from_features(features)

        self.assertIn("risk_score", result.columns)
        self.assertEqual(result.loc[0, "risk_score_status"], "HEURISTIC")
        self.assertTrue((result["risk_score"] >= 0).all())
        self.assertIn(
            result.loc[0, "demonstration_risk_category"],
            {"LOW", "MEDIUM", "HIGH"},
        )



    def test_rehosted_environmental_audit_is_not_independent_evidence(self):
        import app

        document = {
            "status": "success",
            "text": (
                "The Tata Motors environmental audit reviews waste "
                "management and emissions at its facilities."
            ),
            "source_type": "Other / unverified",
            "source_domain": "www.scribd.com",
            "source_url": (
                "https://www.scribd.com/document/example-tata-audit"
            ),
            "final_url": (
                "https://www.scribd.com/document/example-tata-audit"
            ),
            "title": "Detailed Environmental Audit Tata Motors | PDF",
            "publisher": "Scribd",
        }

        self.assertEqual(
            identify_source_role(document),
            "rehosted_report_unverified",
        )
        claims, external, statements = app.collect_candidate_statements(
            "Tata Motors",
            [document],
        )

        self.assertEqual(claims, [])
        self.assertEqual(external, [])
        self.assertTrue(statements)
        self.assertTrue(
            all(
                item["candidate_status"] == "rejected_external_candidate"
                and item["source_role"] == "rehosted_report_unverified"
                and "independent provenance was not established"
                in item["candidate_rejection_reason"]
                for item in statements
            )
        )


    def test_synthetic_collection_separates_company_and_unverified_sources(self):
        """Offline counterpart of the saved-collection check, using the
        committed synthetic fixture. Company-domain statements must be
        company claims, and statements from an unverified third-party
        source must be neither company claims nor verified external
        evidence."""
        import app

        path = PROJECT_DIR / "tests" / "fixtures" / "collection_synthetic.json"
        collection = json.loads(path.read_text(encoding="utf-8"))
        documents = collection["documents"]
        successful = [
            document for document in documents
            if (
                document.get("status") == "success"
                and str(document.get("text") or "").strip()
            )
        ]
        statements = [
            statement
            for document in successful
            for statement in extract_environmental_statements(document)
        ]

        self.assertGreater(len(documents), 0)
        self.assertGreater(len(successful), 0)
        self.assertGreater(len(statements), 0)
        self.assertTrue(
            all(statement.get("source_url") for statement in statements)
        )
        company_claims, external, _ = app.collect_candidate_statements(
            collection["search_name"],
            successful,
        )
        self.assertEqual(collection.get("search_name"), "Example Corp")
        self.assertTrue(company_claims)
        self.assertTrue(
            all(
                statement["source_role"] == "company_source"
                for statement in company_claims
            )
        )
        self.assertEqual(
            external,
            [],
            "An unverified third-party source must not become external evidence.",
        )
        self.assertFalse(
            any(
                "independent-review.test" in statement.get("source_url", "")
                for statement in company_claims
            ),
            "Statements from an unverified source must not be company claims.",
        )

    def test_matching_filters_sources_and_keeps_weak_matches_unconfirmed(self):
        statements = [
            {
                "statement": "Independent report on renewable energy.",
                "source_role": "external_source",
                "source_url": "https://research.example/report",
                "source_title": "Research report",
                "original_source_url": "https://origin.example/report",
                "publisher": "Research Institute",
                "source_domain": "research.example",
                "entity_context": "Acme",
                "publication_date": "2024-05-01",
            },
            {
                "statement": "Weakly related climate discussion.",
                "source_role": "external_source",
                "source_url": "https://news.example/article",
                "source_title": "News article",
            },
            {
                "statement": "Unrelated financial discussion.",
                "source_role": "external_source",
                "source_url": "https://news.example/finance",
            },
            {
                "statement": "Company sustainability statement.",
                "source_role": "company_source",
                "source_url": "https://company.example/esg",
            },
            {
                "statement": "A mirrored manufacturer report.",
                "source_role": "rehosted_report_unverified",
                "source_url": "https://archive.org/report",
            },
            {
                "statement": "External statement without a link.",
                "source_role": "external_source",
                "source_url": "",
            },
        ]

        with patch(
            "sentence_transformers.util.cos_sim",
            return_value=FakeSimilarityMatrix([[0.82, 0.42, 0.12]]),
        ):
            results = match_online_evidence(
                ["Company renewable electricity claim."],
                statements,
                FakeEmbeddingModel(),
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["matches_found"], 2)
        self.assertEqual(
            results[0]["evidence_status"],
            "potentially_relevant_evidence_found",
        )
        strong, weak = results[0]["matches"]
        self.assertEqual(strong["match_label"], "potentially_relevant")
        self.assertEqual(weak["match_label"], "weak_match")
        self.assertIn("Textual similarity only", weak["interpretation"])
        self.assertEqual(strong["source_title"], "Research report")
        self.assertEqual(strong["publisher"], "Research Institute")
        self.assertEqual(strong["source_domain"], "research.example")
        self.assertEqual(strong["entity_context"], "Acme")
        self.assertEqual(
            strong["source_url"],
            "https://research.example/report",
        )
        self.assertEqual(
            strong["original_source_url"],
            "https://origin.example/report",
        )
        self.assertEqual(strong["publication_date"], "2024-05-01")

        with patch(
            "sentence_transformers.util.cos_sim",
            return_value=FakeSimilarityMatrix([[0.12]]),
        ):
            no_match = match_online_evidence(
                ["Claim"],
                [statements[2]],
                FakeEmbeddingModel(),
            )
        self.assertEqual(
            no_match[0]["evidence_status"],
            "no_sufficiently_relevant_match",
        )
        self.assertEqual(
            match_online_evidence(["Claim"], statements[3:5], FakeEmbeddingModel()),
            [],
        )

    def test_matcher_excludes_unverified_external_source_role(self):
        statement = {
            "statement": "Acme emissions reporting and environmental data.",
            "source_role": "external_source_unverified",
            "source_url": "https://news.example/acme",
        }

        self.assertEqual(
            match_online_evidence(
                ["Acme emissions claim."],
                [statement],
                FakeEmbeddingModel(),
            ),
            [],
        )

    def test_matcher_accepts_distinct_attributed_source_roles(self):
        statements = [
            {
                "statement": "Independent emissions assessment for Acme.",
                "source_role": "independent_third_party_source",
                "source_url": "https://research.example/acme",
            },
            {
                "statement": "Configured climate data about Acme.",
                "source_role": "configured_environmental_source",
                "source_url": "https://climate.example/acme",
            },
        ]

        with patch(
            "sentence_transformers.util.cos_sim",
            return_value=FakeSimilarityMatrix([[0.81, 0.76]]),
        ):
            results = match_online_evidence(
                ["Acme environmental claim."],
                statements,
                FakeEmbeddingModel(),
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["matches_found"], 2)
        self.assertEqual(
            {
                match["source_role"]
                for match in results[0]["matches"]
            },
            {
                "independent_third_party_source",
                "configured_environmental_source",
            },
        )

    def test_low_similarity_or_duplicate_evidence_is_not_used_as_contradiction(self):
        statements = [
            {
                "statement": "Company reports 90% renewable electricity.",
                "source_role": "external_source",
                "source_url": "https://example.com/report",
                "source_title": "Example report",
                "publisher": "Example publisher",
                "source_domain": "example.com",
                "publication_date": "2024-03-01",
            },
            {
                "statement": "Company reports 90% renewable electricity.",
                "source_role": "external_source",
                "source_url": "https://example.com/report",
                "source_title": "Example report",
                "publisher": "Example publisher",
                "source_domain": "example.com",
                "publication_date": "2024-03-01",
            },
            {
                "statement": "Another company says nothing about this claim.",
                "source_role": "external_source",
                "source_url": "https://example.com/unrelated",
                "source_title": "Unrelated",
                "publisher": "Other",
                "source_domain": "example.com",
                "publication_date": "2024-03-02",
            },
        ]
        with patch(
            "sentence_transformers.util.cos_sim",
            return_value=FakeSimilarityMatrix([[0.91, 0.10]]),
        ):
            results = match_online_evidence(
                ["Company reports 90% renewable electricity."],
                statements,
                FakeEmbeddingModel(),
            )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["matches_found"], 1)
        self.assertEqual(results[0]["matches"][0]["source_url"], "https://example.com/report")

        low_similarity_data = {
            "results": [{
                "claim": "Claim",
                "matches": [{
                    "matched_evidence": "Evidence",
                    "similarity": 0.18,
                    "match_label": "no_sufficiently_relevant_match",
                    "source_title": "Weak source",
                    "source_url": "https://example.com/weak",
                    "publisher": "Pub",
                    "source_domain": "example.com",
                    "publication_date": "2024-01-01",
                }],
            }]
        }
        with patch(
            "transformers.pipeline",
            return_value=lambda *args, **kwargs: {"label": "CONTRADICTION", "score": 0.97},
        ):
            low_similarity_results = detect_online_contradictions(low_similarity_data)
        self.assertEqual(len(low_similarity_results), 1)
        self.assertEqual(low_similarity_results[0]["nli_result"], "UNAVAILABLE")
        self.assertIn("below the evidence threshold", low_similarity_results[0]["warning"])

    def test_nli_mock_labels_are_separate_from_similarity_and_keep_provenance(self):
        data = {
            "results": [{
                "claim": "Company reports lower emissions.",
                "matches": [
                    {
                        "matched_evidence": "Evidence supports lower emissions.",
                        "similarity": 0.64,
                        "match_label": "potentially_relevant",
                        "source_title": "Agency report",
                        "source_url": "https://epa.gov/report",
                        "publisher": "Agency",
                        "source_domain": "epa.gov",
                        "publication_date": "2024-01-01",
                    },
                    {
                        "matched_evidence": "Evidence conflicts with the claim.",
                        "similarity": 0.41,
                        "match_label": "weak_match",
                    },
                    {
                        "matched_evidence": "Evidence gives no clear answer.",
                        "similarity": 0.7,
                        "match_label": "potentially_relevant",
                    },
                    {"matched_evidence": ""},
                ],
            }],
        }
        predictions = [
            {"label": "ENTAILMENT", "score": 0.91},
            {"label": "NEUTRAL", "score": 0.76},
        ]

        with patch(
            "transformers.pipeline",
            return_value=FakeNLIClassifier(predictions),
        ) as pipeline:
            results = detect_online_contradictions(data)

        pipeline.assert_called_once()
        self.assertEqual(
            [row["nli_result"] for row in results],
            ["ENTAILMENT", "NEUTRAL", "UNAVAILABLE"],
        )
        self.assertEqual(results[0]["match_label"], "potentially_relevant")
        self.assertEqual(results[2]["match_label"], "weak_match")
        self.assertIn(
            "below the strong-match threshold",
            results[2]["warning"],
        )
        self.assertEqual(results[0]["source_title"], "Agency report")
        self.assertEqual(results[0]["publisher"], "Agency")
        self.assertEqual(results[0]["source_url"], "https://epa.gov/report")
        self.assertIn("not proof", results[0]["warning"])

        with patch("transformers.pipeline") as pipeline:
            self.assertEqual(
                detect_online_contradictions({
                    "results": [{
                        "claim": "Claim",
                        "matches": [{"matched_evidence": ""}],
                    }],
                }),
                [],
            )
            pipeline.assert_not_called()

        with patch(
            "transformers.pipeline",
            side_effect=RuntimeError("offline model error"),
        ):
            with self.assertRaisesRegex(RuntimeError, "offline model error"):
                detect_online_contradictions({
                    "results": [{
                        "claim": "Claim",
                        "matches": [{"matched_evidence": "Evidence"}],
                    }],
                })

    def test_risk_prediction_and_explanation_match_the_heuristic(self):
        matching_results = [{
            "claim": "The company reports 50% renewable electricity.",
            "matches": [{
                "matched_evidence": "The report discusses renewable electricity.",
                "similarity": 0.42,
                "source_url": "https://independent.example/report.pdf",
            }],
        }]
        # NLI rows must carry their pair identity; coverage is attributed to
        # matches through (claim, evidence, source_url).
        nli_results = [{
            "claim": matching_results[0]["claim"],
            "evidence": "The report discusses renewable electricity.",
            "source_url": "https://independent.example/report.pdf",
            "nli_result": "NEUTRAL",
            "scoring_eligible": True,
        }]
        features = build_features_from_results(
            matching_results,
            nli_results,
        )
        features["similarity"] = "invalid numeric value"
        features["nli_entailment"] = None

        predictions = predict_risk_from_features(features)
        # A missing NLI count and an invalid similarity are unassessed, not zero.
        # The row must carry no score rather than a score built from missing values.
        self.assertTrue(pd.isna(predictions.iloc[0]["risk_score"]))
        self.assertEqual(predictions.iloc[0]["risk_score_status"], "NOT_PREDICTED")
        self.assertTrue(
            predictions["demonstration_risk_probability"].isna().all()
        )
        self.assertEqual(
            predictions.iloc[0]["demonstration_risk_category"],
            "NOT_PREDICTED",
        )

        # An unassessed row is not explained, because there is no score to
        # explain. The explainer returns [] for it (see test_risk_unavailable_states).
        explanations = explain_risk_predictions(
            predictions,
            None,
            "Offline Demo Company",
        )
        self.assertEqual(explanations, [])
        self.assertTrue(
            all(
                row["company"] == "Offline Demo Company"
                and row["claim"] == matching_results[0]["claim"]
                and row["feature"] in FEATURE_COLUMNS
                for row in explanations
            )
        )
        self.assertEqual(explain_risk_predictions(pd.DataFrame(), None, "X"), [])

        with self.assertRaisesRegex(ValueError, "Missing explanation columns"):
            explain_risk_predictions(
                pd.DataFrame([{"claim_id": 1}]),
                None,
                "Offline Demo Company",
            )

        with self.assertRaisesRegex(ValueError, "separate trained model"):
            explain_risk_predictions(predictions, train_risk_model(), "X")

    def test_timeline_uses_explicit_years_and_withholds_missing_years(self):
        documents = [{
            "status": "success",
            "title": "Company ESG report",
            "publisher": "Example Publisher",
            "publication_date": "2023-04-01",
            "source_url": "https://company.example/esg",
            "text": (
                "Emissions were 100 t CO2e in 2021 for Scope 1. "
                "Emissions were 80 t CO2e in 2022 for Scope 1. "
                "Renewable electricity supplied 70% of total electricity."
            ),
        }]

        records = analyze_timeline(documents)
        reported = [
            record for record in records
            if record["performance_type"] == "Reported result"
        ]
        self.assertEqual(
            [record["reporting_year"] for record in reported],
            [2021, 2022],
        )
        self.assertIn("decreased by 20", reported[1]["change_over_time"])
        self.assertEqual(
            reported[0]["source_url"],
            "https://company.example/esg",
        )
        metrics = get_comparable_metrics(records)
        emissions_metric = next(
            metric for metric in metrics
            if metric["metric"] == "Absolute emissions"
        )
        trend_points = build_trend_points(records, emissions_metric)
        self.assertEqual(
            [(point["Year"], point["Value"]) for point in trend_points],
            [(2021, 100.0), (2022, 80.0)],
        )
        self.assertTrue(
            all(
                "Company ESG report" in point["Source titles"]
                and "https://company.example/esg" in point["Source URLs"]
                for point in trend_points
            )
        )
        changes = build_timeline_change_rows(records, emissions_metric)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["Absolute change"], -20)
        self.assertEqual(changes[0]["Percentage change"], -20)
        self.assertEqual(changes[0]["Direction"], "decreased")
        self.assertEqual(reported[0]["source_publisher"], "Example Publisher")

        no_year = [{
            "status": "success",
            "publication_date": "2025-01-01",
            "text": "The company reduced carbon emissions by 20%.",
        }]
        self.assertEqual(extract_timeline_records(no_year), [])

    def test_timeline_does_not_treat_possessive_baseline_as_reporting_year(self):
        source_url = "https://company.example/esg"
        baseline_only = [{
            "status": "success",
            "title": "Company ESG report",
            "source_url": source_url,
            "text": (
                "90 percent reduction in emissions from our 2015 baseline."
            ),
        }]
        self.assertEqual(extract_timeline_records(baseline_only), [])

        reported_measurement = [{
            "status": "success",
            "title": "Company ESG report",
            "source_url": source_url,
            "text": (
                "In 2024 we reduced emissions 90 percent "
                "from our 2015 baseline."
            ),
        }]
        records = extract_timeline_records(reported_measurement)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["reporting_year"], 2024)
        self.assertEqual(records[0]["timeline_year"], 2024)
        self.assertEqual(records[0]["source_url"], source_url)

    def test_timeline_requires_multiple_comparable_years_and_units(self):
        one_year = analyze_timeline([{
            "status": "success",
            "title": "Single-year report",
            "source_url": "https://company.example/one-year",
            "text": "Emissions were 100 t CO2e in 2022 for Scope 1.",
        }])
        self.assertEqual(get_comparable_metrics(one_year), [])
        self.assertEqual(build_trend_points(one_year, {
            "metric": "Absolute emissions",
            "metric_key": "absolute_emissions",
            "unit": "t CO2e",
            "boundary": "scope 1",
        }), [])
        self.assertEqual(build_timeline_change_rows(one_year, {
            "metric": "Absolute emissions",
            "metric_key": "absolute_emissions",
            "unit": "t CO2e",
            "boundary": "scope 1",
        }), [])

        incompatible_units = analyze_timeline([
            {
                "status": "success",
                "title": "2021 report",
                "text": (
                    "Renewable energy consumption was 50 MWh in 2021."
                ),
            },
            {
                "status": "success",
                "title": "2022 report",
                "text": (
                    "Renewable energy consumption was 60 GWh in 2022."
                ),
            },
        ])
        self.assertEqual(get_comparable_metrics(incompatible_units), [])

    def test_timeline_separates_reported_measurement_from_target_year(self):
        records = analyze_timeline([{
            "status": "success",
            "source_url": "https://company.example/report",
            "text": (
                "Renewable electricity was 40% in 2024, with a target "
                "of 100% by 2030."
            ),
        }])

        reported, target = records
        self.assertEqual(reported["performance_type"], "Reported result")
        self.assertEqual(reported["value"], 40.0)
        self.assertEqual(reported["reporting_year"], 2024)
        self.assertIsNone(reported["target_year"])
        self.assertEqual(target["performance_type"], "Target")
        self.assertEqual(target["value"], 100.0)
        self.assertIsNone(target["reporting_year"])
        self.assertEqual(target["target_year"], 2030)
        self.assertEqual(target["timeline_year"], 2030)

    def test_timeline_conflict_detection_does_not_mix_different_metrics(self):
        from timeline_analyzer import find_timeline_review_points

        records = [
            {
                "reporting_year": 2024,
                "metric": "Renewable electricity share",
                "metric_key": "renewable_electricity_percent",
                "value": 80.0,
                "unit": "%",
                "boundary": "not stated",
                "performance_type": "Reported result",
            },
            {
                "reporting_year": 2024,
                "metric": "Emissions reduction",
                "metric_key": "emissions_reduction_percent",
                "value": 30.0,
                "unit": "%",
                "boundary": "not stated",
                "performance_type": "Reported result",
            },
        ]
        self.assertEqual(find_timeline_review_points(records), [])








    def test_company_pipeline_connects_claims_nli_risk_and_shap_offline(self):
        import app

        company_document = {
            "status": "success",
            "title": "Acme Sustainability Report",
            "source_type": "Official company source",
            "source_domain": "acme.example",
            "source_url": "https://acme.example/esg",
            "final_url": "https://acme.example/esg",
            "publication_date": "2025-01-01",
            "text": (
                "Acme reports that renewable electricity supplied "
                "65% of its operations in 2024."
            ),
        }
        evidence_document = {
            "status": "success",
            "title": "Independent Agency Report",
            "source_type": "Government source",
            "source_domain": "epa.gov",
            "source_url": "https://epa.gov/acme",
            "final_url": "https://epa.gov/acme",
            "publication_date": "2025-02-01",
            "text": (
                "An independent review of Acme found renewable "
                "electricity use was not consistently reported in 2024."
            ),
        }
        documents = [company_document, evidence_document]
        company_claim = extract_environmental_statements(company_document)[0]
        evidence = extract_environmental_statements(evidence_document)[0]
        matching_result = {
            "claim": company_claim["statement"],
            "evidence_status": "potentially_relevant_evidence_found",
            "matches_found": 1,
            "matches": [{
                "matched_evidence": evidence["statement"],
                "similarity": 0.62,
                "match_label": "potentially_relevant",
                "source_title": evidence["source_title"],
                "source_url": evidence["source_url"],
                "publisher": evidence["publisher"],
                "source_domain": evidence["source_domain"],
                "source_role": evidence["source_role"],
                "publication_date": evidence["publication_date"],
            }],
        }
        collection = {
            "search_results_count": 2,
            "successful_documents": 2,
            "output_file": "mocked-collection.json",
            "documents": documents,
        }
        stages = []

        with (
            patch.object(
                app,
                "collect_company_esg_information",
                return_value=collection,
            ),
            patch.object(
                app,
                "collect_claim_specific_evidence",
                return_value=([], None, []),
            ),
            patch.object(
                app,
                "match_claims_to_evidence",
                return_value=[matching_result],
            ),
            patch.object(
                app,
                "detect_online_contradictions",
                return_value=[{
                    "claim": company_claim["statement"],
                    "evidence": evidence["statement"],
                    "source_url": evidence["source_url"],
                    "nli_result": "NEUTRAL",
                    "confidence": 0.8,
                }],
            ),
        ):
            analysis = app.build_analysis(
                "Acme",
                stage_callback=lambda *args: stages.append(args),
            )

        self.assertEqual(len(analysis["company_claims"]), 1)
        self.assertEqual(len(analysis["external_evidence"]), 1)
        self.assertEqual(
            analysis["matching_results"][0]["matches"][0]["source_url"],
            "https://epa.gov/acme",
        )
        self.assertEqual(analysis["nli_results"][0]["nli_result"], "NEUTRAL")
        self.assertEqual(len(analysis["features"]), 1)
        self.assertIn(analysis["features"][0]["demonstration_risk_category"], {
            "LOW",
            "MEDIUM",
            "HIGH",
        })
        self.assertTrue(analysis["risk_explanations"])
        self.assertTrue(
            all(
                row["company"] == "Acme"
                and row["claim"] == company_claim["statement"]
                for row in analysis["risk_explanations"]
            )
        )
        self.assertTrue(any(stage[0] == "Risk Prediction" for stage in stages))

    def test_eligible_sources_reach_matching_nli_and_experimental_risk(self):
        import app

        claim_document = {
            "status": "success",
            "title": "Acme Sustainability Report",
            "source_type": "Official company source",
            "source_domain": "www.acme.example",
            "source_url": "https://www.acme.example/esg",
            "final_url": "https://www.acme.example/esg",
            "publication_date": "2023-03-01",
            "text": (
                "Acme reported Scope 1 emissions of 100 t CO2e in 2022 for global "
                "operations under the GHG Protocol using the location-based "
                "method."
            ),
        }
        regulatory_document = {
            "status": "success",
            "title": "Agency Acme emissions record",
            "source_type": "Government / regulatory source",
            "source_domain": "epa.gov",
            "source_url": "https://epa.gov/records/acme-2022",
            "final_url": "https://epa.gov/records/acme-2022",
            "publication_date": "2023-04-01",
            "text": (
                "The regulatory record reports Acme's Scope 1 emissions "
                "were 80 t CO2e in 2022 for global operations under the GHG "
                "Protocol using the location-based method."
            ),
        }
        configured_document = {
            "status": "success",
            "title": "Configured Acme emissions record",
            "source_type": "Configured environmental source",
            "source_domain": "ipcc.ch",
            "source_url": "https://www.ipcc.ch/acme-assessment",
            "final_url": "https://www.ipcc.ch/acme-assessment",
            "publication_date": "2023-05-01",
            "text": (
                "The configured environmental record reports Acme's Scope 1 emissions "
                "were 100 t CO2e in 2022 for global operations under the GHG "
                "Protocol using the location-based method."
            ),
        }
        collection = {
            "search_results_count": 3,
            "successful_documents": 3,
            "output_file": None,
            "search_errors": [],
            "documents": [
                claim_document,
                regulatory_document,
                configured_document,
            ],
        }
        classifier = FakeNLIClassifier([
            {"label": "NEUTRAL", "score": 0.91},
            {"label": "NEUTRAL", "score": 0.88},
        ])
        stages = []

        with (
            patch.object(
                app,
                "collect_company_esg_information",
                return_value=collection,
            ),
            patch.object(
                app,
                "collect_claim_specific_evidence",
                return_value=([], None, []),
            ),
            patch.object(
                app,
                "load_embedding_model",
                return_value=FakeEmbeddingModel(),
            ),
            patch(
                "sentence_transformers.util.cos_sim",
                return_value=FakeSimilarityMatrix([[0.91, 0.89]]),
            ),
            patch(
                "contradiction_detector._get_nli_classifier",
                return_value=classifier,
            ),
        ):
            analysis = app.build_analysis(
                "Acme",
                stage_callback=lambda *args: stages.append(args),
            )

        self.assertEqual(len(analysis["company_claims"]), 1)
        self.assertEqual(
            {
                item["source_role"]
                for item in analysis["external_evidence"]
            },
            {
                "external_regulatory_evidence",
                "configured_environmental_source",
            },
        )
        self.assertEqual(len(analysis["matching_results"]), 1)
        matching_result = analysis["matching_results"][0]
        self.assertEqual(matching_result["matches_found"], 2)
        self.assertEqual(
            {
                item["source_role"]
                for item in matching_result["matches"]
            },
            {
                "external_regulatory_evidence",
                "configured_environmental_source",
            },
        )
        self.assertEqual(len(analysis["nli_results"]), 2)
        self.assertTrue(
            all(
                item["nli_result"] == "NEUTRAL"
                and item["compatibility_status"] == "COMPATIBLE"
                and item["scoring_eligible"]
                for item in analysis["nli_results"]
            )
        )
        self.assertEqual(len(analysis["features"]), 1)
        self.assertEqual(analysis["features"][0]["evidence_available"], 1)
        self.assertEqual(
            analysis["features"][0]["risk_score_status"],
            "HEURISTIC",
        )
        self.assertIn(
            analysis["features"][0]["demonstration_risk_category"],
            {"LOW", "MEDIUM", "HIGH"},
        )
        self.assertEqual(
            analysis["diagnostics"]["pipeline"]["evidence"][
                "matched_pair_count"
            ],
            2,
        )
        self.assertTrue(
            any(
                stage[0] == "Evidence Matching" and stage[1] == "COMPLETED"
                for stage in stages
            )
        )
        self.assertTrue(
            any(
                stage[0] == "NLI Analysis" and stage[1] == "COMPLETED"
                for stage in stages
            )
        )
        self.assertTrue(
            any(
                stage[0] == "Risk Prediction" and stage[1] == "COMPLETED"
                for stage in stages
            )
        )

    def test_weak_only_evidence_does_not_mark_nli_stage_completed(self):
        import app

        company_document = {
            "status": "success",
            "title": "Microsoft Sustainability Report",
            "source_type": "Official company source",
            "source_domain": "www.microsoft.com",
            "source_url": "https://www.microsoft.com/sustainability",
            "final_url": "https://www.microsoft.com/sustainability",
            "text": (
                "Microsoft reported that Scope 3 emissions increased by "
                "26% from its 2020 baseline."
            ),
        }
        independent_document = {
            "status": "success",
            "title": "Independent Microsoft climate assessment",
            "source_type": "Independent third-party source",
            "source_domain": "research.example",
            "source_url": "https://research.example/microsoft-assessment",
            "final_url": "https://research.example/microsoft-assessment",
            "independence_verified": True,
            "text": (
                "An independent assessment discusses Microsoft's "
                "supply-chain Scope 3 emissions."
            ),
        }
        company_claim = extract_environmental_statements(company_document)[0]
        matching_result = {
            "claim": company_claim["statement"],
            "evidence_status": "only_weak_matches_found",
            "matches_found": 1,
            "matches": [{
                "matched_evidence": (
                    "An independent assessment discusses Microsoft's "
                    "supply-chain Scope 3 emissions."
                ),
                "similarity": 0.39,
                "match_label": "weak_match",
                "source_title": independent_document["title"],
                "source_url": independent_document["source_url"],
                "original_source_url": independent_document["source_url"],
                "publisher": "research.example",
                "source_domain": "research.example",
                "source_role": "independent_third_party_source",
                "entity_context": "Microsoft",
            }],
        }
        collection = {
            "search_results_count": 2,
            "successful_documents": 2,
            "output_file": None,
            "documents": [company_document, independent_document],
        }
        stages = []

        with (
            patch.object(
                app,
                "collect_company_esg_information",
                return_value=collection,
            ),
            patch.object(
                app,
                "collect_claim_specific_evidence",
                return_value=([], None, []),
            ),
            patch.object(
                app,
                "match_claims_to_evidence",
                return_value=[matching_result],
            ),
            patch(
                "contradiction_detector._get_nli_classifier",
                side_effect=AssertionError(
                    "Weak evidence must not reach the NLI classifier."
                ),
            ) as get_classifier,
            patch.object(
                app,
                "load_demo_risk_model",
                side_effect=AssertionError(
                    "Evidence-unavailable claims must not reach risk prediction."
                ),
            ) as load_risk_model,
        ):
            analysis = app.build_analysis(
                "Microsoft",
                stage_callback=lambda *args: stages.append(args),
            )

        get_classifier.assert_not_called()
        load_risk_model.assert_not_called()
        self.assertEqual(analysis["nli_results"][0]["nli_result"], "UNAVAILABLE")
        self.assertFalse(analysis["nli_results"][0]["scoring_eligible"])
        self.assertEqual(analysis["features"][0]["evidence_available"], 0)
        self.assertEqual(analysis["features"][0]["risk_score"], None)
        self.assertEqual(
            analysis["features"][0]["risk_score_status"],
            "NOT_PREDICTED",
        )
        nli_stage = next(
            stage for stage in reversed(stages)
            if stage[0] == "NLI Analysis"
        )
        self.assertEqual(nli_stage[1], "PARTIAL")
        self.assertIn("unavailable or ineligible", nli_stage[2])
        matching_stage = next(
            stage for stage in reversed(stages)
            if stage[0] == "Evidence Matching"
        )
        self.assertEqual(matching_stage[1], "PARTIAL")
        risk_stage = next(
            stage for stage in reversed(stages)
            if stage[0] == "Risk Prediction"
        )
        self.assertEqual(risk_stage[1], "NOT AVAILABLE")

    def test_mixed_evidence_marks_nli_and_risk_stages_partial(self):
        import app

        company_document = {
            "status": "success",
            "title": "Acme Sustainability Report",
            "source_type": "Official company source",
            "source_domain": "www.acme.example",
            "source_url": "https://www.acme.example/esg",
            "final_url": "https://www.acme.example/esg",
            "text": "Acme reported Scope 1 emissions of 100 t CO2e in 2024.",
        }
        company_claims = [
            {
                "statement": "Acme reported Scope 1 emissions of 100 t CO2e in 2024.",
                "source_title": company_document["title"],
                "source_url": company_document["source_url"],
                "source_role": "company_source",
            },
            {
                "statement": "Acme reported Scope 2 emissions of 50 t CO2e in 2024.",
                "source_title": company_document["title"],
                "source_url": company_document["source_url"],
                "source_role": "company_source",
            },
        ]
        independent_candidate = {
            "statement": "An independent source reports Acme emissions.",
            "source_title": "Independent Acme record",
            "source_url": "https://research.example/acme",
            "source_role": "independent_third_party_source",
        }
        eligible_evidence = "An independent record reports Acme Scope 1 emissions."
        weak_evidence = "A weak external source mentions Acme."
        matching_results = [
            {
                "claim": company_claims[0]["statement"],
                "evidence_status": "potentially_relevant_evidence_found",
                "matches_found": 1,
                "matches": [{
                    "matched_evidence": eligible_evidence,
                    "similarity": 0.9,
                    "match_label": "strong_match",
                    "source_title": "Independent Acme record",
                    "source_url": "https://research.example/acme",
                    "source_role": "independent_third_party_source",
                }],
            },
            {
                "claim": company_claims[1]["statement"],
                "evidence_status": "only_weak_matches_found",
                "matches_found": 1,
                "matches": [{
                    "matched_evidence": weak_evidence,
                    "similarity": 0.39,
                    "match_label": "weak_match",
                    "source_title": "Weak Acme mention",
                    "source_url": "https://research.example/weak",
                    "source_role": "independent_third_party_source",
                }],
            },
        ]
        nli_results = [
            {
                "claim": company_claims[0]["statement"],
                "evidence": eligible_evidence,
                "source_url": "https://research.example/acme",
                "nli_result": "NEUTRAL",
                "compatibility_status": "COMPATIBLE",
                "scoring_eligible": True,
            },
            {
                "claim": company_claims[1]["statement"],
                "evidence": weak_evidence,
                "source_url": "https://research.example/weak",
                "nli_result": "UNAVAILABLE",
                "compatibility_status": "INCONCLUSIVE",
                "scoring_eligible": False,
            },
        ]
        collection = {
            "search_results_count": 1,
            "successful_documents": 1,
            "output_file": None,
            "documents": [company_document],
        }
        stages = []

        with (
            patch.object(
                app,
                "collect_company_esg_information",
                return_value=collection,
            ),
            patch.object(
                app,
                "collect_candidate_statements",
                side_effect=[
                    (company_claims, [], company_claims),
                        (
                            company_claims,
                            [independent_candidate],
                            [*company_claims, independent_candidate],
                        ),
                    ],
                ),
                patch.object(
                    app,
                    "collect_claim_specific_evidence",
                    side_effect=RuntimeError("Search unavailable."),
            ),
            patch.object(
                app,
                "match_claims_to_evidence",
                return_value=matching_results,
            ),
            patch.object(
                app,
                "detect_online_contradictions",
                return_value=nli_results,
            ),
            patch.object(
                app,
                "explain_risk_predictions",
                return_value=[{"claim": company_claims[0]["statement"]}],
            ),
        ):
            analysis = app.build_analysis(
                "Acme",
                stage_callback=lambda *args: stages.append(args),
            )

        self.assertEqual(
            [item["evidence_available"] for item in analysis["features"]],
            [1, 0],
        )
        for stage_name in (
            "Independent Evidence Search",
            "Evidence Matching",
            "NLI Analysis",
            "Risk Prediction",
            "Risk Explanation",
        ):
            stage = next(
                item for item in reversed(stages)
                if item[0] == stage_name
            )
            self.assertEqual(stage[1], "PARTIAL")

    def test_claim_specific_evidence_collection_deduplicates_known_urls(self):
        import app

        search_results = [
            {
                "url": "https://research.example/acme-2024",
                "title": "Independent Acme emissions review",
            },
            {
                "url": "https://acme.example/report",
                "title": "Already-collected company report",
            },
        ]
        collected = [{
            "status": "success",
            "source_url": "https://research.example/acme-2024",
            "text": "Acme emissions review",
        }]
        with (
            patch.object(
                app,
                "search_web",
                return_value=search_results,
            ) as search,
            patch.object(
                app,
                "collect_documents",
                return_value=(collected, "supplemental.json"),
            ) as collect,
        ):
            documents, output_file, diagnostics = (
                app.collect_claim_specific_evidence(
                    "Acme",
                    [{
                        "statement": (
                            "Acme Scope 1 emissions were 100 t CO2e "
                            "in 2024."
                        ),
                    }],
                    [{"source_url": "https://acme.example/report"}],
                )
            )

        self.assertEqual(documents, collected)
        self.assertEqual(output_file, "supplemental.json")
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(search.call_count, 2)
        self.assertIn('"100 t CO2e"', search.call_args_list[0].args[0])
        self.assertNotIn('"100 t CO2e"', search.call_args_list[1].args[0])
        collect.assert_called_once_with(
            [search_results[0]],
            "Acme",
            max_documents=1,
        )

    def test_claim_specific_collection_prioritizes_external_candidates(self):
        import app

        company_result = {
            "url": "https://www.samsung.com/esg/report",
            "title": "Samsung Sustainability Report",
            "source_type": "Official company source",
        }
        external_result = {
            "url": "https://research.example/samsung-energy-review",
            "title": "Samsung Renewable Energy Review",
            "source_type": "Other / unverified",
        }
        collected = [{
            "status": "success",
            "source_url": external_result["url"],
            "text": "Review of renewable energy figures.",
        }]
        with (
            patch.object(
                app,
                "search_web",
                side_effect=[
                    [company_result],
                    [external_result],
                    [],
                ],
            ) as patch_search,
            patch.object(
                app,
                "collect_documents",
                return_value=(collected, "supplemental.json"),
            ) as collect,
        ):
            documents, output_file, _ = app.collect_claim_specific_evidence(
                "Samsung",
                [{
                    "statement": (
                        "Renewable energy use was 8,704 GWh in 2022 "
                        "at a 31% transition rate."
                    ),
                }],
                [],
            )

        self.assertEqual(documents, collected)
        self.assertEqual(output_file, "supplemental.json")
        self.assertEqual(
            collect.call_args.args[0],
            [external_result, company_result],
        )
        self.assertEqual(
            [call.kwargs["backend"] for call in patch_search.call_args_list],
            ["google", "brave", "yahoo"],
        )


    def test_database_saves_and_reads_only_from_temporary_database(self):
        features = build_features_from_results(
            [{
                "claim": "Claim about renewable energy.",
                "matches": [{"similarity": 0.45}],
            }],
            [],
        )
        predictions = predict_risk_from_features(
            features,
            model=train_risk_model(),
        )
        timeline_records = analyze_timeline([{
            "status": "success",
            "title": "ESG report",
            "source_url": "https://company.example/report",
            "text": "Emissions were 100 t CO2e in 2022 for Scope 1.",
        }])

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "test.sqlite"
            self.assertEqual(
                save_company_predictions(
                    "Offline Demo Company",
                    "analysis-test",
                    predictions,
                    database_path,
                ),
                0,
            )
            self.assertEqual(
                save_timeline_records(
                    "Offline Demo Company",
                    "analysis-test",
                    timeline_records,
                    database_path,
                ),
                1,
            )
            connection = sqlite3.connect(database_path)
            try:
                prediction = connection.execute(
                    "SELECT company_name, analysis_id, claim_id, "
                    "risk_score, demonstration_risk_probability "
                    "FROM risk_predictions"
                ).fetchone()
                timeline = connection.execute(
                    "SELECT company_name, timeline_year, source_url "
                    "FROM environmental_timeline"
                ).fetchone()
            finally:
                connection.close()

        self.assertIsNone(prediction)
        self.assertEqual(
            timeline,
            (
                "Offline Demo Company",
                2022,
                "https://company.example/report",
            ),
        )

    def test_database_does_not_save_unpredicted_feature_rows(self):
        predictions = pd.DataFrame([
            {
                "claim_id": 1,
                "similarity": 0.8,
                "evidence_available": 1,
                "nli_entailment": 1,
                "nli_contradiction": 0,
                "nli_neutral": 0,
                "percentage_difference": None,
                "historical_change": None,
                "risk_score": 15.0,
                "predicted_risk_class": 0,
                "demonstration_risk_probability": None,
                "demonstration_risk_category": "LOW",
                "risk_score_status": "HEURISTIC",
            },
            {
                "claim_id": 2,
                "similarity": None,
                "evidence_available": 0,
                "nli_entailment": None,
                "nli_contradiction": None,
                "nli_neutral": None,
                "percentage_difference": None,
                "historical_change": None,
                "risk_score": None,
                "predicted_risk_class": None,
                "demonstration_risk_probability": None,
                "demonstration_risk_category": "NOT_PREDICTED",
                "risk_score_status": "NOT_PREDICTED",
            },
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "risk.sqlite"
            inserted = save_company_predictions(
                "Offline Demo Company",
                "analysis-test",
                predictions,
                database_path,
            )
            connection = sqlite3.connect(database_path)
            try:
                rows = connection.execute(
                    "SELECT claim_id, risk_score FROM risk_predictions"
                ).fetchall()
            finally:
                connection.close()

        self.assertEqual(inserted, 1)
        self.assertEqual(rows, [(1, 15.0)])

    def test_database_adds_heuristic_score_without_losing_legacy_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "legacy.sqlite"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute("""
                    CREATE TABLE risk_predictions (
                        id INTEGER PRIMARY KEY,
                        claim_id INTEGER,
                        similarity REAL,
                        evidence_available INTEGER,
                        nli_entailment REAL,
                        nli_contradiction REAL,
                        nli_neutral REAL,
                        percentage_difference REAL,
                        historical_change REAL,
                        predicted_risk_class INTEGER,
                        demonstration_risk_probability REAL,
                        demonstration_risk_category TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                connection.execute(
                    "INSERT INTO risk_predictions "
                    "(claim_id, demonstration_risk_probability) VALUES (1, 0.5)"
                )
                connection.commit()
            finally:
                connection.close()

            initialize_database(database_path)
            connection = sqlite3.connect(database_path)
            try:
                row = connection.execute(
                    "SELECT claim_id, demonstration_risk_probability, risk_score "
                    "FROM risk_predictions"
                ).fetchone()
            finally:
                connection.close()

        self.assertEqual(row, (1, 0.5, None))

    def _analysis_payload_for_persistence_test(self):
        return {
            "company": "Atomic Save Co",
            "collection": {"documents": []},
            "company_claims": [],
            "matching_results": [],
            "nli_results": [],
            "features": [],
            "timeline_records": [],
        }

    def test_interrupted_analysis_write_does_not_publish_partial_json(self):
        import app

        with tempfile.TemporaryDirectory() as temp_dir:
            analyses_dir = Path(temp_dir)
            analysis = self._analysis_payload_for_persistence_test()

            named_temporary_file = app.tempfile.NamedTemporaryFile

            def interrupted_temporary_file(*args, **kwargs):
                temporary_file = named_temporary_file(*args, **kwargs)

                class PartialWriteFailure:
                    def __enter__(self):
                        self.file = temporary_file.__enter__()
                        return self

                    def __exit__(self, *exc_info):
                        return self.file.__exit__(*exc_info)

                    @property
                    def name(self):
                        return self.file.name

                    def write(self, text):
                        self.file.write(text[:1])
                        raise OSError("simulated interrupted write")

                    def __getattr__(self, name):
                        return getattr(self.file, name)

                return PartialWriteFailure()

            with (
                patch.object(app, "ANALYSES_DIR", analyses_dir),
                patch.object(app, "save_timeline_records", return_value=0),
                patch.object(
                    app.tempfile,
                    "NamedTemporaryFile",
                    side_effect=interrupted_temporary_file,
                ),
            ):
                app.persist_analysis(analysis)

            expected_path = (
                analyses_dir
                / f"atomic_save_co_{analysis['analysis_id'][:16]}.json"
            )
            self.assertFalse(expected_path.exists())
            self.assertIn("OSError", analysis["persistence_error"])
            self.assertEqual(list(analyses_dir.iterdir()), [])

    def test_invalid_temporary_analysis_json_is_not_published(self):
        import app

        with tempfile.TemporaryDirectory() as temp_dir:
            analyses_dir = Path(temp_dir)
            analysis = self._analysis_payload_for_persistence_test()

            with (
                patch.object(app, "ANALYSES_DIR", analyses_dir),
                patch.object(app, "save_timeline_records", return_value=0),
                patch.object(
                    app,
                    "dumps_analysis_json",
                    return_value='{"incomplete":',
                ),
            ):
                app.persist_analysis(analysis)

            expected_path = (
                analyses_dir
                / f"atomic_save_co_{analysis['analysis_id'][:16]}.json"
            )
            self.assertFalse(expected_path.exists())
            self.assertIn("JSONDecodeError", analysis["persistence_error"])
            self.assertEqual(list(analyses_dir.iterdir()), [])

    def test_existing_valid_analysis_is_preserved(self):
        import app

        with tempfile.TemporaryDirectory() as temp_dir:
            analyses_dir = Path(temp_dir)
            analysis = self._analysis_payload_for_persistence_test()
            with (
                patch.object(app, "ANALYSES_DIR", analyses_dir),
                patch.object(app, "save_timeline_records", return_value=0),
            ):
                app.persist_analysis(analysis)
                analysis_path = Path(analysis["analysis_path"])
                original_bytes = analysis_path.read_bytes()
                app.persist_analysis(analysis)

            self.assertEqual(analysis_path.read_bytes(), original_bytes)
            self.assertEqual(json.loads(original_bytes)["company"], "Atomic Save Co")
            self.assertEqual(
                [path.name for path in analyses_dir.iterdir()],
                [analysis_path.name],
            )

    def test_existing_invalid_analysis_is_preserved_and_reports_error(self):
        import app

        with tempfile.TemporaryDirectory() as temp_dir:
            analyses_dir = Path(temp_dir)
            analysis = self._analysis_payload_for_persistence_test()
            identity_data = {
                "company": app.normalize_company_name(analysis["company"]),
                "documents": [],
                "claims": [],
                "matches": [],
                "nli": [],
                "features": [],
                "timeline_records": [],
            }
            analysis_id = hashlib.sha256(
                json.dumps(
                    identity_data,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            safe_company = (
                app.normalize_company_name(analysis["company"]) or "company"
            )
            analysis_path = analyses_dir / f"{safe_company}_{analysis_id[:16]}.json"
            original_bytes = b'{"partial":'
            analysis_path.write_bytes(original_bytes)

            with (
                patch.object(app, "ANALYSES_DIR", analyses_dir),
                patch.object(app, "save_timeline_records", return_value=0),
            ):
                app.persist_analysis(analysis)

            self.assertEqual(analysis_path.read_bytes(), original_bytes)
            self.assertIn("JSONDecodeError", analysis["persistence_error"])
            self.assertNotIn("analysis_path", analysis)


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from streamlit.testing.v1 import AppTest

        cls.AppTest = AppTest
        cls.app_path = str(PROJECT_DIR / "app.py")

    def test_dashboard_retains_company_pages_and_search_input(self):
        at = self.AppTest.from_file(self.app_path, default_timeout=60).run()

        self.assertFalse(at.exception)
        self.assertEqual(
            list(at.radio(key="dashboard_navigation").options),
            ["Dashboard Overview", "Company ESG Analysis"],
        )
        self.assertEqual(len(at.get("progress")), 1)
        self.assertEqual(at.get("progress")[0].proto.value, 0)
        self.assertEqual(len(at.dataframe), 1)
        self.assertTrue(
            any(
                "COMPANY ESG ANALYSIS PROGRESS" in item.value
                for item in at.markdown
            )
        )

        at.radio(key="dashboard_navigation").set_value(
            "Company ESG Analysis"
        ).run()
        self.assertFalse(at.exception)
        self.assertTrue(
            any(item.label == "Company Name" for item in at.text_input)
        )
        self.assertEqual(len(at.get("progress")), 1)
        at.button(key="company_search_submit").click().run()
        self.assertTrue(
            any(
                "Enter a company name before starting" in item.value
                for item in at.error
            )
        )
        self.assertEqual(
            at.session_state["company_progress_status"]["status"],
            "WAITING",
        )

    def test_dashboard_progress_tracks_company_workflow(self):
        at = self.AppTest.from_file(self.app_path, default_timeout=60).run()
        company_state = at.session_state["company_progress_status"]
        company_state["status"] = "RUNNING"
        company_state["finished"] = False
        applicable_stages = company_state["applicable_stages"]
        company_state["stages"][applicable_stages[0]]["status"] = "COMPLETED"
        company_state["stages"][applicable_stages[1]].update(
            status="RUNNING",
            progress=0.5,
        )

        at.run()

        expected_progress = 1.5 / len(applicable_stages)
        self.assertFalse(at.exception)
        progress = at.get("progress")
        self.assertEqual(len(progress), 1)
        self.assertEqual(
            progress[0].proto.value,
            int(expected_progress * 100),
        )
        self.assertIn(f"{expected_progress:.0%}", progress[0].proto.text)
        self.assertLess(progress[0].proto.value, 100)
        statuses = at.dataframe[0].value.set_index("Stage")["Status"]
        self.assertEqual(statuses["Web Search"], "COMPLETED")
        self.assertEqual(statuses["Web Data Collection"], "RUNNING")

    def test_partial_company_workflow_never_renders_as_complete(self):
        import app

        state = app.create_progress_state(app.COMPANY_STAGES)
        state["finished"] = True
        state["status"] = "PARTIAL"
        for stage in state["applicable_stages"]:
            state["stages"][stage]["status"] = "PARTIAL"

        self.assertLess(app.calculate_progress(state), 1.0)
        self.assertEqual(app.calculate_progress(state), 0.99)

    def test_unavailable_stages_make_company_workflow_partial(self):
        import app

        state = app.create_progress_state(app.COMPANY_STAGES)
        for stage in state["applicable_stages"]:
            state["stages"][stage]["status"] = "COMPLETED"
        for stage in ("NLI Analysis", "Risk Prediction", "Risk Explanation"):
            state["stages"][stage]["status"] = "NOT AVAILABLE"
        task = {
            "lock": threading.RLock(),
            "state": state,
            "result": None,
        }
        analysis = {
            "errors": {},
            "risk_error": None,
            "explainability_error": None,
            "timeline_error": None,
        }

        with (
            patch.object(app, "build_analysis", return_value=analysis),
            patch.object(app, "persist_analysis"),
        ):
            app.company_analysis_worker(task, "Acme")

        self.assertEqual(task["state"]["status"], "PARTIAL")
        self.assertIn("Limitations", task["state"]["message"])

    def test_company_background_workflow_survives_navigation_without_duplicates(self):
        import app

        session_id = f"offline-{id(self)}"
        session_state = {
            "_workflow_session_id": session_id,
            "dashboard_navigation": "Company ESG Analysis",
            "company_progress_status": app.create_progress_state(
                app.COMPANY_STAGES
            ),
            "active_analysis": None,
        }
        started = threading.Event()
        release = threading.Event()
        calls = []

        def worker(task, company_name):
            calls.append(company_name)
            app.publish_background_stage(
                task,
                "Web Search",
                "RUNNING",
                "Mock search is in progress.",
                progress=0.4,
            )
            app.save_background_result(
                task,
                {"company": company_name, "partial": True},
            )
            started.set()
            if not release.wait(5):
                raise TimeoutError("test did not release company task")
            app.publish_background_stage(
                task,
                "Web Search",
                "COMPLETED",
                "Mock search completed.",
            )
            app.save_background_result(
                task,
                {"company": company_name, "partial": False},
            )
            with task["lock"]:
                app.finish_progress_state(
                    task["state"],
                    "COMPLETED",
                    app.COMPANY_SUCCESS_MESSAGE,
                )

        try:
            with patch.object(app.st, "session_state", session_state):
                self.assertTrue(
                    app.schedule_background_workflow(
                        "company",
                        app.COMPANY_STAGES,
                        worker,
                        "Offline Company",
                    )
                )
                self.assertTrue(started.wait(5))
                self.assertFalse(
                    app.schedule_background_workflow(
                        "company",
                        app.COMPANY_STAGES,
                        worker,
                        "Offline Company",
                    )
                )
                task = app._WORKFLOW_TASKS[
                    session_state["_company_task_id"]
                ]
                session_state["dashboard_navigation"] = "Dashboard Overview"
                app.sync_background_workflow_state()
                self.assertEqual(
                    session_state["company_progress_status"]["status"],
                    "RUNNING",
                )
                self.assertEqual(
                    session_state["active_analysis"],
                    {"company": "Offline Company", "partial": True},
                )
                release.set()
                task["future"].result(timeout=5)
                app.sync_background_workflow_state()
                self.assertEqual(
                    session_state["company_progress_status"]["status"],
                    "COMPLETED",
                )
                self.assertEqual(
                    session_state["active_analysis"],
                    {"company": "Offline Company", "partial": False},
                )
                self.assertEqual(calls, ["Offline Company"])
        finally:
            release.set()
            with app._WORKFLOW_TASKS_LOCK:
                registry_key = (session_id, "company")
                task_id = app._WORKFLOW_TASKS_BY_SESSION.pop(
                    registry_key,
                    None,
                )
                if task_id:
                    app._WORKFLOW_TASKS.pop(task_id, None)

    def test_new_company_search_supersedes_running_search_for_display(self):
        import app

        session_id = f"offline-switch-{id(self)}"
        session_state = {
            "_workflow_session_id": session_id,
            "dashboard_navigation": "Company ESG Analysis",
            "company_progress_status": app.create_progress_state(
                app.COMPANY_STAGES
            ),
            "active_analysis": None,
        }
        first_started = threading.Event()
        release_first = threading.Event()

        def first_worker(task, company_name):
            first_started.set()
            if not release_first.wait(5):
                raise TimeoutError("test did not release first search")
            app.save_background_result(task, {"company": company_name})

        def second_worker(task, company_name):
            app.save_background_result(task, {"company": company_name})

        try:
            with patch.object(app.st, "session_state", session_state):
                self.assertTrue(
                    app.schedule_background_workflow(
                        "company",
                        app.COMPANY_STAGES,
                        first_worker,
                        "First Company",
                    )
                )
                self.assertTrue(first_started.wait(5))
                first_task = app._WORKFLOW_TASKS[
                    session_state["_company_task_id"]
                ]
                self.assertTrue(
                    app.schedule_background_workflow(
                        "company",
                        app.COMPANY_STAGES,
                        second_worker,
                        "Second Company",
                    )
                )
                second_task_id = session_state["_company_task_id"]
                self.assertNotEqual(second_task_id, first_task["id"])
                second_task = app._WORKFLOW_TASKS[second_task_id]
                second_task["future"].result(timeout=5)
                app.sync_background_workflow_state()
                self.assertEqual(
                    session_state["active_analysis"],
                    {"company": "Second Company"},
                )
                release_first.set()
                first_task["future"].result(timeout=5)
                app.sync_background_workflow_state()
                self.assertEqual(
                    session_state["active_analysis"],
                    {"company": "Second Company"},
                )
        finally:
            release_first.set()
            with app._WORKFLOW_TASKS_LOCK:
                registry_key = (session_id, "company")
                task_id = app._WORKFLOW_TASKS_BY_SESSION.pop(
                    registry_key,
                    None,
                )
                if task_id:
                    app._WORKFLOW_TASKS.pop(task_id, None)

    def test_synthetic_saved_analysis_fixture_matches_pipeline_schema(self):
        # Guards the committed fixture against drift from the keys that
        # run_company_search writes (app.py analysis constructor).
        fixture = json.loads(
            SYNTHETIC_SAVED_ANALYSIS_PATH.read_text(encoding="utf-8")
        )
        pipeline_keys = {
            "company", "created_at", "collection", "relevant_document_count",
            "company_claims", "external_evidence", "all_statements",
            "timeline_records", "timeline_error", "matching_results",
            "nli_results", "features", "risk_error", "risk_explanations",
            "explainability_error", "errors",
        }
        self.assertTrue(
            pipeline_keys.issubset(fixture),
            f"Fixture is missing keys: {sorted(pipeline_keys - set(fixture))}",
        )
        self.assertEqual(fixture["company"], "Example Corp")

    def test_company_errors_render_on_retained_analysis_page(self):
        fixture_path = SYNTHETIC_SAVED_ANALYSIS_PATH
        saved_analysis = json.loads(
            fixture_path.read_text(encoding="utf-8")
        )
        saved_analysis["errors"] = {"Mocked company stage": "offline"}

        at = self.AppTest.from_file(self.app_path, default_timeout=90)
        at.session_state["active_analysis"] = saved_analysis
        at.session_state["dashboard_navigation"] = "Company ESG Analysis"
        at.run()

        self.assertFalse(at.exception)
        self.assertTrue(
            any("Mocked company stage failed" in item.value for item in at.error)
        )
        self.assertEqual(
            list(at.radio(key="dashboard_navigation").options),
            ["Dashboard Overview", "Company ESG Analysis"],
        )

    def test_workflow_page_renderer_is_periodically_refreshed(self):
        import app

        self.assertEqual(app.WORKFLOW_UI_REFRESH_SECONDS, 1)
        self.assertTrue(
            hasattr(app.render_selected_page, "__wrapped__"),
            "The page renderer must remain a Streamlit fragment.",
        )



    def test_saved_analysis_does_not_override_dashboard_navigation(self):
        at = self.AppTest.from_file(self.app_path, default_timeout=60).run()
        at.session_state["active_analysis"] = {"company": "Saved company"}
        at.run()

        self.assertFalse(at.exception)
        self.assertEqual(
            at.radio(key="dashboard_navigation").value,
            "Dashboard Overview",
        )
        self.assertEqual(len(at.text_input), 0)
        self.assertEqual(len(at.get("progress")), 1)

    def test_saved_company_analysis_renders_without_reprocessing(self):
        fixture_path = SYNTHETIC_SAVED_ANALYSIS_PATH
        self.assertTrue(fixture_path.is_file())
        saved_analysis = json.loads(
            fixture_path.read_text(encoding="utf-8")
        )
        self.assertTrue(saved_analysis["company_claims"])

        at = self.AppTest.from_file(self.app_path, default_timeout=90).run()
        at.session_state["active_analysis"] = saved_analysis
        at.session_state["dashboard_navigation"] = "Company ESG Analysis"
        at.run()

        self.assertFalse(at.exception)
        self.assertTrue(
            any(
                "COMPANY ENVIRONMENTAL TIMELINE" in item.value
                for item in at.markdown
            )
        )
        self.assertTrue(
            any(
                "ENVIRONMENTAL CLAIMS" in item.value
                for item in at.markdown
            )
        )




    def test_latest_collection_and_statement_files_prefer_active_company(self):
        import json
        import os
        import tempfile
        from pathlib import Path

        import analyze_collected_evidence
        import online_evidence_matcher

        original_collection_dir = analyze_collected_evidence.COLLECTED_DIR
        original_output_dir = online_evidence_matcher.OUTPUT_DIR

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                collection_dir = tmp_path / "data" / "collected_documents"
                collection_dir.mkdir(parents=True)
                output_dir = tmp_path / "output"
                output_dir.mkdir()

                beta_collection = collection_dir / "beta_collection.json"
                beta_collection.write_text(
                    json.dumps({"search_name": "Beta", "documents": []}),
                    encoding="utf-8",
                )
                alpha_collection = collection_dir / "alpha_collection.json"
                alpha_collection.write_text(
                    json.dumps({"search_name": "Alpha", "documents": []}),
                    encoding="utf-8",
                )
                os.utime(beta_collection, (1000, 2000))
                os.utime(alpha_collection, (1000, 1000))

                analyze_collected_evidence.COLLECTED_DIR = collection_dir
                self.assertEqual(
                    analyze_collected_evidence.load_latest_collection("Alpha")["search_name"],
                    "Alpha",
                )
                with self.assertRaisesRegex(FileNotFoundError, "Unknown"):
                    analyze_collected_evidence.load_latest_collection("Unknown")

                beta_statements = output_dir / "beta_statements.json"
                beta_statements.write_text(
                    json.dumps({"company": "Beta", "statements": []}),
                    encoding="utf-8",
                )
                alpha_statements = output_dir / "alpha_statements.json"
                alpha_statements.write_text(
                    json.dumps({"company": "Alpha", "statements": []}),
                    encoding="utf-8",
                )
                os.utime(beta_statements, (1000, 2000))
                os.utime(alpha_statements, (1000, 1000))

                online_evidence_matcher.OUTPUT_DIR = output_dir
                self.assertEqual(
                    online_evidence_matcher.load_latest_statements("Alpha")["company"],
                    "Alpha",
                )
                with self.assertRaisesRegex(FileNotFoundError, "Unknown"):
                    online_evidence_matcher.load_latest_statements("Unknown")
        finally:
            analyze_collected_evidence.COLLECTED_DIR = original_collection_dir
            online_evidence_matcher.OUTPUT_DIR = original_output_dir

    def test_nli_classifier_uses_cached_model_without_reloading(self):
        import contradiction_detector

        original_classifier = contradiction_detector._NLI_CLASSIFIER
        cached = object()
        contradiction_detector._NLI_CLASSIFIER = cached

        def fake_pipeline(*args, **kwargs):
            raise AssertionError("Pipeline should not reload the cached model.")

        try:
            with patch.object(
                contradiction_detector.transformers,
                "pipeline",
                new=fake_pipeline,
            ):
                self.assertIs(
                    contradiction_detector._get_nli_classifier(),
                    cached,
                )
        finally:
            contradiction_detector._NLI_CLASSIFIER = original_classifier


if __name__ == "__main__":
    unittest.main()
