"""Explicitly enabled, bounded live company-workflow smoke test."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


LIVE_ENABLED = os.environ.get("ESG_LIVE_TESTS") == "1"


@unittest.skipUnless(
    LIVE_ENABLED,
    "Set ESG_LIVE_TESTS=1 (run the test script with -Live) to enable live checks.",
)
class LiveWorkflowTests(unittest.TestCase):
    def test_company_web_workflow(self):
        import app
        import automatic_collector
        import web_search

        search_function = web_search.search_environmental_information
        search_log = []

        def bounded_search(name, **kwargs):
            results = search_function(
                name=name,
                max_results_per_query=1,
                delay_seconds=0,
                diagnostics=kwargs.get("diagnostics"),
            )
            search_log.extend(
                result.get("url", "")
                for result in results
                if result.get("url")
            )
            return results

        with tempfile.TemporaryDirectory(
            prefix="esg-live-company-"
        ) as temporary_directory:
            with (
                patch.object(
                    web_search,
                    "SEARCH_QUERIES",
                    {"company": web_search.SEARCH_QUERIES["company"][:3]},
                ),
                patch.object(
                    automatic_collector,
                    "search_environmental_information",
                    side_effect=bounded_search,
                ),
                patch.object(
                    automatic_collector,
                    "OUTPUT_DIR",
                    Path(temporary_directory),
                ),
            ):
                collection = automatic_collector.collect_company_esg_information(
                    "Apple",
                    max_documents=2,
                )

        self.assertEqual(collection["company"], "Apple")
        self.assertLessEqual(len(collection["documents"]), 2)
        self.assertIsInstance(collection.get("diagnostics"), dict)
        relevant_documents = [
            document
            for document in collection["documents"]
            if (
                document.get("status") == "success"
                and str(document.get("text") or "").strip()
                and app.document_mentions_company("Apple", document)
            )
        ]
        claims, external, statements = app.collect_candidate_statements(
            "Apple",
            relevant_documents,
        )
        self.assertGreaterEqual(len(statements), len(claims) + len(external))
        self.assertTrue(
            all(
                document.get("source_url") or document.get("final_url")
                for document in collection["documents"]
            )
        )
        if not search_log:
            self.skipTest(
                "No search result URLs were returned. Internet or "
                "search-provider availability may be the cause."
            )
