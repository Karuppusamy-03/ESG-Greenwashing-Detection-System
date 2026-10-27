"""Opt-in checks against the user's PRIVATE local data.

These tests read files from the gitignored data/ and output/ folders of a
real working copy. They are NOT part of the offline suite and are never run
by default. Enable them explicitly:

    $env:ESG_LOCAL_DATA = "1"          # PowerShell
    python -m unittest discover -s local_checks -p "test_*.py" -v

When enabled, a missing file FAILS the test; it is never silently passed.
When not enabled, the tests are reported as skipped with the reason.
"""

import json
import os
import unittest
from pathlib import Path

from online_claim_extractor import extract_environmental_statements

PROJECT_DIR = Path(__file__).resolve().parents[1]
COLLECTED_DIR = PROJECT_DIR / "data" / "collected_documents"

LOCAL_DATA_ENABLED = os.environ.get("ESG_LOCAL_DATA") == "1"
SKIP_REASON = (
    "Local-data check: set ESG_LOCAL_DATA=1 to run against this machine's "
    "data/ and output/ folders."
)


@unittest.skipUnless(LOCAL_DATA_ENABLED, SKIP_REASON)
class LocalSavedDataTests(unittest.TestCase):

    def test_saved_apple_and_tesla_collections_are_read_only_fixtures(self):
        import app

        fixtures = {
            "Apple": COLLECTED_DIR / "apple_20261011_024917.json",
            "Tesla": COLLECTED_DIR / "tesla_20261010_182212.json",
        }

        for company, path in fixtures.items():
            with self.subTest(company=company):
                self.assertTrue(path.is_file(), f"Missing saved fixture: {path}")
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
                    for statement in extract_environmental_statements(
                        document
                    )
                ]

                self.assertGreater(len(documents), 0)
                self.assertGreater(len(successful), 0)
                self.assertGreater(len(statements), 0)
                self.assertTrue(
                    all(statement.get("source_url") for statement in statements)
                )
                company_claims, external, _ = app.collect_candidate_statements(
                    company,
                    successful,
                )
                if company == "Tesla":
                    self.assertEqual(
                        collection.get("search_name", "").casefold(),
                        "tesla",
                    )
                    self.assertEqual(company_claims, [])
                    self.assertTrue(external)
                else:
                    self.assertTrue(company_claims)
                    self.assertEqual(
                        external,
                        [],
                        "Saved Apple collection contains no relevant "
                        "independent external statements.",
                    )
                    self.assertTrue(
                        all(
                            statement["source_role"] == "company_source"
                            for statement in company_claims
                        )
                    )
                    self.assertTrue(
                        all(
                            statement["source_role"] != "company_source"
                            for statement in external
                        )
                    )
