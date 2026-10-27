"""Deterministic regression tests for reporting-year selection in evidence queries.

Target years, baseline years and comparison years must never be used as the
reporting period in generated evidence searches. These tests use no network.
"""

import unittest

import web_search


class ClaimReportingYearTests(unittest.TestCase):
    def test_target_year_is_not_used_as_reporting_year(self):
        self.assertIsNone(web_search.claim_reporting_year(
            "Acme aims to reach 100% renewable electricity by 2030."
        ))

    def test_reporting_year_is_kept_when_target_year_is_present(self):
        self.assertEqual(
            web_search.claim_reporting_year(
                "Acme targets net zero by 2040 and reported 45% "
                "renewable electricity in 2023."
            ),
            "2023",
        )

    def test_comparison_year_is_excluded(self):
        self.assertEqual(
            web_search.claim_reporting_year(
                "Acme cut Scope 1 emissions 20% in 2024 compared with 2019."
            ),
            "2024",
        )

    def test_baseline_phrase_year_is_excluded(self):
        self.assertEqual(
            web_search.claim_reporting_year(
                "Acme reduced Scope 2 emissions by 30% from its 2020 "
                "baseline in 2024."
            ),
            "2024",
        )

    def test_labelled_baseline_year_is_excluded(self):
        self.assertEqual(
            web_search.claim_reporting_year(
                "Acme cut emissions 10% (2019 baseline: 100kt) and "
                "reported 20% in 2024."
            ),
            "2024",
        )

    def test_ambiguous_multiple_reporting_years_return_no_year(self):
        self.assertIsNone(web_search.claim_reporting_year(
            "Acme reported 20% renewable use in 2022 and 25% in 2024."
        ))

    def test_claim_without_year_returns_none(self):
        self.assertIsNone(web_search.claim_reporting_year(
            "Acme uses recycled packaging materials."
        ))


class ClaimEvidenceQueryYearTests(unittest.TestCase):
    def _queries(self, statement):
        return web_search.build_claim_evidence_queries(
            "Acme Corp",
            [{"statement": statement}],
            max_queries=3,
        )

    def test_target_year_absent_from_generated_queries(self):
        for query in self._queries(
            "Acme will cut emissions 50% by 2030 versus 2019 levels."
        ):
            self.assertNotIn("2030", query)
            self.assertNotIn("2019", query)

    def test_reporting_year_used_for_baseline_claim(self):
        queries = self._queries(
            "Acme reduced Scope 2 emissions by 30% from its 2020 "
            "baseline in 2024."
        )
        self.assertTrue(queries)
        for query in queries:
            self.assertIn("2024", query)
            self.assertNotIn("2020", query)


if __name__ == "__main__":
    unittest.main()


class SourceClassificationUnseenCompanyTests(unittest.TestCase):
    """Unseen company names must not turn government hosts into company sources."""

    def test_government_host_named_after_company_is_not_company_source(self):
        label = web_search.classify_source("https://www.acme.gov/report", "Acme")
        self.assertNotIn("company source", label.lower())
        self.assertEqual(label, "Government source / regulatory status unverified")

    def test_configured_government_host_is_not_company_source(self):
        label = web_search.classify_source("https://epa.gov/acme", "EPA")
        self.assertEqual(label, "Government / regulatory source")

    def test_company_domain_name_match_still_recognised(self):
        self.assertEqual(
            web_search.classify_source("https://www.acme.com/esg", "Acme Corporation"),
            "Potential company source (domain-name match)",
        )
