"""Deterministic tests for arbitrary-company identity handling.

Covers name-derived aliases, evidence-based official-domain discovery,
ambiguous-name disambiguation, subsidiaries and similar names. All search
calls are mocked; nothing here touches the network.
"""

import unittest
from unittest.mock import patch

import web_search


def result(url, title, snippet="", domain=None):
    return {
        "url": url,
        "title": title,
        "snippet": snippet,
        "domain": domain or web_search.get_domain(url),
    }


class DerivedAliasTests(unittest.TestCase):
    def test_compact_and_initialism_aliases_are_derived_from_name(self):
        aliases = web_search.derive_company_aliases("Procter & Gamble")
        self.assertIn("proctergamble", aliases)
        self.assertIn("pg", aliases)

    def test_initialism_for_long_multiword_name(self):
        self.assertIn(
            "ibm",
            web_search.derive_company_aliases("International Business Machines"),
        )

    def test_legal_suffixes_are_not_part_of_aliases(self):
        self.assertEqual(
            web_search.derive_company_aliases("Acme Corporation Ltd."),
            ["acme"],
        )

    def test_single_word_name_has_no_initialism(self):
        self.assertEqual(web_search.derive_company_aliases("Tesla"), ["tesla"])

    def test_empty_name_returns_no_aliases(self):
        self.assertEqual(web_search.derive_company_aliases("  "), [])


class AmbiguousNameTests(unittest.TestCase):
    def test_delta_air_lines_is_not_accepted_as_bare_delta(self):
        self.assertTrue(web_search.ambiguous_name_continuation(
            "Delta", "Delta Air Lines cuts emissions"
        ))
        self.assertFalse(web_search.search_result_mentions_entity(
            "Delta", "company", "Delta Air Lines cuts emissions"
        ))

    def test_delta_dental_is_not_accepted_as_bare_delta(self):
        self.assertFalse(web_search.search_result_mentions_entity(
            "Delta", "company", "Delta Dental sustainability report"
        ))

    def test_bare_name_usage_is_accepted(self):
        self.assertTrue(web_search.search_result_mentions_entity(
            "Delta", "company", "Delta reports a 20% emissions cut"
        ))

    def test_corporate_descriptor_continuation_is_accepted(self):
        self.assertTrue(web_search.search_result_mentions_entity(
            "Samsung", "company", "Samsung Electronics emissions report"
        ))

    def test_subsidiary_cue_in_same_sentence_is_accepted(self):
        self.assertFalse(web_search.ambiguous_name_continuation(
            "Acme", "Acme Holdings is a subsidiary reporting emissions"
        ))

    def test_similar_name_is_not_a_match(self):
        self.assertFalse(web_search.search_result_mentions_entity(
            "Acme", "company", "Acmeware emissions report"
        ))

    def test_esg_headings_after_name_are_accepted(self):
        self.assertTrue(web_search.search_result_mentions_entity(
            "Acme", "company", "Acme Sustainability Report 2024"
        ))
        self.assertTrue(web_search.search_result_mentions_entity(
            "Acme", "company", "Acme Independent Environmental Review"
        ))

    def test_lowercase_continuation_is_not_ambiguous(self):
        self.assertFalse(web_search.ambiguous_name_continuation(
            "Acme", "Acme emissions fell 10% in 2024"
        ))


class OfficialDomainDiscoveryTests(unittest.TestCase):
    def test_domain_matching_derived_alias_is_discovered(self):
        results = [
            result(
                "https://pg.com/esg",
                "Procter & Gamble ESG report",
                "Procter & Gamble emissions",
            ),
            result(
                "https://pg.com/esg2",
                "Procter & Gamble carbon",
                "Procter & Gamble carbon data",
            ),
        ]
        discovered = web_search.discover_official_domains(
            "Procter & Gamble", results
        )
        self.assertEqual([item["domain"] for item in discovered], ["pg.com"])
        self.assertEqual(discovered[0]["verification_status"], "candidate_unverified")
        self.assertEqual(len(discovered[0]["supporting_urls"]), 2)

    def test_discovery_is_not_hardcoded_to_known_companies(self):
        results = [result(
            "https://ibm.com/sustainability",
            "International Business Machines emissions report",
            "International Business Machines reports emissions",
        )]
        domains = [
            item["domain"]
            for item in web_search.discover_official_domains(
                "International Business Machines", results
            )
        ]
        self.assertEqual(domains, ["ibm.com"])

    def test_aggregator_host_is_not_discovered(self):
        results = [result(
            "https://www.wikipedia.org/wiki/Medium",
            "Medium company emissions",
            "Medium company emissions",
        )]
        self.assertEqual(
            web_search.discover_official_domains("Medium", results), []
        )

    def test_government_host_is_not_discovered(self):
        results = [result(
            "https://www.acme.gov/report",
            "Acme emissions report",
            "Acme emissions report",
        )]
        self.assertEqual(web_search.discover_official_domains("Acme", results), [])

    def test_domain_without_company_mention_is_not_discovered(self):
        results = [result(
            "https://acme.example/home",
            "Welcome to our homepage",
            "We sell widgets",
        )]
        self.assertEqual(web_search.discover_official_domains("Acme", results), [])

    def test_ambiguous_result_does_not_create_candidate(self):
        results = [result(
            "https://delta.example/air",
            "Delta Air Lines emissions",
            "Delta Air Lines emissions",
        )]
        self.assertEqual(web_search.discover_official_domains("Delta", results), [])

    def test_discovered_domain_gets_own_label_not_official(self):
        label = web_search.classify_source(
            "https://pg.com/esg",
            "Procter & Gamble",
            discovered_domains=["pg.com"],
        )
        self.assertEqual(label, "Potential company source (discovered domain)")
        self.assertNotEqual(label, "Official company source")

    def test_unrelated_domain_is_not_labelled_company_source(self):
        self.assertEqual(
            web_search.classify_source(
                "https://news.example/pg",
                "Procter & Gamble",
                discovered_domains=["pg.com"],
            ),
            "Other / unverified",
        )


class SearchFlowDiscoveryTests(unittest.TestCase):
    """End-to-end search flow with a mocked search client."""

    def test_discovery_issues_site_queries_and_reclassifies_results(self):
        queries_seen = []

        class MockDDGS:
            def __init__(self, timeout):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def text(self, query, max_results, **kwargs):
                queries_seen.append(query)
                if query.startswith("site:pg.com"):
                    return [{
                        "title": "Procter & Gamble ESG hub",
                        "href": "https://pg.com/esg-hub",
                        "body": "Procter & Gamble emissions data",
                    }]
                return [{
                    "title": "Procter & Gamble emissions review",
                    "href": "https://pg.com/esg",
                    "body": "Procter & Gamble emissions and carbon",
                }]

        diagnostics = {}
        with patch.object(web_search, "DDGS", MockDDGS):
            results = web_search.search_environmental_information(
                "Procter & Gamble",
                search_type="company",
                max_results_per_query=2,
                delay_seconds=0,
                diagnostics=diagnostics,
            )

        self.assertTrue(
            any(query.startswith("site:pg.com") for query in queries_seen),
            "discovered domain should trigger a site-restricted query",
        )
        discovery = diagnostics["official_domain_discovery"]
        self.assertEqual(
            [item["domain"] for item in discovery["discovered_domains"]],
            ["pg.com"],
        )
        self.assertIn("pg", diagnostics["name_derived_aliases"])
        by_url = {item["url"]: item for item in results}
        self.assertEqual(
            by_url["https://pg.com/esg-hub"]["source_type"],
            "Potential company source (discovered domain)",
        )
        candidate = next(
            item for item in diagnostics["candidate_results"]
            if item["url"] == "https://pg.com/esg-hub"
        )
        self.assertTrue(
            any(
                origin["query"].startswith("site:pg.com")
                for origin in candidate["queries"]
            ),
            "candidate must keep the site query that found it (provenance)",
        )

    def test_no_discovery_queries_when_no_candidate_domain(self):
        queries_seen = []

        class MockDDGS:
            def __init__(self, timeout):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def text(self, query, max_results, **kwargs):
                queries_seen.append(query)
                return [{
                    "title": "Delta Air Lines emissions",
                    "href": "https://delta-news.example/a",
                    "body": "Delta Air Lines emissions",
                }]

        diagnostics = {}
        with patch.object(web_search, "DDGS", MockDDGS):
            web_search.search_environmental_information(
                "Delta",
                search_type="company",
                max_results_per_query=1,
                delay_seconds=0,
                diagnostics=diagnostics,
            )
        self.assertFalse(any(q.startswith("site:") and "delta" in q for q in queries_seen
                             if "sec.gov" not in q and "epa.gov" not in q))
        self.assertEqual(
            diagnostics["official_domain_discovery"]["discovery_query_count"], 0
        )


if __name__ == "__main__":
    unittest.main()


class PageRelevanceDisambiguationTests(unittest.TestCase):
    """The page-level company check must apply the same disambiguation."""

    def test_page_about_different_company_is_not_relevant_to_bare_name(self):
        import app

        document = {
            "title": "Delta Air Lines emissions",
            "snippet": "",
            "text": "Delta Air Lines reported emissions and carbon cuts in 2024.",
        }
        self.assertFalse(app.document_mentions_company("Delta", document))

    def test_page_about_bare_name_is_relevant(self):
        import app

        document = {
            "title": "Delta reports emissions",
            "snippet": "",
            "text": "Delta reported emissions and carbon cuts in 2024.",
        }
        self.assertTrue(app.document_mentions_company("Delta", document))
