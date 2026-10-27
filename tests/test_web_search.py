import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class WebSearchTests(unittest.TestCase):
    def test_samsung_electronics_alias_keeps_official_report_as_company_source(self):
        import web_search
        from online_claim_extractor import identify_source_role

        report_url = (
            "https://www.samsung.com/global/sustainability/media/pdf/"
            "Samsung_Electronics_Sustainability_Report_2023_ENG.pdf"
        )
        source_type = web_search.classify_source(
            report_url,
            "Samsung Electronics",
        )

        self.assertEqual(source_type, "Official company source")
        self.assertEqual(
            identify_source_role({
                "source_type": source_type,
                "source_url": report_url,
                "source_domain": "www.samsung.com",
            }),
            "company_source",
        )

    def test_samsung_semiconductor_newsroom_is_company_source(self):
        import web_search
        from online_claim_extractor import identify_source_role

        source_url = (
            "https://news.samsungsemiconductor.com/global/"
            "samsung-electronics-announces-new-environmental-strategy/"
        )
        source_type = web_search.classify_source(
            source_url,
            "Samsung",
        )

        self.assertEqual(source_type, "Official company source")
        self.assertEqual(
            identify_source_role({
                "source_type": source_type,
                "source_url": source_url,
                "source_domain": "news.samsungsemiconductor.com",
            }),
            "company_source",
        )

    def test_url_validation_handles_blank_non_text_and_credential_urls(self):
        import web_search

        for url in (None, "", " ", 7, "https://user:pass@example.com/report"):
            with self.subTest(url=url):
                self.assertFalse(web_search.is_valid_web_url(url))
        for url in (
            "https://example.com/report",
            "http://example.com:8080/report",
        ):
            with self.subTest(url=url):
                self.assertTrue(web_search.is_valid_web_url(url))
        self.assertEqual(web_search.get_domain(None), "")
        self.assertEqual(web_search.get_domain(""), "")

    def test_search_skips_malformed_items_and_search_provider_redirects(self):
        import web_search

        class MalformedDDGS:
            def __init__(self, timeout):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def text(self, query, max_results):
                return [
                    None,
                    {"href": None, "title": "Missing URL", "body": ""},
                    {"href": "", "title": "", "body": ""},
                    {
                        "href": "https://www.bing.com/aclick?ld=tracking",
                        "title": "Ad redirect",
                        "body": "Environmental reporting",
                    },
                    {
                        "href": "https://research.example/acme-report",
                        "title": "Acme environmental report",
                        "body": "Independent emissions assessment for Acme.",
                    },
                ]

        diagnostics = {}
        with patch.object(web_search, "DDGS", MalformedDDGS):
            results = web_search.search_web(
                "Acme emissions",
                company_name="Acme",
                diagnostics=diagnostics,
            )

        self.assertEqual(
            [item["url"] for item in results],
            ["https://research.example/acme-report"],
        )
        self.assertEqual(diagnostics["raw_result_count"], 5)
        self.assertEqual(diagnostics["accepted_result_count"], 1)
        self.assertEqual(diagnostics["rejected_result_count"], 4)
        self.assertEqual(
            [item["reason"] for item in diagnostics["rejected_results"]],
            [
                "invalid_search_result",
                "invalid_or_unsupported_url",
                "provider_empty_result",
                "search_provider_redirect",
            ],
        )

    def test_search_diagnostics_distinguish_results_empty_and_failure(self):
        import web_search

        class ResultsDDGS:
            def __init__(self, timeout):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def text(self, query, max_results):
                return [
                    {
                        "title": "Acme environmental report",
                        "href": "https://research.example/acme-report",
                        "body": "Independent emissions assessment for Acme.",
                    },
                    {
                        "title": "Invalid",
                        "href": "javascript:alert(1)",
                        "body": "Ignored",
                    },
                ]

        with patch.object(web_search, "DDGS", ResultsDDGS):
            diagnostics = {}
            results = web_search.search_web(
                "Acme emissions",
                company_name="Acme",
                diagnostics=diagnostics,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(diagnostics["status"], "completed")
        self.assertEqual(diagnostics["raw_result_count"], 2)
        self.assertEqual(diagnostics["accepted_result_count"], 1)
        self.assertEqual(diagnostics["rejected_result_count"], 1)
        self.assertEqual(
            diagnostics["rejected_results"][0]["url"],
            "javascript:alert(1)",
        )
        self.assertEqual(
            diagnostics["rejected_results"][0]["reason"],
            "invalid_or_unsupported_url",
        )

        class EmptyDDGS(ResultsDDGS):
            def text(self, query, max_results):
                return []

        with patch.object(web_search, "DDGS", EmptyDDGS):
            empty_diagnostics = {}
            web_search.search_web(
                "Acme emissions",
                diagnostics=empty_diagnostics,
            )

        self.assertEqual(empty_diagnostics["status"], "no_results")
        self.assertIsNone(empty_diagnostics["error"])

        class FilteredDDGS(ResultsDDGS):
            def text(self, query, max_results):
                return [{
                    "title": "Unrelated result",
                    "href": "https://research.example/unrelated",
                    "body": "Unrelated company and topic.",
                }]

        with patch.object(web_search, "DDGS", FilteredDDGS):
            filtered_diagnostics = {}
            web_search.search_web(
                "Acme emissions",
                company_name="Acme",
                diagnostics=filtered_diagnostics,
            )

        self.assertEqual(filtered_diagnostics["status"], "filtered_out")
        self.assertEqual(filtered_diagnostics["raw_result_count"], 1)
        self.assertEqual(filtered_diagnostics["accepted_result_count"], 0)
        self.assertEqual(
            filtered_diagnostics["rejected_results"][0]["url"],
            "https://research.example/unrelated",
        )
        self.assertEqual(
            filtered_diagnostics["rejected_results"][0]["reason"],
            "no_environmental_relevance_cue",
        )

        with patch.object(
            web_search,
            "DDGS",
            side_effect=RuntimeError("offline"),
        ):
            failed_diagnostics = {}
            web_search.search_web(
                "Acme emissions",
                diagnostics=failed_diagnostics,
            )

        self.assertEqual(failed_diagnostics["status"], "failed")
        self.assertEqual(failed_diagnostics["error"], "offline")

    def test_duplicate_search_results_retain_best_metadata_and_date(self):
        import web_search

        class DuplicateDDGS:
            def __init__(self, timeout):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def text(self, query, max_results):
                return [
                    {
                        "title": "Acme report",
                        "href": "https://research.example/acme-report",
                        "body": "Acme environmental report.",
                    },
                    {
                        "title": "Independent Acme Environmental Assessment",
                        "href": "https://research.example/acme-report",
                        "body": (
                            "Independent assessment of Acme emissions in "
                            "2024."
                        ),
                        "date": "2025-02-10",
                    },
                ]

        with patch.object(web_search, "DDGS", DuplicateDDGS):
            results = web_search.search_web(
                "Acme environmental assessment",
                company_name="Acme",
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["title"],
            "Independent Acme Environmental Assessment",
        )
        self.assertEqual(
            results[0]["snippet"],
            "Independent assessment of Acme emissions in 2024.",
        )
        self.assertEqual(results[0]["search_result_date"], "2025-02-10")
        import automatic_collector

        document = automatic_collector.create_document(
            results[0]["url"],
            results[0],
        )
        self.assertEqual(document["search_result_date"], "2025-02-10")

    def test_search_web_uses_requested_ddgs_backend(self):
        import web_search

        class BackendDDGS:
            def __init__(self, timeout):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def text(self, query, max_results, backend):
                self.backend = backend
                return [{
                    "title": "Acme independent emissions review",
                    "href": "https://research.example/acme-review",
                    "body": "Independent environmental review for Acme.",
                }]

        diagnostics = {}
        with patch.object(web_search, "DDGS", BackendDDGS):
            results = web_search.search_web(
                "Acme emissions assurance",
                company_name="Acme",
                diagnostics=diagnostics,
                backend="brave",
            )

        self.assertEqual(results[0]["search_backend"], "brave")
        self.assertEqual(diagnostics["backend"], "brave")

    def test_empty_company_search_persists_clear_diagnostics(self):
        import automatic_collector

        def empty_search(name, search_type, diagnostics=None, **_kwargs):
            diagnostics.update({
                "search_type": search_type,
                "status": "no_results",
                "query_count": 1,
                "queries": [{
                    "query": f"{name} environment",
                    "status": "no_results",
                    "raw_result_count": 0,
                    "accepted_result_count": 0,
                    "error": None,
                }],
                "result_count": 0,
                "search_error_count": 0,
            })
            return []

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(
                    automatic_collector,
                    "OUTPUT_DIR",
                    Path(temporary_directory),
                ),
                patch.object(
                    automatic_collector,
                    "search_environmental_information",
                    side_effect=empty_search,
                ),
            ):
                result = automatic_collector.collect_company_esg_information(
                    "Acme",
                )

            output_file = Path(result["output_file"])
            self.assertTrue(output_file.is_file())
            with output_file.open("r", encoding="utf-8") as file:
                saved = json.load(file)

        self.assertEqual(result["search_results_count"], 0)
        self.assertEqual(result["documents"], [])
        self.assertEqual(saved["diagnostics"]["search"]["status"], "no_results")
        self.assertEqual(saved["diagnostics"]["downloads"]["attempt_count"], 0)

    def test_http_failure_retains_status_and_final_url(self):
        import requests
        import automatic_collector

        class ForbiddenResponse:
            status_code = 403
            url = "https://source.example/report"
            headers = {}
            is_redirect = False

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def raise_for_status(self):
                raise requests.HTTPError(
                    "403 Forbidden",
                    response=self,
                )

        class Session:
            def get(self, *args, **kwargs):
                return ForbiddenResponse()

        with patch.object(
            automatic_collector,
            "is_allowed_by_robots",
            return_value=True,
        ):
            document = automatic_collector.extract_html_document(
                Session(),
                "https://source.example/report",
            )

        self.assertEqual(document["status"], "failed")
        self.assertEqual(document["http_status"], 403)
        self.assertEqual(
            document["final_url"],
            "https://source.example/report",
        )


    def test_search_web_normalizes_mocked_company_results(self):
        import web_search

        calls = []

        class MockDDGS:
            def __init__(self, timeout):
                calls.append(("timeout", timeout))

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def text(self, query, max_results):
                calls.append((query, max_results))
                return [
                    {
                        "title": "Apple Environmental Report",
                        "href": (
                            "https://www.apple.com/environment/report.pdf"
                            "#details"
                        ),
                        "body": "Apple environmental reporting.",
                    },
                    {
                        "title": "Invalid result",
                        "href": "javascript:alert(1)",
                        "body": "Not a usable source",
                    },
                ]

        with patch.object(web_search, "DDGS", MockDDGS):
            results = web_search.search_web(
                "Apple environmental report",
                max_results=2,
                company_name="Apple",
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["url"],
            "https://www.apple.com/environment/report.pdf",
        )
        self.assertEqual(results[0]["source_type"], "Official company source")
        self.assertEqual(results[0]["domain"], "apple.com")
        self.assertEqual(calls[-1], ("Apple environmental report", 2))

    def test_company_collector_uses_mocked_company_search_and_preserves_source(self):
        import automatic_collector

        searches = []

        def search(name, search_type, **_kwargs):
            searches.append((name, search_type))
            return [{
                "title": "Apple Environmental Report",
                "url": "https://www.apple.com/environment/report",
                "content_type": "Web page or unknown",
            }]

        def collect_documents(
            search_results,
            search_name,
            max_documents,
            progress_callback=None,
        ):
            return ([{
                "title": search_results[0]["title"],
                "source_url": search_results[0]["url"],
                "status": "success",
                "text": f"Environmental report for {search_name}.",
            }], "apple.json")

        with (
            patch.object(
                automatic_collector,
                "search_environmental_information",
                side_effect=search,
            ),
            patch.object(
                automatic_collector,
                "collect_documents",
                side_effect=collect_documents,
            ),
        ):
            result = automatic_collector.collect_company_esg_information(
                "Apple",
                max_documents=1,
            )

        self.assertEqual(searches, [("Apple", "company")])
        self.assertEqual(result["company"], "Apple")
        self.assertEqual(result["successful_documents"], 1)
        self.assertEqual(
            result["documents"][0]["source_url"],
            "https://www.apple.com/environment/report",
        )

    def test_company_collection_persists_candidate_attempt_outcomes(self):
        import automatic_collector

        search_results = [
            {
                "title": f"Acme emissions source {index}",
                "url": f"https://research.example/report-{index}",
                "domain": "research.example",
                "source_type": "Other / unverified",
                "search_query": "Acme emissions verification",
                "search_backend": "brave",
                "attribution_status": "not_recorded",
            }
            for index in range(4)
        ]

        def search(name, search_type, diagnostics=None, **_kwargs):
            diagnostics.update({
                "search_type": search_type,
                "status": "completed",
                "result_count": len(search_results),
            })
            return search_results

        def collect_documents(
            search_results,
            search_name,
            max_documents,
            progress_callback=None,
        ):
            document = {
                "title": search_results[0]["title"],
                "source_url": search_results[0]["url"],
                "status": "success",
                "text": f"Environmental report for {search_name}.",
            }
            return (
                [document],
                automatic_collector.save_documents([document], search_name),
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(
                    automatic_collector,
                    "OUTPUT_DIR",
                    Path(temporary_directory),
                ),
                patch.object(
                    automatic_collector,
                    "search_environmental_information",
                    side_effect=search,
                ),
                patch.object(
                    automatic_collector,
                    "collect_documents",
                    side_effect=collect_documents,
                ),
            ):
                result = automatic_collector.collect_company_esg_information(
                    "Acme",
                    max_documents=1,
                )

            with Path(result["output_file"]).open(
                "r",
                encoding="utf-8",
            ) as file:
                saved = json.load(file)

        candidates = saved["diagnostics"]["search"]["candidate_results"]
        self.assertEqual(len(candidates), 4)
        self.assertEqual(
            [candidate["download_status"] for candidate in candidates],
            ["attempted", "not_attempted", "not_attempted", "not_attempted"],
        )
        self.assertEqual(candidates[0]["document_status"], "success")
        self.assertEqual(
            candidates[1]["not_attempted_reason"],
            "successful_document_limit_reached",
        )
        self.assertEqual(
            candidates[3]["not_attempted_reason"],
            "attempt_limit_reached",
        )
        self.assertEqual(
            saved["diagnostics"]["search"]["attempted_result_count"],
            1,
        )
        self.assertEqual(
            saved["diagnostics"]["search"]["not_attempted_result_count"],
            3,
        )

    def test_search_web_reports_mocked_search_failure(self):
        import web_search

        errors = []

        with patch.object(web_search, "DDGS", side_effect=RuntimeError("offline")):
            results = web_search.search_web(
                "Acme sustainability",
                max_results=1,
                error_callback=lambda query, message: errors.append(
                    (query, message)
                ),
            )

        self.assertEqual(results, [])
        self.assertEqual(errors, [("Acme sustainability", "offline")])

    def test_company_search_rejects_unrelated_epa_apple_food_result(self):
        import web_search

        class MockDDGS:
            def __init__(self, timeout):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def text(self, query, max_results):
                return [
                    {
                        "title": "Canned and Preserved Fruits and Vegetables",
                        "href": "https://www.epa.gov/example/canned-fruit",
                        "body": (
                            "Energy information for canned apple products "
                            "and preserved vegetables."
                        ),
                    },
                    {
                        "title": "Oil and Petroleum Products Explained",
                        "href": "https://www.eia.gov/energyexplained/oil/",
                        "body": (
                            "General national energy data and emissions "
                            "statistics."
                        ),
                    },
                    {
                        "title": "Apple Inc. Environmental Assessment",
                        "href": "https://research.example/apple-assessment",
                        "body": (
                            "Independent assessment of Apple Inc. supplier "
                            "emissions and operations."
                        ),
                    },
                ]

        with patch.object(web_search, "DDGS", MockDDGS):
            results = web_search.search_web(
                "Apple environmental emissions",
                max_results=2,
                company_name="Apple",
                search_type="company",
            )

        self.assertEqual(len(results), 2)
        relevant_result = next(
            result
            for result in results
            if result["url"]
            == "https://research.example/apple-assessment"
        )
        pending_result = next(
            result
            for result in results
            if result["url"]
            == "https://www.eia.gov/energyexplained/oil/"
        )
        self.assertEqual(
            relevant_result["attribution_status"],
            "confirmed_by_search_metadata",
        )
        self.assertEqual(
            pending_result["attribution_status"],
            "pending_page_content",
        )
        self.assertIn("must confirm entity attribution", pending_result[
            "attribution_basis"
        ])
        self.assertNotIn(
            "https://www.epa.gov/example/canned-fruit",
            {result["url"] for result in results},
        )

    def test_result_without_entity_metadata_requires_environmental_cue(self):
        import web_search

        class MockDDGS:
            def __init__(self, timeout):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def text(self, query, max_results):
                return [
                    {
                        "title": "Environmental disclosure index",
                        "href": "https://research.example/environmental-index",
                        "body": "A review of corporate climate disclosures.",
                    },
                    {
                        "title": "Unrelated company profile",
                        "href": "https://research.example/company-profile",
                        "body": "Corporate contacts and product history.",
                    },
                ]

        diagnostics = {}
        with patch.object(web_search, "DDGS", MockDDGS):
            results = web_search.search_web(
                '"Acme Holdings" environmental disclosure',
                company_name="Acme Holdings",
                diagnostics=diagnostics,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["attribution_status"],
            "pending_page_content",
        )
        self.assertEqual(
            results[0]["search_query"],
            '"Acme Holdings" environmental disclosure',
        )
        self.assertEqual(
            diagnostics["rejected_results"][0]["url"],
            "https://research.example/company-profile",
        )
        self.assertEqual(
            diagnostics["rejected_results"][0]["reason"],
            "no_environmental_relevance_cue",
        )

    def test_empty_search_metadata_is_pending_not_entity_confirmed(self):
        import web_search

        class MockDDGS:
            def __init__(self, timeout):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def text(self, query, max_results):
                return [{
                    "title": "",
                    "href": "https://research.example/climate-report",
                    "body": "",
                }]

        with patch.object(web_search, "DDGS", MockDDGS):
            results = web_search.search_web(
                '"Acme" environmental emissions',
                company_name="Acme",
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["attribution_status"],
            "pending_page_content",
        )

    def test_entity_matching_requires_full_multiword_company_name(self):
        import web_search

        self.assertFalse(
            web_search.search_result_mentions_entity(
                "Acme Holdings",
                "company",
                "Acme Services environmental report",
            )
        )

    def test_ambiguous_apple_food_result_is_not_deferred(self):
        import web_search

        class MockDDGS:
            def __init__(self, timeout):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def text(self, query, max_results):
                return [{
                    "title": "Environmental impacts of canned apple products",
                    "href": "https://research.example/canned-apple",
                    "body": "Carbon emissions from preserved fruit processing.",
                }]

        with patch.object(web_search, "DDGS", MockDDGS):
            diagnostics = {}
            results = web_search.search_web(
                '"Apple" environmental emissions',
                company_name="Apple",
                diagnostics=diagnostics,
            )

        self.assertEqual(results, [])
        self.assertEqual(
            diagnostics["rejected_results"][0]["reason"],
            "ambiguous_company_reference",
        )



    def test_company_search_targets_independent_and_regulatory_reporting(self):
        import web_search

        search_calls = []

        class EmptyDDGS:
            def __init__(self, timeout):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def text(self, query, max_results):
                search_calls.append(query)
                return []

        with patch.object(web_search, "DDGS", EmptyDDGS):
            web_search.search_environmental_information(
                "Apple",
                search_type="company",
                max_results_per_query=1,
                delay_seconds=0,
            )

        query_text = " ".join(search_calls).casefold()
        self.assertIn("independent environmental assessment", query_text)
        self.assertIn("regulatory filing emissions", query_text)
        self.assertIn("limited assurance", query_text)

    def test_company_search_diagnostics_retain_deduplicated_result_origins(self):
        import web_search

        class RepeatedDDGS:
            def __init__(self, timeout):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def text(self, query, max_results):
                return [{
                    "title": "Acme emissions assessment",
                    "href": "https://research.example/acme-assessment",
                    "body": "Independent emissions assessment for Acme.",
                }]

        diagnostics = {}
        with patch.object(web_search, "DDGS", RepeatedDDGS):
            results = web_search.search_environmental_information(
                "Acme",
                max_results_per_query=1,
                delay_seconds=0,
                diagnostics=diagnostics,
            )

        self.assertEqual(len(results), 1)
        candidates = diagnostics["candidate_results"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0]["url"],
            "https://research.example/acme-assessment",
        )
        self.assertGreater(len(candidates[0]["queries"]), 1)
        self.assertEqual(
            {origin["query"] for origin in candidates[0]["queries"]},
            {origin["query"] for origin in diagnostics["queries"]},
        )

    def test_claim_evidence_queries_are_company_metric_and_year_specific(self):
        import web_search

        samsung_queries = web_search.build_claim_evidence_queries(
            "Samsung",
            [{
                "statement": (
                    "Samsung's Scope 1 emissions were 100 t CO2e "
                    "in 2024."
                ),
            }],
        )
        apple_queries = web_search.build_claim_evidence_queries(
            "Apple",
            [{
                "statement": (
                    "Apple reported renewable electricity at 80 percent "
                    "in 2023."
                ),
            }],
        )

        self.assertEqual(len(samsung_queries), 2)
        self.assertIn('"Samsung"', samsung_queries[0])
        self.assertIn('"carbon emissions"', samsung_queries[0])
        self.assertIn('"100 t CO2e"', samsung_queries[0])
        self.assertIn("2024", samsung_queries[0])
        self.assertIn("independent", samsung_queries[0])
        self.assertNotIn('"100 t CO2e"', samsung_queries[1])
        self.assertIn("2024", samsung_queries[1])
        self.assertIn("assurance", samsung_queries[1])
        self.assertEqual(len(apple_queries), 2)
        self.assertIn('"Apple"', apple_queries[0])
        self.assertIn('"renewable electricity"', apple_queries[0])
        self.assertIn("2023", apple_queries[0])
        self.assertNotIn("Samsung", apple_queries[0])
        self.assertNotIn('"80 percent"', apple_queries[1])
        self.assertIn("2023", apple_queries[1])
        self.assertIn("verification", apple_queries[1])

    def test_claim_evidence_queries_use_headline_percent_not_comparison_footnote(self):
        import web_search

        queries = web_search.build_claim_evidence_queries(
            "Samsung",
            [{
                "statement": (
                    "Transition to Renewable Energy 31 % transition rate; "
                    "11%p increase "
                    "compared to 2021."
                ),
            }],
        )

        self.assertEqual(len(queries), 3)
        self.assertIn('"31 %"', queries[0])
        self.assertIn('"renewable energy"', queries[0])
        self.assertNotIn("11%", queries[0])
        self.assertNotIn("2021", queries[0])
        self.assertIn('"renewable energy"', queries[1])
        self.assertNotIn('"31 %"', queries[1])
        self.assertNotIn("2021", queries[1])
        self.assertIn("assurance", queries[1])
        self.assertIn('"renewable energy transition rate"', queries[2])
        self.assertNotIn('"31 %"', queries[2])
        self.assertNotIn("2021", queries[2])

    def test_company_search_prioritizes_external_sources_after_one_official(self):
        import web_search

        class MockDDGS:
            def __init__(self, timeout):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception, traceback):
                return False

            def text(self, query, max_results):
                return [
                    {
                        "title": "Apple Environmental Report",
                        "href": "https://www.apple.com/environment/report.pdf",
                        "body": "Apple environmental report",
                    },
                    {
                        "title": "Apple Independent Environmental Review",
                        "href": "https://research.example/apple-review",
                        "body": "Independent Apple emissions assessment",
                    },
                    {
                        "title": "Apple Supplier Emissions Filing",
                        "href": "https://www.apple.com/environment/suppliers.pdf",
                        "body": "Apple supplier emissions",
                    },
                ]

        with patch.object(web_search, "DDGS", MockDDGS):
            results = web_search.search_environmental_information(
                "Apple",
                search_type="company",
                max_results_per_query=1,
                delay_seconds=0,
            )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["source_type"], "Official company source")
        self.assertEqual(
            results[1]["domain"],
            "research.example",
        )
        self.assertEqual(results[2]["source_type"], "Official company source")

    def test_html_collection_preserves_publisher_and_publication_date(self):
        import automatic_collector

        html = b"""
        <html>
          <head>
            <title>Galaxy S25 Environmental Assessment</title>
            <meta property="og:site_name" content="Independent Lab">
            <meta property="article:published_time" content="2025-03-04">
          </head>
          <body>
            <p>The Galaxy S25 lifecycle assessment reviews recycled materials.</p>
          </body>
        </html>
        """
        response_metadata = {
            "final_url": "https://research.example/galaxy-s25",
            "status_code": 200,
            "content_type": "text/html",
            "headers": {"Content-Type": "text/html; charset=utf-8"},
        }
        with patch.object(
            automatic_collector,
            "download_content",
            return_value=(response_metadata, html),
        ):
            document = automatic_collector.extract_html_document(
                session=object(),
                url="https://research.example/galaxy-s25",
                search_result={
                    "title": "Search result title",
                    "source_type": "Other / unverified",
                },
            )

        self.assertEqual(document["status"], "success")
        self.assertEqual(document["title"], "Galaxy S25 Environmental Assessment")
        self.assertEqual(document["publisher"], "Independent Lab")
        self.assertEqual(document["publication_date"], "2025-03-04")
        self.assertEqual(document["source_url"], "https://research.example/galaxy-s25")
