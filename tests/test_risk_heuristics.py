import unittest
from unittest import mock

import pandas as pd

from contradiction_detector import detect_online_contradictions
from feature_builder import build_features_from_results
from online_evidence_matcher import match_online_evidence
from risk_predictor import predict_risk_from_features
from timeline_analyzer import compare_measurement_compatibility


class _SimilarityRow(list):
    def tolist(self):
        return list(self)


class _SimilarityMatrix(list):
    def __getitem__(self, index):
        return _SimilarityRow(super().__getitem__(index))


class RiskHeuristicTests(unittest.TestCase):
    def test_synthetic_end_to_end_matching_nli_and_risk_pipeline(self):
        claim = (
            "Acme reported 1,200 metric tons CO2e of Scope 1 emissions in "
            "2023 for global operations under the GHG Protocol using the "
            "location-based method."
        )
        evidence = (
            "Acme reported 1,000 metric tons CO2e of Scope 1 emissions in "
            "2023 for global operations under the GHG Protocol using the "
            "location-based method, according to independently audited "
            "facility data."
        )
        source_url = "https://research.example/acme-2023-inventory"
        candidate = {
            "statement": evidence,
            "source_role": "independent_third_party_source",
            "source_url": source_url,
            "original_source_url": source_url,
            "source_title": "Synthetic integration fixture (not live evidence)",
            "publisher": "Independent inventory provider (fixture)",
            "source_domain": "research.example",
            "entity_context": "Acme",
            "publication_date": "2024-06-01",
        }
        embedding_model = mock.Mock()
        embedding_model.encode.side_effect = lambda texts, **_kwargs: texts

        with mock.patch(
            "sentence_transformers.util.cos_sim",
            return_value=_SimilarityMatrix([[0.83]]),
        ):
            matching_results = match_online_evidence(
                [claim],
                [candidate],
                embedding_model,
            )

        self.assertEqual(len(matching_results), 1)
        self.assertEqual(matching_results[0]["matches_found"], 1)
        matched = matching_results[0]["matches"][0]
        self.assertEqual(matched["source_url"], source_url)
        self.assertEqual(
            matched["source_role"],
            "independent_third_party_source",
        )
        self.assertEqual(matched["similarity"], 0.83)
        self.assertEqual(matched["match_label"], "potentially_relevant")

        classifier = mock.Mock(return_value=[{
            "label": "CONTRADICTION",
            "score": 0.91,
        }])
        with mock.patch(
            "contradiction_detector._get_nli_classifier",
            return_value=classifier,
        ):
            nli_results = detect_online_contradictions({
                "company": "Acme",
                "results": matching_results,
            })

        self.assertEqual(len(nli_results), 1)
        nli_result = nli_results[0]
        self.assertEqual(nli_result["compatibility_status"], "COMPATIBLE")
        self.assertTrue(nli_result["scoring_eligible"])
        self.assertEqual(nli_result["nli_result"], "CONTRADICTION")
        self.assertEqual(nli_result["source_url"], source_url)
        self.assertIn("not proof", nli_result["warning"])
        classifier.assert_called_once()

        features = build_features_from_results(
            matching_results,
            nli_results,
        )
        self.assertEqual(features.loc[0, "evidence_available"], 1)
        self.assertEqual(features.loc[0, "nli_contradiction"], 1)

        predictions = predict_risk_from_features(features)
        self.assertEqual(predictions.loc[0, "risk_score_status"], "HEURISTIC")
        self.assertEqual(
            predictions.loc[0, "demonstration_risk_category"],
            "HIGH",
        )
        self.assertTrue(
            pd.isna(predictions.loc[0, "demonstration_risk_probability"])
        )

    def test_heuristic_scores_distinguish_support_contradiction_and_missing_evidence(self):
        rows = [
            {
                "claim_id": 1,
                "similarity": 0.95,
                "evidence_available": 1,
                "nli_entailment": 1,
                "nli_contradiction": 0,
                "nli_neutral": 0,
                "percentage_difference": None,
                "percentage_difference_missing": 0,
                "historical_change": None,
            },
            {
                "claim_id": 2,
                "similarity": 0.90,
                "evidence_available": 1,
                "nli_entailment": 0,
                "nli_contradiction": 1,
                "nli_neutral": 0,
                "percentage_difference": None,
                "percentage_difference_missing": 0,
                "historical_change": None,
            },
            {
                "claim_id": 3,
                "similarity": 0.0,
                "evidence_available": 0,
                "nli_entailment": 0,
                "nli_contradiction": 1,
                "nli_neutral": 0,
                "percentage_difference": None,
                "percentage_difference_missing": 1,
                "historical_change": None,
            },
            {
                "claim_id": 4,
                "similarity": 0.70,
                "evidence_available": 1,
                "nli_entailment": None,
                "nli_contradiction": None,
                "nli_neutral": None,
                "percentage_difference": None,
                "percentage_difference_missing": 0,
                "historical_change": None,
            },
            {
                "claim_id": 5,
                "similarity": 0.55,
                "evidence_available": 1,
                "nli_entailment": 0,
                "nli_contradiction": 0,
                "nli_neutral": 1,
                "percentage_difference": None,
                "percentage_difference_missing": 0,
                "historical_change": None,
            },
        ]
        output = predict_risk_from_features(pd.DataFrame(rows))
        # Claim 4 (index 3) has evidence but no completed NLI run. Per the
        # documented policy it is unassessed, so it is not compared as a score.
        scores = output.loc[[0, 1, 4], "risk_score"].astype(float).tolist()
        self.assertTrue(
            output["demonstration_risk_probability"].isna().all()
        )
        self.assertTrue(all(0 <= score <= 100 for score in scores))
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], scores[2])
        self.assertGreater(scores[2], scores[0])
        self.assertTrue(len(set(scores)) > 1)
        self.assertEqual(
            output["risk_score_status"].tolist(),
            [
                "HEURISTIC",
                "HEURISTIC",
                "NOT_PREDICTED",
                "NOT_PREDICTED",
                "HEURISTIC",
            ],
        )
        self.assertTrue(pd.isna(output.loc[2, "risk_score"]))
        self.assertTrue(pd.isna(output.loc[2, "predicted_risk_class"]))
        self.assertEqual(
            output.loc[2, "demonstration_risk_category"],
            "NOT_PREDICTED",
        )
        self.assertTrue(pd.isna(output.loc[3, "risk_score"]))
        self.assertTrue(pd.isna(output.loc[3, "predicted_risk_class"]))
        self.assertEqual(
            output.loc[3, "demonstration_risk_category"],
            "NOT_PREDICTED",
        )

    def test_malformed_inputs_fail_safely(self):
        malformed = pd.DataFrame([
            {
                "claim_id": 99,
                "similarity": "not-a-number",
                "evidence_available": "yes",
                "nli_entailment": "bad",
                "nli_contradiction": "bad",
                "nli_neutral": "bad",
                "percentage_difference": "oops",
                "percentage_difference_missing": "bad",
                "historical_change": None,
            }
        ])
        output = predict_risk_from_features(malformed)
        # Unparseable NLI values are unavailable NLI, not a completed run, so
        # the claim is unassessed. Malformed input must not crash the predictor
        # or produce an out-of-range or invented score.
        self.assertEqual(output["risk_score_status"].iloc[0], "NOT_PREDICTED")
        self.assertTrue(pd.isna(output["risk_score"].iloc[0]))
        self.assertTrue(pd.isna(output["predicted_risk_class"].iloc[0]))
        self.assertEqual(
            output["demonstration_risk_category"].iloc[0],
            "NOT_PREDICTED",
        )
        self.assertIn(
            "NLI verification did not produce a usable result",
            output["risk_unavailable_reason"].iloc[0],
        )

    def test_missing_or_unavailable_nli_is_not_treated_as_greenwashing(self):
        rows = [
            {
                "claim_id": 1,
                "similarity": 0.12,
                "evidence_available": 0,
                "nli_entailment": 0,
                "nli_contradiction": 0,
                "nli_neutral": 0,
                "percentage_difference": None,
                "percentage_difference_missing": 1,
                "historical_change": None,
            },
            {
                "claim_id": 2,
                "similarity": 0.72,
                "evidence_available": 1,
                "nli_entailment": 0,
                "nli_contradiction": 0,
                "nli_neutral": 0,
                "percentage_difference": None,
                "percentage_difference_missing": 0,
                "historical_change": None,
            },
            {
                "claim_id": 3,
                "similarity": 0.55,
                "evidence_available": 1,
                "nli_entailment": 0,
                "nli_contradiction": 0,
                "nli_neutral": 1,
                "percentage_difference": 12.0,
                "percentage_difference_missing": 0,
                "historical_change": None,
            },
        ]
        output = predict_risk_from_features(pd.DataFrame(rows))
        self.assertTrue(pd.isna(output["risk_score"].iloc[0]))
        self.assertLess(output["risk_score"].iloc[1], 35)
        self.assertLess(output["risk_score"].iloc[2], 50)
        self.assertEqual(
            output["demonstration_risk_category"].iloc[0],
            "NOT_PREDICTED",
        )
        self.assertEqual(output["demonstration_risk_category"].iloc[1], "LOW")
        self.assertEqual(output["percentage_difference_missing"].iloc[0], 1)
        self.assertEqual(output["percentage_difference_missing"].iloc[1], 0)

    def test_nli_unexpected_output_is_marked_unavailable(self):
        data = {
            "results": [{
                "claim": "Claim",
                "matches": [{
                    "matched_evidence": "Evidence",
                    "similarity": 0.81,
                    "match_label": "potentially_relevant",
                    "source_title": "Source",
                    "source_url": "https://example.com",
                    "publisher": "Pub",
                    "source_domain": "example.com",
                    "publication_date": "2024-01-01",
                }],
            }]
        }
        with mock.patch("transformers.pipeline", return_value=lambda *args, **kwargs: {"weird": "shape"}):
            results = detect_online_contradictions(data)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["nli_result"], "UNAVAILABLE")
        self.assertTrue(results[0]["warning"].startswith("The model returned an unrecognized"))

    def test_nli_skipped_low_similarity_pairs_do_not_load_classifier(self):
        data = {
            "results": [{
                "claim": "A company claim.",
                "matches": [{
                    "matched_evidence": "Unrelated text.",
                    "similarity": 0.2,
                    "match_label": "weak_match",
                    "source_url": "https://example.com/source",
                }],
            }]
        }

        with mock.patch(
            "contradiction_detector._get_nli_classifier"
        ) as get_classifier:
            results = detect_online_contradictions(data)

        get_classifier.assert_not_called()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["nli_result"], "UNAVAILABLE")
        self.assertIn("not used as a contradiction", results[0]["warning"])

    def test_weak_matches_cannot_create_contradiction_risk(self):
        match = {
            "matched_evidence": "Another company reports rising emissions.",
            "similarity": 0.42,
            "match_label": "weak_match",
            "source_url": "https://example.com/source",
        }
        data = {
            "results": [{
                "claim": "The company reduced its emissions.",
                "matches": [match],
            }]
        }

        with mock.patch(
            "contradiction_detector._get_nli_classifier"
        ) as get_classifier:
            nli_results = detect_online_contradictions(data)

        get_classifier.assert_not_called()
        self.assertEqual(nli_results[0]["nli_result"], "UNAVAILABLE")
        self.assertIn(
            "below the strong-match threshold",
            nli_results[0]["warning"],
        )

        features = build_features_from_results(
            [{
                "claim": data["results"][0]["claim"],
                "matches": [match],
            }],
            nli_results,
        )
        predictions = predict_risk_from_features(features)
        self.assertTrue(pd.isna(predictions.iloc[0]["nli_contradiction"]))
        self.assertEqual(predictions.iloc[0]["evidence_available"], 0)
        self.assertTrue(pd.isna(predictions.iloc[0]["risk_score"]))
        self.assertEqual(
            predictions.iloc[0]["risk_score_status"],
            "NOT_PREDICTED",
        )
        self.assertEqual(
            predictions.iloc[0]["demonstration_risk_category"],
            "NOT_PREDICTED",
        )

    def test_compatible_measurements_can_be_reviewed_by_nli(self):
        claim = (
            "Acme reported Scope 1 emissions of 1 tonne CO2e in 2022 for "
            "global operations under the GHG Protocol using the "
            "location-based method."
        )
        evidence = (
            "The company reported Scope 1 emissions of 1000 kg CO2e in 2022 "
            "for global operations under the GHG Protocol using the "
            "location-based method."
        )
        match = {
            "matched_evidence": evidence,
            "similarity": 0.91,
            "match_label": "potentially_relevant",
            "source_url": "https://example.com/report",
            "entity_context": "Acme Inc.",
        }
        self.assertEqual(
            compare_measurement_compatibility(claim, evidence)[0],
            "COMPATIBLE",
        )

        classifier = mock.Mock(return_value=[{
            "label": "CONTRADICTION",
            "score": 0.9,
        }])
        with mock.patch(
            "contradiction_detector._get_nli_classifier",
            return_value=classifier,
        ):
            results = detect_online_contradictions({
                "company": "Acme",
                "results": [{"claim": claim, "matches": [match]}],
            })

        classifier.assert_called_once()
        self.assertEqual(results[0]["nli_result"], "CONTRADICTION")
        self.assertEqual(results[0]["compatibility_status"], "COMPATIBLE")
        self.assertEqual(results[0]["source_url"], match["source_url"])
        self.assertTrue(results[0]["scoring_eligible"])

    def test_qualitative_environmental_pairs_are_not_blocked_by_measurement_gate(self):
        claim = "Company reports lower emissions."
        evidence = "Evidence supports lower emissions."
        self.assertEqual(
            compare_measurement_compatibility(claim, evidence)[0],
            "NOT_QUANTITATIVE",
        )

        match = {
            "matched_evidence": evidence,
            "similarity": 0.9,
            "match_label": "potentially_relevant",
            "source_url": "https://example.com/report",
        }
        classifier = mock.Mock(return_value=[{
            "label": "ENTAILMENT",
            "score": 0.9,
        }])
        with mock.patch(
            "contradiction_detector._get_nli_classifier",
            return_value=classifier,
        ):
            results = detect_online_contradictions({
                "results": [{"claim": claim, "matches": [match]}],
            })

        classifier.assert_called_once()
        self.assertEqual(results[0]["nli_result"], "ENTAILMENT")
        self.assertEqual(results[0]["compatibility_status"], "NOT_QUANTITATIVE")
        self.assertEqual(results[0]["source_url"], match["source_url"])
        self.assertTrue(results[0]["scoring_eligible"])

    def test_incompatible_year_scope_and_units_are_not_scored(self):
        cases = [
            (
                "Acme Scope 1 emissions were 100 t CO2e in 2022.",
                "Acme Scope 1 emissions were 80 t CO2e in 2023.",
            ),
            (
                "Acme Scope 1 emissions were 100 t CO2e in 2022.",
                "Acme Scope 2 emissions were 80 t CO2e in 2022.",
            ),
            (
                "Acme renewable energy consumption was 50 MWh in 2022.",
                "Acme renewable energy consumption was 50 GWh in 2022.",
            ),
            (
                "Acme Scope 1 emissions were 100 t CO2e in 2022.",
                "Acme renewable electricity supplied 80% in 2022.",
            ),
        ]
        for claim, evidence in cases:
            with self.subTest(claim=claim, evidence=evidence):
                match = {
                    "matched_evidence": evidence,
                    "similarity": 0.9,
                    "match_label": "potentially_relevant",
                    "source_url": "https://example.com/report",
                    "entity_context": "Acme",
                }
                status, _ = compare_measurement_compatibility(
                    claim,
                    evidence,
                )
                self.assertEqual(status, "INCOMPATIBLE")

                with mock.patch(
                    "contradiction_detector._get_nli_classifier"
                ) as get_classifier:
                    results = detect_online_contradictions({
                        "company": "Acme",
                        "results": [{"claim": claim, "matches": [match]}],
                    })

                get_classifier.assert_not_called()
                self.assertEqual(results[0]["nli_result"], "UNAVAILABLE")
                self.assertEqual(
                    results[0]["compatibility_status"],
                    "INCOMPATIBLE",
                )
                self.assertFalse(results[0]["scoring_eligible"])
                self.assertEqual(
                    results[0]["source_url"],
                    match["source_url"],
                )
                features = build_features_from_results(
                    [{"claim": claim, "matches": [match]}],
                    results,
                )
                self.assertEqual(features.iloc[0]["evidence_available"], 0)

    def test_contract_estimate_and_target_are_not_compared_as_outcomes(self):
        cases = [
            (
                "Samsung signed a 20-year solar power purchase agreement "
                "for 76.2 GWh per year.",
                "The solar plant generated 76.2 GWh in 2025.",
                "contractual_supply",
                "unspecified_outcome",
            ),
            (
                "The company targets 100% renewable electricity by 2030.",
                "Renewable electricity supplied 80% in 2024.",
                "target_or_commitment",
                "unspecified_outcome",
            ),
            (
                "The company forecasts renewable generation of 80 GWh.",
                "Renewable generation was 75 GWh in 2024.",
                "estimate_or_forecast",
                "unspecified_outcome",
            ),
        ]
        for claim, evidence, claim_mode, evidence_mode in cases:
            with self.subTest(claim=claim):
                status, reason = compare_measurement_compatibility(
                    claim,
                    evidence,
                )
                self.assertEqual(status, "INCONCLUSIVE")
                self.assertIn(claim_mode.replace("_", " "), reason)
                self.assertIn(evidence_mode.replace("_", " "), reason)

        status, reason = compare_measurement_compatibility(
            "The company targets 100% renewable electricity by 2030.",
            "The company forecasts renewable electricity of 80% by 2030.",
        )
        self.assertEqual(status, "INCOMPATIBLE")
        self.assertIn("Performance modalities differ", reason)

    def test_modeling_uncertainty_and_assurance_contract_are_not_supply_modes(self):
        reported = (
            "Apple reported Scope 1 emissions of 100 t CO2e in 2025 for "
            "global operations under the GHG Protocol."
        )
        assurance_context = (
            f"{reported} The assurance provider was contracted by Apple."
        )

        status, reason = compare_measurement_compatibility(
            reported,
            assurance_context,
        )
        self.assertEqual(status, "COMPATIBLE")
        self.assertNotIn("Performance modalities differ", reason)

        modeled_claim = (
            "Apple's illustrative FY2025 footprint was 15.3 million "
            "metric tons CO2e with modeling uncertainty for global "
            "operations under the GHG Protocol."
        )
        assured_outcome = (
            "Apple reported a FY2025 footprint of 15.28 million metric tons CO2e "
            "for global operations under the GHG Protocol."
        )
        status, reason = compare_measurement_compatibility(
            modeled_claim,
            assured_outcome,
        )
        self.assertEqual(status, "INCOMPATIBLE")
        self.assertIn("estimate or forecast", reason)
        self.assertIn("reported outcome", reason)

    def test_unspecified_measurement_status_is_inconclusive(self):
        claim = (
            "Acme Scope 1 emissions were 100 t CO2e in 2025 for global "
            "operations under the GHG Protocol."
        )
        evidence = claim.replace("100 t CO2e", "80 t CO2e")

        status, reason = compare_measurement_compatibility(claim, evidence)

        self.assertEqual(status, "INCONCLUSIVE")
        self.assertIn("reported-versus-estimated status", reason)

    def test_separate_historical_reduction_and_future_target_are_ambiguous(self):
        claim = (
            "Acme reduced Scope 1 emissions by 60 percent since 2015 and "
            "is committed to reducing them by 75 percent by 2030."
        )
        evidence = (
            "Acme reported Scope 1 emissions of 100 t CO2e in 2025 for "
            "global operations under the GHG Protocol."
        )

        status, reason = compare_measurement_compatibility(claim, evidence)

        self.assertEqual(status, "INCONCLUSIVE")
        self.assertIn("performance modality is ambiguous", reason)

    def test_emissions_reduction_requires_an_explicit_matching_baseline(self):
        claim = (
            "Acme reported Scope 1 emissions decreased 60 percent in 2025 "
            "from a 2015 baseline for global operations under the GHG "
            "Protocol."
        )
        missing_baseline_evidence = (
            "Acme reported Scope 1 emissions decreased 55 percent in 2025 "
            "for global operations under the GHG Protocol."
        )
        status, reason = compare_measurement_compatibility(
            claim,
            missing_baseline_evidence,
        )
        self.assertEqual(status, "INCONCLUSIVE")
        self.assertIn("baseline year", reason)

        matching_baseline_evidence = missing_baseline_evidence.replace(
            "in 2025 for",
            "in 2025 from a 2015 baseline for",
        )
        status, _ = compare_measurement_compatibility(
            claim,
            matching_baseline_evidence,
        )
        self.assertEqual(status, "COMPATIBLE")

    def test_worldwide_occupied_properties_is_an_explicit_boundary(self):
        claim = (
            "Apple reported Scope 1 emissions of 100 t CO2e in 2025 for "
            "worldwide occupied properties under the GHG Protocol."
        )
        evidence = claim.replace("100 t CO2e", "80 t CO2e")

        status, reason = compare_measurement_compatibility(claim, evidence)

        self.assertEqual(status, "COMPATIBLE")
        self.assertIn("organizational boundary", reason)

    def test_comparability_requires_matching_baseline_boundary_and_method(self):
        claim = (
            "Acme Scope 1 emissions were 100 t CO2e in 2025 from a 2015 "
            "baseline for global operations under the GHG Protocol using "
            "the location-based method."
        )
        cases = [
            (
                claim,
                claim.replace("2015 baseline", "2020 baseline"),
                "baseline years differ",
            ),
            (
                claim,
                claim.replace("global operations", "United States operations"),
                "organizational boundaries differ",
            ),
            (
                claim,
                claim.replace("location-based", "market-based"),
                "measurement methodologies differ",
            ),
        ]
        for first, second, reason in cases:
            with self.subTest(reason=reason):
                status, detail = compare_measurement_compatibility(
                    first,
                    second,
                )
                self.assertEqual(status, "INCOMPATIBLE")
                self.assertIn(reason, detail)

        missing_details = [
            (
                claim,
                claim.replace(" from a 2015 baseline", ""),
                "baseline",
            ),
            (
                claim,
                claim.replace(" for global operations", ""),
                "organizational boundary",
            ),
            (
                claim,
                claim.replace(
                    " under the GHG Protocol using the location-based method",
                    "",
                ),
                "methodology",
            ),
        ]
        for first, second, missing in missing_details:
            with self.subTest(missing=missing):
                status, reason = compare_measurement_compatibility(
                    first,
                    second,
                )
                self.assertEqual(status, "INCONCLUSIVE")
                self.assertIn(missing, reason)

        fiscal_claim = claim.replace("in 2025", "in FY2025")
        status, _ = compare_measurement_compatibility(
            fiscal_claim,
            claim,
        )
        self.assertEqual(status, "INCONCLUSIVE")

        q1_claim = claim.replace("in 2025", "in Q1 2025")
        status, reason = compare_measurement_compatibility(
            q1_claim,
            q1_claim.replace("Q1", "Q2"),
        )
        self.assertEqual(status, "INCOMPATIBLE")
        self.assertIn("reporting subperiods differ", reason)
        status, reason = compare_measurement_compatibility(q1_claim, claim)
        self.assertEqual(status, "INCONCLUSIVE")
        self.assertIn("reporting-period detail", reason)

    def test_known_incompatibility_is_not_hidden_by_other_missing_metadata(self):
        claim = (
            "Acme Scope 1 emissions were 100 t CO2e in 2025 for global "
            "operations under the GHG Protocol using the location-based method."
        )
        evidence = (
            "Acme Scope 1 emissions were 80 t CO2e in 2024 for global operations."
        )
        status, reason = compare_measurement_compatibility(claim, evidence)
        self.assertEqual(status, "INCOMPATIBLE")
        self.assertIn("reporting years differ", reason)

    def test_incompatible_baseline_is_not_sent_to_nli(self):
        claim = (
            "Acme Scope 1 emissions were 100 t CO2e in 2025 from a 2015 "
            "baseline for global operations under the GHG Protocol using "
            "the location-based method."
        )
        evidence = claim.replace("2015 baseline", "2020 baseline")
        match = {
            "matched_evidence": evidence,
            "similarity": 0.95,
            "match_label": "potentially_relevant",
            "source_url": "https://example.com/report",
            "entity_context": "Acme",
        }
        with mock.patch(
            "contradiction_detector._get_nli_classifier"
        ) as get_classifier:
            results = detect_online_contradictions({
                "company": "Acme",
                "results": [{"claim": claim, "matches": [match]}],
            })

        get_classifier.assert_not_called()
        self.assertEqual(results[0]["compatibility_status"], "INCOMPATIBLE")
        self.assertFalse(results[0]["scoring_eligible"])
        self.assertEqual(results[0]["nli_result"], "UNAVAILABLE")

    def test_contract_vs_outcome_does_not_call_nli_or_create_risk_signal(self):
        claim = (
            "The company signed a 20-year renewable energy supply contract "
            "for 76 GWh per year."
        )
        evidence = "The facility generated 76 GWh of renewable electricity in 2025."
        match = {
            "matched_evidence": evidence,
            "similarity": 0.92,
            "match_label": "potentially_relevant",
            "source_url": "https://regulator.example/project",
            "entity_context": "Acme",
        }
        with mock.patch(
            "contradiction_detector._get_nli_classifier"
        ) as get_classifier:
            results = detect_online_contradictions({
                "company": "Acme",
                "results": [{"claim": claim, "matches": [match]}],
            })

        get_classifier.assert_not_called()
        self.assertEqual(results[0]["nli_result"], "UNAVAILABLE")
        self.assertEqual(results[0]["compatibility_status"], "INCONCLUSIVE")
        self.assertFalse(results[0]["scoring_eligible"])
        features = build_features_from_results(
            [{"claim": claim, "matches": [match]}],
            results,
        )
        self.assertEqual(features.iloc[0]["evidence_available"], 0)

    def test_entity_context_mismatch_is_inconclusive_but_contextual_source_is_allowed(self):
        claim = (
            "Acme reported Scope 1 emissions of 100 t CO2e in 2022 for "
            "global operations under the GHG Protocol using the "
            "location-based method."
        )
        contextual_evidence = (
            "Scope 1 emissions were reported as 80 t CO2e in 2022 for "
            "global operations under the GHG Protocol using the "
            "location-based method."
        )
        contextual_match = {
            "matched_evidence": contextual_evidence,
            "similarity": 0.9,
            "match_label": "potentially_relevant",
            "source_url": "https://example.com/acme-report",
            "entity_context": "Acme",
        }
        classifier = mock.Mock(return_value=[{
            "label": "NEUTRAL",
            "score": 0.8,
        }])
        with mock.patch(
            "contradiction_detector._get_nli_classifier",
            return_value=classifier,
        ):
            contextual_result = detect_online_contradictions({
                "company": "Acme",
                "results": [{
                    "claim": claim,
                    "matches": [contextual_match],
                }],
            })

        classifier.assert_called_once()
        self.assertEqual(
            contextual_result[0]["nli_result"],
            "NEUTRAL",
        )
        self.assertEqual(
            contextual_result[0]["compatibility_status"],
            "COMPATIBLE",
        )
        self.assertEqual(
            contextual_result[0]["source_url"],
            contextual_match["source_url"],
        )

        wrong_entity_match = {
            **contextual_match,
            "entity_context": "Beta",
        }
        with mock.patch(
            "contradiction_detector._get_nli_classifier"
        ) as get_classifier:
            mismatch_result = detect_online_contradictions({
                "company": "Acme",
                "results": [{
                    "claim": claim,
                    "matches": [wrong_entity_match],
                }],
            })

        get_classifier.assert_not_called()
        self.assertEqual(mismatch_result[0]["nli_result"], "UNAVAILABLE")
        self.assertEqual(
            mismatch_result[0]["compatibility_status"],
            "INCOMPATIBLE",
        )
        self.assertEqual(
            mismatch_result[0]["source_url"],
            wrong_entity_match["source_url"],
        )

        explicitly_other_entity = {
            **contextual_match,
            "matched_evidence": (
                "Beta's Scope 1 emissions were 80 t CO2e in 2022 for global "
                "operations under the GHG Protocol using the location-based "
                "method."
            ),
            "entity_context": "Acme",
        }
        with mock.patch(
            "contradiction_detector._get_nli_classifier"
        ) as get_classifier:
            explicit_mismatch = detect_online_contradictions({
                "company": "Acme",
                "results": [{
                    "claim": claim,
                    "matches": [explicitly_other_entity],
                }],
            })
        get_classifier.assert_not_called()
        self.assertEqual(
            explicit_mismatch[0]["compatibility_status"],
            "INCOMPATIBLE",
        )
        self.assertEqual(
            explicit_mismatch[0]["source_url"],
            explicitly_other_entity["source_url"],
        )

        possible_subsidiary = {
            **explicitly_other_entity,
            "matched_evidence": (
                "Beta's Scope 1 emissions were 80 t CO2e in 2022; "
                "Beta is a subsidiary of Acme."
            ),
        }
        with mock.patch(
            "contradiction_detector._get_nli_classifier"
        ) as get_classifier:
            affiliate_result = detect_online_contradictions({
                "company": "Acme",
                "results": [{
                    "claim": claim,
                    "matches": [possible_subsidiary],
                }],
            })
        get_classifier.assert_not_called()
        self.assertEqual(
            affiliate_result[0]["compatibility_status"],
            "INCONCLUSIVE",
        )

    def test_missing_entity_or_measurement_metadata_is_inconclusive(self):
        cases = [
            (
                "Acme Scope 1 emissions were 100 t CO2e.",
                "Acme Scope 1 emissions were 80 t CO2e.",
                "Acme",
            ),
            (
                "Acme emissions were 100 in 2022.",
                "Acme emissions were 80 in 2022.",
                "Acme",
            ),
        ]
        for claim, evidence, entity_context in cases:
            with self.subTest(claim=claim, evidence=evidence):
                self.assertEqual(
                    compare_measurement_compatibility(
                        claim,
                        evidence,
                    )[0],
                    "INCONCLUSIVE",
                )

        claim = "Acme sustainability claims were reviewed."
        match = {
            "matched_evidence": "The report discusses environmental claims.",
            "similarity": 0.9,
            "match_label": "potentially_relevant",
            "source_url": "https://example.com/report",
        }
        with mock.patch(
            "contradiction_detector._get_nli_classifier"
        ) as get_classifier:
            results = detect_online_contradictions({
                "company": "Acme",
                "results": [{"claim": claim, "matches": [match]}],
            })

        get_classifier.assert_not_called()
        self.assertEqual(results[0]["nli_result"], "UNAVAILABLE")
        self.assertEqual(
            results[0]["compatibility_status"],
            "INCONCLUSIVE",
        )


if __name__ == "__main__":
    unittest.main()
