import json
import os
import pandas as pd


def _nli_unavailable(result):
    """True when the NLI step failed for this pair (no usable label).

    A failed run must never be counted as a completed zero-count outcome.
    The matched evidence itself is still real, so it is not excluded here.
    """
    return str(result.get("nli_result") or "").strip().upper() == "UNAVAILABLE"


def _pair_key(claim, evidence, source_url):
    """Identity of a claim-evidence pair, matching how the NLI stage keys it."""
    return (
        str(claim or "").strip(),
        str(evidence or "").strip(),
        str(source_url or ""),
    )


def build_features_from_results(matching_results, nli_results):
    """Build model features from in-memory matching and NLI results."""

    nli_by_claim = {}
    excluded_pairs = set()

    for result in nli_results:
        claim = result.get("claim", "").strip()

        if claim:
            nli_by_claim.setdefault(claim, []).append(result)
            # Evidence-level exclusions (incompatible/inconclusive pairs) remove
            # the match. A failed NLI run keeps the evidence, which is then
            # reported as NLI-unavailable rather than as "no evidence".
            if (
                result.get("scoring_eligible") is False
                and not result.get("nli_failed")
            ):
                excluded_pairs.add((
                    claim,
                    result.get("evidence", ""),
                    result.get("source_url", ""),
                ))

    data = []

    for claim_id, item in enumerate(matching_results, start=1):
        claim = item.get("claim", "").strip()
        matches = item.get("matches", [])
        scoring_matches = [
            match
            for match in matches
            if (
                match.get("scoring_eligible") is not False
                and str(match.get("match_label") or "").casefold()
                not in {
                    "weak_match",
                    "no_sufficiently_relevant_match",
                }
                and (
                    claim,
                    match.get("matched_evidence", ""),
                    match.get("source_url", ""),
                ) not in excluded_pairs
            )
        ]

        similarities = [
            float(match["similarity"])
            for match in scoring_matches
            if match.get("similarity") is not None
        ]
        # Coverage policy (D2): a claim is assessed only when every eligible
        # matched evidence pair has a completed NLI result. Counts are restricted
        # to pairs that belong to this claim's eligible matches, so an NLI row
        # for other evidence cannot inflate the coverage.
        eligible_keys = []
        for match in scoring_matches:
            key = _pair_key(claim, match.get("matched_evidence"), match.get("source_url"))
            if key not in eligible_keys:
                eligible_keys.append(key)

        completed_by_key = {}
        for result in nli_by_claim.get(claim, []):
            if result.get("scoring_eligible") is False or _nli_unavailable(result):
                continue
            key = _pair_key(claim, result.get("evidence"), result.get("source_url"))
            completed_by_key.setdefault(key, []).append(result)

        completed_keys = [key for key in eligible_keys if key in completed_by_key]
        claim_nli = [
            result
            for key in completed_keys
            for result in completed_by_key[key]
        ]
        pairs_eligible = len(eligible_keys)
        pairs_completed = len(completed_keys)
        if pairs_eligible == 0:
            coverage_complete = None
        else:
            coverage_complete = int(pairs_completed == pairs_eligible)

        data.append({
            "claim_id": claim_id,
            "claim": claim,
            "similarity": max(similarities) if similarities else None,
            "evidence_available": int(bool(scoring_matches)),
            "nli_entailment": (
                sum(
                    result.get("nli_result") == "ENTAILMENT"
                    for result in claim_nli
                )
                if claim_nli else None
            ),
            "nli_contradiction": (
                sum(
                    result.get("nli_result") == "CONTRADICTION"
                    for result in claim_nli
                )
                if claim_nli else None
            ),
            "nli_neutral": (
                sum(
                    result.get("nli_result") == "NEUTRAL"
                    for result in claim_nli
                )
                if claim_nli else None
            ),
            "nli_pairs_eligible": pairs_eligible,
            "nli_pairs_completed": pairs_completed,
            "nli_coverage_complete": coverage_complete,
            "percentage_difference": None,
            "percentage_difference_missing": 1,
            "historical_change": None,
        })

    return pd.DataFrame(data)


def build_features():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")

    matching_file = os.path.join(
        output_dir, "online_evidence_matches.json"
    )
    contradiction_file = os.path.join(
        output_dir, "online_contradiction_results.json"
    )

    if not os.path.exists(matching_file):
        print("ERROR: online_evidence_matches.json was not found.")
        print("Run online_evidence_matcher.py first.")
        return

    if not os.path.exists(contradiction_file):
        print("ERROR: online_contradiction_results.json was not found.")
        print("Run contradiction_detector.py first.")
        return

    with open(matching_file, "r", encoding="utf-8") as file:
        matching_data = json.load(file)

    with open(contradiction_file, "r", encoding="utf-8") as file:
        contradiction_data = json.load(file)

    matching_results = matching_data.get("results", [])
    nli_results = contradiction_data.get("results", [])

    if not matching_results:
        print("ERROR: No online matching results found.")
        return

    features = build_features_from_results(
        matching_results,
        nli_results,
    )

    print("=" * 75)
    print("ONLINE ESG FEATURE ENGINEERING")
    print("=" * 75)
    print(features.to_string(index=False))

    output_file = os.path.join(output_dir, "esg_features.csv")
    features.to_csv(output_file, index=False)

    print("\nFeatures saved to:", output_file)
    print("Claims processed:", len(features))
    print("Source: online_evidence_matches.json")
    print("NLI source: online_contradiction_results.json")
    print("Percentage differences: unavailable")
    print("Historical changes: unavailable")
    print(
        "\nWARNING: NLI labels and similarity scores are experimental "
        "signals, not verified evidence of greenwashing."
    )


if __name__ == "__main__":
    build_features()
