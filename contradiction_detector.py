import json
import re
import sys
from itertools import zip_longest
from pathlib import Path
from unittest.mock import Mock

import transformers

from online_evidence_matcher import (
    POTENTIAL_MATCH_THRESHOLD,
    STRONG_MATCH_THRESHOLD,
)
from timeline_analyzer import compare_measurement_compatibility

if "pipeline" not in transformers.__dict__:
    try:
        from transformers.pipelines import pipeline as _transformers_pipeline
        transformers.pipeline = _transformers_pipeline
    except Exception:
        pass

sys.modules["transformers"] = transformers

_NLI_CLASSIFIER = None

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
INPUT_FILE = OUTPUT_DIR / "online_evidence_matches.json"
OUTPUT_FILE = OUTPUT_DIR / "online_contradiction_results.json"

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
NLI_BATCH_SIZE = 16
EXPLICIT_ENTITY_PATTERNS = (
    re.compile(
        r"\b(?P<entity>[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,2})"
        r"(?:['’]s)\s+(?:(?i:scope\s*[123])\s+)?"
        r"(?i:emissions?|carbon footprint|operations|facilities)\b"
    ),
    re.compile(
        r"\b(?P<entity>[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,2})"
        r"\s+(?:reported|reports|disclosed|discloses|emitted|measured)\b"
    ),
)


def load_online_matches():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Online matching results not found: {INPUT_FILE}\n"
            "Run online_evidence_matcher.py first."
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_label(label):
    if label is None:
        return "UNKNOWN"

    normalized = str(label).strip().upper()

    if normalized in {"0", "LABEL_0"}:
        return "ENTAILMENT"
    if normalized in {"1", "LABEL_1"}:
        return "NEUTRAL"
    if normalized in {"2", "LABEL_2"}:
        return "CONTRADICTION"

    if "CONTRADICTION" in normalized:
        return "CONTRADICTION"

    if "ENTAILMENT" in normalized:
        return "ENTAILMENT"

    if "NEUTRAL" in normalized:
        return "NEUTRAL"

    return "UNKNOWN"


def _get_nli_classifier():
    global _NLI_CLASSIFIER
    pipeline = getattr(transformers, "pipeline", None)
    if _NLI_CLASSIFIER is not None and not isinstance(pipeline, Mock):
        return _NLI_CLASSIFIER

    try:
        _NLI_CLASSIFIER = transformers.pipeline(
            "text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
        )
    except Exception as error:
        raise RuntimeError(
            "NLI model could not be loaded; offline or missing model support "
            f"prevented inference ({type(error).__name__}: {error})."
        ) from error
    return _NLI_CLASSIFIER


def _coerce_float(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or abs(numeric) == float("inf"):
        return None
    return numeric


def _normalize_entity_name(value):
    tokens = re.findall(r"[a-z0-9]+", str(value or "").casefold())
    suffixes = {
        "co", "company", "corp", "corporation", "inc", "incorporated",
        "limited", "llc", "ltd", "plc",
    }
    while tokens and tokens[-1] in suffixes:
        tokens.pop()
    while tokens and tokens[0] == "the":
        tokens.pop(0)
    return " ".join(tokens)


def _explicit_entities(text):
    entities = set()
    for pattern in EXPLICIT_ENTITY_PATTERNS:
        for match in pattern.finditer(str(text or "")):
            entity = _normalize_entity_name(match.group("entity"))
            if entity not in {
                "a", "an", "the", "this", "these", "those",
                "company", "group", "business",
            }:
                entities.add(entity)
    return entities


def detect_online_contradictions(data):
    claims_data = data.get("results", [])

    if not claims_data:
        print("No online claim results found.")
        return []

    evidence_pairs = []
    skipped_pairs = []

    for claim_item in claims_data:
        claim = claim_item.get("claim", "")

        for match in claim_item.get("matches", []):
            evidence = match.get("matched_evidence", "").strip()

            if not claim.strip() or not evidence:
                continue

            pair = {
                "claim": claim,
                "evidence": evidence,
                "similarity": match.get("similarity"),
                "source_title": match.get("source_title", ""),
                "source_url": match.get("source_url", ""),
                "original_source_url": match.get(
                    "original_source_url",
                    match.get("source_url", ""),
                ),
                "publisher": match.get("publisher", ""),
                "source_domain": match.get("source_domain", ""),
                "source_role": match.get("source_role", "unknown"),
                "entity_context": match.get("entity_context"),
                "publication_date": match.get("publication_date"),
                "match_label": match.get("match_label", ""),
                "assurance_report": match.get("assurance_report", False),
                "assurance_scope": match.get("assurance_scope", ""),
                "covered_fiscal_period": match.get(
                    "covered_fiscal_period",
                    "",
                ),
                "scope_exclusions": match.get("scope_exclusions", []),
                "original_text": match.get("original_text", ""),
                "attribution_basis": match.get("attribution_basis", ""),
            }
            similarity = _coerce_float(pair.get("similarity"))
            match_label = str(pair.get("match_label") or "").strip().lower()

            def mark_inconclusive(status, reason):
                pair["nli_result"] = "UNAVAILABLE"
                pair["compatibility_status"] = status
                pair["compatibility_reason"] = reason
                pair["scoring_eligible"] = False
                pair["warning"] = (
                    f"{reason} The candidate and source link are retained, "
                    "but this pair was not used as a contradiction or support signal."
                )
                skipped_pairs.append(pair)

            if match_label == "no_sufficiently_relevant_match" or (
                similarity is not None
                and similarity < POTENTIAL_MATCH_THRESHOLD
            ):
                mark_inconclusive(
                    "INCONCLUSIVE",
                    "Similarity was below the evidence threshold.",
                )
                continue
            if match_label == "weak_match" or (
                similarity is not None
                and similarity < STRONG_MATCH_THRESHOLD
            ):
                mark_inconclusive(
                    "INCONCLUSIVE",
                    "Similarity was below the strong-match threshold.",
                )
                continue

            expected_entity = data.get("company") or data.get("product")
            entity_context = pair.get("entity_context")
            if expected_entity:
                if not entity_context:
                    mark_inconclusive(
                        "INCONCLUSIVE",
                        "The source has no company or product context metadata.",
                    )
                    continue
                if (
                    _normalize_entity_name(entity_context)
                    != _normalize_entity_name(expected_entity)
                ):
                    mark_inconclusive(
                        "INCOMPATIBLE",
                        "The evidence source is associated with a different entity.",
                    )
                    continue
                explicit_entities = (
                    _explicit_entities(evidence)
                    if data.get("company")
                    else set()
                )
                expected_entity_name = _normalize_entity_name(expected_entity)
                if explicit_entities and expected_entity_name not in explicit_entities:
                    if re.search(
                        r"\b(subsidiar(?:y|ies)|affiliate|owned by|"
                        r"parent company|division of|business unit)\b",
                        evidence,
                        re.IGNORECASE,
                    ):
                        mark_inconclusive(
                            "INCONCLUSIVE",
                            "The named entity may have an organizational relationship that is not verified.",
                        )
                    else:
                        mark_inconclusive(
                            "INCOMPATIBLE",
                            "The evidence explicitly attributes the measurement to a different entity.",
                        )
                    continue

            compatibility, compatibility_reason = (
                compare_measurement_compatibility(
                    claim,
                    evidence,
                    pair,
                )
            )
            if compatibility in {"INCONCLUSIVE", "INCOMPATIBLE"}:
                mark_inconclusive(compatibility, compatibility_reason)
                continue
            pair["compatibility_status"] = compatibility
            pair["compatibility_reason"] = compatibility_reason
            pair["scoring_eligible"] = True
            evidence_pairs.append(pair)

    if not evidence_pairs and not skipped_pairs:
        print("No evidence statements passed the matching threshold.")
        return []

    if not evidence_pairs:
        return skipped_pairs

    print("Evidence pairs to analyze:", len(evidence_pairs))
    print("Loading NLI model...")

    classifier = _get_nli_classifier()

    results = []
    inputs = [
        {"text": pair["evidence"], "text_pair": pair["claim"]}
        for pair in evidence_pairs
    ]

    print(f"Running NLI in batches of {NLI_BATCH_SIZE}...")

    try:
        predictions = classifier(
            inputs,
            truncation=True,
            batch_size=NLI_BATCH_SIZE,
        )
    except Exception as error:
        print(f"Batched NLI failed ({error}); falling back to single pairs.")
        predictions = []
        for item in inputs:
            try:
                predictions.append(classifier(
                    item["text"],
                    text_pair=item["text_pair"],
                    truncation=True,
                ))
            except Exception as pair_error:
                print(f"Could not classify a pair: {pair_error}")
                predictions.append(None)

    for pair, prediction in zip_longest(evidence_pairs, predictions, fillvalue=None):
        if pair is None:
            continue
        if isinstance(prediction, list):
            prediction = prediction[0] if prediction else None

        if not isinstance(prediction, dict):
            results.append({
                **pair,
                "nli_result": "UNAVAILABLE",
                "scoring_eligible": False,
                "nli_failed": True,
                "raw_nli_label": None,
                "confidence": None,
                "interpretation": (
                    "The model returned an unrecognized output structure; "
                    "the result was marked unavailable and was not used for risk."
                ),
                "warning": (
                    "The model returned an unrecognized output structure; "
                    "this is not evidence of support or contradiction."
                ),
            })
            continue

        raw_label = prediction.get("label")
        if raw_label is None:
            results.append({
                **pair,
                "nli_result": "UNAVAILABLE",
                "scoring_eligible": False,
                "nli_failed": True,
                "raw_nli_label": None,
                "confidence": None,
                "interpretation": "The model did not return a usable label.",
                "warning": (
                    "The model output was incomplete; this prediction was not used "
                    "for risk scoring."
                ),
            })
            continue

        category = normalize_label(raw_label)
        if category == "UNKNOWN":
            category = "UNAVAILABLE"

        try:
            confidence = float(prediction.get("score", 0.0))
        except (TypeError, ValueError):
            confidence = None

        results.append({
            **pair,
            "nli_result": category,
            # An UNAVAILABLE label is a failed NLI run, not a completed
            # outcome: it is flagged so it cannot be counted as a zero-count
            # result. The matched evidence itself remains available.
            "scoring_eligible": category != "UNAVAILABLE",
            "nli_failed": category == "UNAVAILABLE",
            "raw_nli_label": raw_label,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "interpretation": (
                "Model predicts that the evidence supports the claim."
                if category == "ENTAILMENT"
                else
                "Model predicts that the evidence conflicts with the claim. "
                "Manual verification is required."
                if category == "CONTRADICTION"
                else
                "The evidence does not clearly support or contradict "
                "the claim according to the model."
                if category == "NEUTRAL"
                else
                "The model returned an unrecognized label or unavailable output; "
                "inspect manually before drawing a conclusion."
            ),
            "warning": (
                "This is an automated NLI prediction, not proof of truth "
                "or greenwashing."
                if category in {"ENTAILMENT", "CONTRADICTION", "NEUTRAL"}
                else "The model output was unavailable and was not used as evidence."
            ),
        })

    results.extend(skipped_pairs)
    return results


def main():
    print("=" * 70)
    print("ONLINE ESG CONTRADICTION DETECTION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_online_matches()
    results = detect_online_contradictions(data)

    output_data = {
        "company": data.get("company", ""),
        "input_file": str(INPUT_FILE),
        "model": MODEL_NAME,
        "result_count": len(results),
        "results": results,
        "limitations": [
            "Claims currently come from the sample ESG report.",
            "Claims are not necessarily claims made by the searched company.",
            "NLI labels are model predictions and require verification.",
            "Similarity and NLI predictions do not independently establish "
            "greenwashing."
        ]
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("ONLINE CONTRADICTION RESULTS")
    print("=" * 70)

    for index, item in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Claim:", item["claim"])
        print("Evidence:", item["evidence"][:350])
        print("Similarity:", item["similarity"])
        print("NLI classification:", item["nli_result"])
        print("Confidence:", item["confidence"])
        print("Source role:", item["source_role"])
        print("Source URL:", item["source_url"])

    print("\nTotal evidence pairs analyzed:", len(results))
    print("Results saved to:", OUTPUT_FILE)
    print("Online contradiction detection completed.")
    print("Review predictions and sources before drawing conclusions.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        print("File error:", error)
    except Exception as error:
        print("An error occurred:", error)
        print("Check the model installation and internet connection.")
