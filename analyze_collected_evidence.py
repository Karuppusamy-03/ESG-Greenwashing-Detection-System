import json
import re
from pathlib import Path

from online_claim_extractor import (
    extract_embedded_assurance_statements,
    identify_source_role,
    split_into_sentences,
)
from report_extractor import extract_text
from claim_extractor import extract_claims_and_evidence
from web_search import classify_source
from evidence_matcher import match_evidence


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
COLLECTED_DIR = BASE_DIR / "data" / "collected_documents"
SAMPLE_REPORT = BASE_DIR / "data" / "sample_esg_report.txt"

# Temporary demonstration threshold.
# This value has not been scientifically validated.
SIMILARITY_THRESHOLD = 0.50


# ============================================================
# ENVIRONMENTAL KEYWORDS
# ============================================================

ENVIRONMENTAL_TERMS = [
    "emissions",
    "greenhouse gas",
    "carbon",
    "climate",
    "renewable energy",
    "renewable electricity",
    "environmental",
    "sustainability",
    "pollution",
    "clean air",
    "hazardous waste",
    "energy efficiency",
    "zero emission",
    "zero-emission",
    "electric vehicle",
    "compliance",
    "net zero",
    "carbon neutral",
    "waste management",
    "energy consumption",
    "water usage",
    "water consumption",
    "recycling",
    "deforestation",
    "renewable power",
]


# ============================================================
# LOAD THE LATEST COLLECTED DOCUMENTS
# ============================================================

def load_latest_collection(company_name=None):
    """Load the newest valid collection JSON for the active company when possible."""

    if not COLLECTED_DIR.exists():
        raise FileNotFoundError(
            f"Collection directory not found: {COLLECTED_DIR}"
        )

    files = sorted(
        COLLECTED_DIR.glob("*.json"),
        key=lambda file: file.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            "No collected JSON files found. "
            "Run automatic_collector.py first."
        )

    def _matches_company(path):
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            return False

        if not isinstance(data, dict):
            return False

        if company_name is None:
            return True

        target = str(company_name).strip()
        if not target:
            return True

        search_name = str(data.get("search_name") or "").strip()
        return search_name.casefold() == target.casefold()

    candidates = [path for path in files if _matches_company(path)]
    if company_name and str(company_name).strip() and not candidates:
        raise FileNotFoundError(
            f"No document collection found for company "
            f"{str(company_name).strip()!r}."
        )
    latest_file = candidates[0] if candidates else files[0]

    with latest_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "The collection JSON must contain a dictionary."
        )

    print(f"\nCollection file: {latest_file.name}")

    return data


# ============================================================
# EXTRACT ENVIRONMENTAL SENTENCES
# ============================================================

def extract_environmental_sentences(documents, entity_name=None):
    """
    Extract potentially relevant sentences from successfully
    collected documents while preserving their source details.
    """

    evidence_records = []
    seen_sentences = set()

    for document in documents:
        if entity_name:
            source_url = (
                document.get("final_url")
                or document.get("source_url")
                or ""
            )
            if source_url:
                document = {
                    **document,
                    "source_type": classify_source(
                        source_url,
                        company_name=entity_name,
                    ),
                }

        if document.get("status") != "success":
            source_url = (
                document.get("final_url")
                or document.get("source_url")
                or ""
            )
            source_role = identify_source_role(document)
            http_status = document.get("http_status")
            error = str(document.get("error") or "").strip()
            failure_detail = (
                f" HTTP {http_status}." if http_status is not None else ""
            )
            if error:
                failure_detail += f" {error}"
            evidence_records.append({
                "text": "",
                "title": document.get("title", "Unknown source"),
                "url": source_url,
                "original_url": document.get("source_url", ""),
                "source_type": document.get("source_type", "Unclassified"),
                "publication_date": document.get("publication_date", ""),
                "source_role": source_role,
                "candidate_status": "rejected",
                "rejection_reason": (
                    f"Document collection status is "
                    f"{document.get('status', 'unknown')}.{failure_detail}"
                ).strip(),
            })
            continue

        text = document.get("text") or ""

        if not text.strip():
            source_url = (
                document.get("final_url")
                or document.get("source_url")
                or ""
            )
            evidence_records.append({
                "text": "",
                "title": document.get("title", "Unknown source"),
                "url": source_url,
                "original_url": document.get("source_url", ""),
                "source_type": document.get("source_type", "Unclassified"),
                "publication_date": document.get("publication_date", ""),
                "source_role": identify_source_role(document),
                "candidate_status": "rejected",
                "rejection_reason": "No document text was extracted.",
            })
            continue

        # Normalize whitespace.
        text = re.sub(r"\s+", " ", text).strip()
        assurance_statements = extract_embedded_assurance_statements(
            document,
            entity_name,
        )
        assurance_text = set()
        for item in assurance_statements:
            for sentence in split_into_sentences(
                str(item.get("statement") or "")
            ):
                assurance_text.add(
                    " ".join(sentence.split()).casefold()
                )

        # Split the document into approximate sentences.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        for sentence in sentences:

            sentence = sentence.strip()
            if " ".join(sentence.split()).casefold() in assurance_text:
                continue

            # Ignore fragments and unusually long text blocks.
            if len(sentence) < 45 or len(sentence) > 1000:
                continue

            sentence_lower = sentence.lower()

            # Keep sentences containing environmental keywords.
            if not any(
                term in sentence_lower
                for term in ENVIRONMENTAL_TERMS
            ):
                continue

            # Remove exact duplicate sentences.
            sentence_key = (
                str(document.get("final_url") or document.get("source_url") or "")
                .strip()
                .casefold(),
                sentence_lower,
            )

            if sentence_key in seen_sentences:
                continue

            seen_sentences.add(sentence_key)

            evidence_records.append({
                "text": sentence,
                "title": document.get(
                    "title",
                    "Unknown source"
                ),
                "url": document.get(
                    "final_url",
                    document.get("source_url", ""),
                ),
                "original_url": document.get(
                    "source_url",
                    ""
                ),
                "source_type": document.get(
                    "source_type",
                    "Unclassified"
                ),
                "publication_date": document.get(
                    "publication_date",
                    ""
                ),
                "source_role": identify_source_role(document),
            })
            source_role = evidence_records[-1]["source_role"]
            if source_role in {
                "external_regulatory_evidence",
                "configured_environmental_source",
                "independent_third_party_source",
                "external_source",
            }:
                evidence_records[-1]["candidate_status"] = "eligible"
                evidence_records[-1]["rejection_reason"] = None
            elif source_role == "company_source":
                evidence_records[-1]["candidate_status"] = "rejected"
                evidence_records[-1]["rejection_reason"] = (
                    "Company-owned statements are not independent external evidence."
                )
            elif source_role == "rehosted_report_unverified":
                evidence_records[-1]["candidate_status"] = "rejected"
                evidence_records[-1]["rejection_reason"] = (
                    "The report is hosted by a known rehosting platform; "
                    "independent provenance was not established."
                )
            elif source_role == "potential_company_source_unverified":
                evidence_records[-1]["candidate_status"] = "rejected"
                evidence_records[-1]["rejection_reason"] = (
                    "The domain resembles the company name, but company "
                    "ownership is not verified."
                )
            else:
                evidence_records[-1]["candidate_status"] = "rejected"
                evidence_records[-1]["rejection_reason"] = (
                    "The source is not classified as regulatory, configured, "
                    "or explicitly attributed independent evidence."
                )

        for statement in assurance_statements:
            sentence = statement["statement"]
            sentence_key = (
                str(statement.get("source_url") or "").strip().casefold(),
                sentence.casefold(),
            )
            if sentence_key in seen_sentences:
                continue
            seen_sentences.add(sentence_key)
            evidence_records.append({
                "text": sentence,
                "title": statement["source_title"],
                "url": statement["source_url"],
                "original_url": statement["original_source_url"],
                "source_type": statement["source_type"],
                "publication_date": statement["publication_date"] or "",
                "source_role": statement["source_role"],
                "candidate_status": "eligible",
                "rejection_reason": None,
                "publisher": statement["publisher"],
                "attribution_basis": statement["attribution_basis"],
            })

    return evidence_records


# ============================================================
# MATCH CLAIMS WITH COLLECTED EVIDENCE
# ============================================================

def analyze_claims(claims, evidence_records):
    """Match claims to evidence and retain source provenance."""

    eligible_records = [
        record
        for record in evidence_records
        if record.get("candidate_status") == "eligible"
    ]

    if not claims:
        print("\nNo claims are available for matching.")
        return []

    if not eligible_records:
        print("\nNo eligible external evidence sentences were extracted.")
        return []

    evidence_texts = [
        record["text"]
        for record in eligible_records
    ]

    print("\nLoading the Sentence Transformer model...")

    # Uses the existing evidence_matcher.py module.
    matches = match_evidence(
        claims,
        evidence_texts
    )

    results = []

    for claim, match in zip(claims, matches):

        matched_text = match["matched_evidence"]

        # Recover the source belonging to the matched sentence.
        source = next(
            (
                record
                for record in eligible_records
                if record["text"] == matched_text
            ),
            None,
        )

        similarity = float(match["similarity"])

        if similarity < SIMILARITY_THRESHOLD:
            status = "INSUFFICIENT MATCH"
            explanation = (
                "The best match is below the temporary "
                "similarity threshold. More relevant evidence "
                "is needed."
            )
        else:
            status = "REQUIRES REVIEW"
            explanation = (
                "The similarity threshold was reached, but "
                "the sentence must be checked to determine "
                "whether it supports, contradicts, or is "
                "unrelated to the claim."
            )

        results.append({
            "claim": claim,
            "matched_evidence": matched_text,
            "similarity": round(similarity, 4),
            "status": status,
            "explanation": explanation,
            "source": source,
        })

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(results):

    print("\n" + "=" * 75)
    print("ESG EVIDENCE MATCHING RESULTS")
    print("=" * 75)

    if not results:
        print("No matching results are available.")
        return

    for number, result in enumerate(results, start=1):

        print(f"\nCLAIM {number}")
        print("-" * 75)

        print("Claim:")
        print(result["claim"])

        print("\nBest-matching sentence:")
        print(result["matched_evidence"])

        print(
            f"\nSemantic similarity: "
            f"{result['similarity']:.4f}"
        )

        print(f"Evidence status: {result['status']}")

        print(f"Explanation: {result['explanation']}")

        source = result.get("source")

        if source:

            print(f"\nSource title: {source['title']}")
            print(f"Source type: {source['source_type']}")

            print(
                "Publication date: "
                f"{source['publication_date'] or 'Unknown'}"
            )

            print(f"Source URL: {source['url']}")

        print("-" * 75)

    insufficient = sum(
        result["status"] == "INSUFFICIENT MATCH"
        for result in results
    )

    requires_review = sum(
        result["status"] == "REQUIRES REVIEW"
        for result in results
    )

    print("\nSUMMARY")
    print("=" * 75)
    print(f"Claims analyzed: {len(results)}")
    print(f"Insufficient matches: {insufficient}")
    print(f"Matches requiring review: {requires_review}")

    print("\nIMPORTANT LIMITATIONS")
    print(
        "- Similarity measures textual relatedness, not truth."
    )
    print(
        "- A high similarity score does not prove a claim is supported."
    )
    print(
        "- A low similarity score does not prove greenwashing."
    )
    print(
        "- Collected sources must be checked for relevance, "
        "date, context, and reliability."
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 75)
    print("ADVANCED ESG GREENWASHING DETECTION")
    print("Collected Evidence Analysis")
    print("=" * 75)

    # Step 1: Check the sample report.
    if not SAMPLE_REPORT.exists():

        print("\nSample ESG report was not found:")
        print(SAMPLE_REPORT)

        print(
            "\nCheck that data/sample_esg_report.txt exists."
        )

        return

    # Step 2: Extract claims from the existing sample report.
    print("\nStep 1: Extracting sample ESG claims...")

    report_text = extract_text(SAMPLE_REPORT)

    claims, report_evidence = extract_claims_and_evidence(
        report_text
    )

    print(f"Claims extracted: {len(claims)}")
    print(f"Sample report evidence items: {len(report_evidence)}")

    if not claims:

        print(
            "\nNo claims were extracted. The sample report "
            "must contain the expected Environmental Claims: "
            "section with numbered items."
        )

        return

    # Step 3: Load the latest collected web documents.
    print("\nStep 2: Loading collected web documents...")

    try:
        collection = load_latest_collection()

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:

        print(f"\nUnable to load collection: {error}")
        return

    documents = collection.get("documents", [])

    if not isinstance(documents, list):

        print(
            "\nThe JSON documents field is not a list. "
            "Check automatic_collector.py output."
        )

        return

    print(f"Documents processed by collector: {len(documents)}")

    successful_count = sum(
        document.get("status") == "success"
        for document in documents
    )

    print(f"Successfully collected documents: {successful_count}")

    # Step 4: Extract environmental sentences.
    print("\nStep 3: Extracting potential evidence sentences...")

    evidence_records = extract_environmental_sentences(
        documents,
        entity_name=collection.get("search_name"),
    )

    print(
        f"Potential evidence sentences: {len(evidence_records)}"
    )

    if not evidence_records:

        print(
            "\nNo potential evidence sentences were found. "
            "Collect more documents or inspect the extraction rules."
        )

        return

    # Step 5: Match claims with evidence.
    print("\nStep 4: Matching claims with collected evidence...")

    try:
        results = analyze_claims(
            claims,
            evidence_records
        )

    except Exception as error:

        print(f"\nEvidence matching failed: {error}")
        print(
            "Check the Sentence Transformer installation "
            "and model availability."
        )

        return

          # Step 6: Save evidence analysis results.
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_file = output_dir / "evidence_analysis.json"

    saved_results = []

    for result in results:
        source = result.get("source") or {}

        saved_results.append({
            "claim": result["claim"],
            "matched_evidence": result["matched_evidence"],
            "similarity": result["similarity"],
            "status": result["status"],
            "explanation": result["explanation"],
            "source_title": source.get("title", ""),
            "source_url": source.get("url", ""),
            "source_type": source.get("source_type", ""),
            "publication_date": source.get("publication_date", ""),
        })

    saved_payload = {
        "results": saved_results,
        "candidate_review": [
            {
                "text": record.get("text", ""),
                "source_title": record.get("title", ""),
                "source_url": record.get("url", ""),
                "original_source_url": record.get("original_url", ""),
                "source_type": record.get("source_type", ""),
                "source_role": record.get("source_role", "unknown"),
                "candidate_status": record.get("candidate_status"),
                "rejection_reason": record.get("rejection_reason"),
            }
            for record in evidence_records
        ],
    }
    with analysis_file.open("w", encoding="utf-8") as file:
        json.dump(saved_payload, file, indent=4, ensure_ascii=False)

    print(f"\nEvidence analysis saved to: {analysis_file}")

    # Step 7: Display results.
    display_results(results)

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()