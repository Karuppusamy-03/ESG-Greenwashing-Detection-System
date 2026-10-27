from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

from claim_extractor import extract_claims_and_evidence
from report_extractor import extract_text


@lru_cache(maxsize=1)
def _load_sentence_transformer_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def _dedupe_evidence_entries(items):
    seen = set()
    unique = []
    for item in items:
        if not isinstance(item, str):
            continue
        normalized = " ".join(item.split()).strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def match_evidence(claims, evidence):
    """Calculate semantic similarity between claims and evidence."""

    if not claims or not evidence:
        return []

    unique_claims = list(claims)
    unique_evidence = _dedupe_evidence_entries(evidence)
    if not unique_evidence:
        return []

    model = _load_sentence_transformer_model()

    claim_embeddings = model.encode(
        unique_claims,
        convert_to_tensor=True,
    )

    evidence_embeddings = model.encode(
        unique_evidence,
        convert_to_tensor=True,
    )

    similarity_matrix = util.cos_sim(
        claim_embeddings,
        evidence_embeddings,
    )

    results = []

    for i, claim in enumerate(unique_claims):
        best_index = int(similarity_matrix[i].argmax().item())
        similarity = float(similarity_matrix[i][best_index].item())

        results.append({
            "claim": claim,
            "matched_evidence": unique_evidence[best_index],
            "similarity": round(similarity, 4),
        })

    return results


if __name__ == "__main__":
    report_path = Path("data") / "sample_esg_report.txt"

    text = extract_text(report_path)
    claims, evidence = extract_claims_and_evidence(text)

    print("Loading Sentence Transformer model...")
    results = match_evidence(claims, evidence)

    print("\nEVIDENCE MATCHING RESULTS")
    print("=" * 70)

    for i, result in enumerate(results, start=1):
        print(f"\nClaim {i}:")
        print(result["claim"])

        print("\nMatched evidence:")
        print(result["matched_evidence"])

        print(f"\nSemantic similarity: {result['similarity']:.4f}")
        print("-" * 70)