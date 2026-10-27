#!/usr/bin/env python3
"""Create blind human-review packets from a saved company analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_EVIDENCE_ROLES = {
    "external_source",
    "external_regulatory_evidence",
    "configured_environmental_source",
    "independent_third_party_source",
}
NLI_PACKET_COLUMNS = (
    "pair_id",
    "analysis_id",
    "company",
    "company_group",
    "report_id",
    "split_group",
    "claim_group",
    "claim",
    "claim_source_title",
    "claim_source_url",
    "claim_source_role",
    "claim_reporting_period",
    "claim_metric",
    "claim_value",
    "claim_unit",
    "claim_scope",
    "claim_baseline_year",
    "claim_organizational_boundary",
    "claim_methodology",
    "claim_performance_status",
    "evidence_source_title",
    "evidence_publisher",
    "evidence_source_url",
    "evidence_source_role",
    "evidence_excerpt",
    "evidence_reporting_period",
    "evidence_metric",
    "evidence_value",
    "evidence_unit",
    "evidence_scope",
    "evidence_baseline_year",
    "evidence_organizational_boundary",
    "evidence_methodology",
    "evidence_performance_status",
    "evidence_availability",
    "eligibility_decision",
    "metric_match",
    "unit_match",
    "period_match",
    "scope_match",
    "baseline_match",
    "organizational_boundary_match",
    "methodology_match",
    "performance_status_match",
    "reviewer_1_id",
    "reviewer_1_label",
    "reviewer_1_notes",
    "reviewer_2_id",
    "reviewer_2_label",
    "reviewer_2_notes",
    "adjudicated_label",
    "adjudicator_id",
    "review_status",
    "review_notes",
    "adjudication_notes",
    "uncertainty_reason",
)
RISK_PACKET_COLUMNS = (
    "annotation_id",
    "analysis_id",
    "claim_id",
    "company",
    "report_id",
    "split_group",
    "claim_group",
    "claim",
    "claim_source_title",
    "claim_source_url",
    "claim_source_role",
    "claim_reporting_period",
    "claim_metric",
    "claim_value",
    "claim_unit",
    "claim_scope",
    "claim_baseline_year",
    "claim_organizational_boundary",
    "claim_methodology",
    "claim_performance_status",
    "evidence_source_titles",
    "evidence_source_urls",
    "evidence_source_roles",
    "evidence_source_url",
    "evidence_source_role",
    "evidence_excerpt",
    "evidence_reporting_period",
    "evidence_metric",
    "evidence_value",
    "evidence_unit",
    "evidence_scope",
    "evidence_baseline_year",
    "evidence_organizational_boundary",
    "evidence_methodology",
    "evidence_performance_status",
    "evidence_availability",
    "compatibility_decision",
    "metric_match",
    "unit_match",
    "period_match",
    "scope_match",
    "baseline_match",
    "organizational_boundary_match",
    "methodology_match",
    "performance_status_match",
    "support_label",
    "reviewer_1_id",
    "reviewer_1_risk_label",
    "reviewer_1_support_label",
    "reviewer_1_notes",
    "reviewer_2_id",
    "reviewer_2_risk_label",
    "reviewer_2_support_label",
    "reviewer_2_notes",
    "label",
    "adjudicator_id",
    "review_status",
    "review_notes",
    "adjudication_notes",
    "uncertainty_reason",
    "review_confidence",
)


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).casefold().strip()


def pair_identifier(
    analysis_id: str,
    claim: str,
    evidence: str,
    evidence_url: str,
) -> str:
    identity = "\n".join(
        (
            analysis_id,
            normalize_text(claim),
            normalize_text(evidence),
            evidence_url.strip(),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _valid_http_url(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = urlparse(value.strip())
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
            and (parsed.port is None or 1 <= parsed.port <= 65535)
        )
    except ValueError:
        return False


def _blank_row(columns: tuple[str, ...]) -> dict[str, str]:
    return dict.fromkeys(columns, "")


def _analysis_claims(analysis: dict) -> dict[str, list[dict]]:
    claims: dict[str, list[dict]] = defaultdict(list)
    for item in analysis.get("company_claims") or []:
        if not isinstance(item, dict):
            continue
        if item.get("source_role") != "company_source":
            continue
        statement = item.get("statement")
        source_url = item.get("source_url") or item.get("original_source_url")
        if isinstance(statement, str) and statement.strip() and _valid_http_url(source_url):
            claims[normalize_text(statement)].append(item)
    return claims


def _eligible_matches(result: dict) -> list[dict]:
    matches = []
    for item in result.get("matches") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("source_role") or "").casefold()
        evidence = item.get("matched_evidence")
        source_url = item.get("source_url")
        if (
            role in ALLOWED_EVIDENCE_ROLES
            and isinstance(evidence, str)
            and evidence.strip()
            and _valid_http_url(source_url)
        ):
            matches.append(item)
    return matches


def _make_nli_packet(analysis: dict) -> tuple[list[dict[str, str]], dict]:
    analysis_id = str(analysis.get("analysis_id") or "")
    company = str(analysis.get("company") or "").strip()
    claims = _analysis_claims(analysis)
    rows = []
    seen = set()
    counts = defaultdict(int)

    for result in analysis.get("matching_results") or []:
        if not isinstance(result, dict):
            counts["invalid_match_records"] += 1
            continue
        claim_text = result.get("claim")
        source_claims = claims.get(normalize_text(claim_text), [])
        if len(source_claims) != 1:
            counts["claim_source_missing_or_ambiguous"] += 1
            continue
        source_claim = source_claims[0]
        matches = _eligible_matches(result)
        counts["candidate_pairs_seen"] += len(result.get("matches") or [])
        for match in matches:
            evidence = match["matched_evidence"].strip()
            evidence_url = match["source_url"].strip()
            pair_id = pair_identifier(
                analysis_id,
                claim_text,
                evidence,
                evidence_url,
            )
            if pair_id in seen:
                counts["duplicate_pairs"] += 1
                continue
            seen.add(pair_id)
            row = _blank_row(NLI_PACKET_COLUMNS)
            row.update({
                "pair_id": pair_id,
                "analysis_id": analysis_id,
                "company": company,
                "company_group": normalize_text(company),
                "report_id": analysis_id,
                "split_group": normalize_text(company),
                "claim_group": hashlib.sha256(
                    normalize_text(claim_text).encode("utf-8")
                ).hexdigest(),
                "claim": str(claim_text).strip(),
                "claim_source_title": str(source_claim.get("source_title") or ""),
                "claim_source_url": str(
                    source_claim.get("source_url")
                    or source_claim.get("original_source_url")
                    or ""
                ).strip(),
                "claim_source_role": "company_source",
                "evidence_source_title": str(match.get("source_title") or ""),
                "evidence_publisher": str(match.get("publisher") or ""),
                "evidence_source_url": evidence_url,
                "evidence_source_role": str(match.get("source_role") or ""),
                "evidence_excerpt": evidence,
                "evidence_availability": "candidate_pending_review",
                "eligibility_decision": "pending_review",
                "metric_match": "pending_review",
                "unit_match": "pending_review",
                "period_match": "pending_review",
                "scope_match": "pending_review",
                "baseline_match": "pending_review",
                "organizational_boundary_match": "pending_review",
                "methodology_match": "pending_review",
                "performance_status_match": "pending_review",
                "review_status": "pending",
            })
            rows.append(row)
            counts["nli_review_candidates"] += 1
    return rows, dict(counts)


def _make_risk_packet(analysis: dict) -> tuple[list[dict[str, str]], dict]:
    analysis_id = str(analysis.get("analysis_id") or "")
    company = str(analysis.get("company") or "").strip()
    claims = _analysis_claims(analysis)
    feature_by_claim = {
        str(item.get("claim_id")): item
        for item in analysis.get("features") or []
        if isinstance(item, dict) and item.get("claim_id") is not None
    }
    rows = []
    seen_claims = set()
    counts = defaultdict(int)

    for result in analysis.get("matching_results") or []:
        if not isinstance(result, dict):
            counts["invalid_match_records"] += 1
            continue
        claim_text = result.get("claim")
        normalized_claim = normalize_text(claim_text)
        source_claims = claims.get(normalized_claim, [])
        if len(source_claims) != 1:
            counts["claim_source_missing_or_ambiguous"] += 1
            continue
        source_claim = source_claims[0]
        claim_ids = {
            str(item.get("claim_id"))
            for item in analysis.get("features") or []
            if isinstance(item, dict)
            and normalize_text(item.get("claim")) == normalized_claim
            and item.get("claim_id") is not None
        }
        if len(claim_ids) != 1:
            counts["feature_record_missing_or_ambiguous"] += 1
            continue
        claim_id = next(iter(claim_ids))
        feature = feature_by_claim.get(claim_id, {})
        if feature.get("evidence_available") != 1:
            counts["no_scoring_eligible_evidence"] += 1
            continue

        matches = _eligible_matches(result)
        if not matches:
            counts["no_eligible_external_candidates"] += 1
            continue
        if claim_id in seen_claims:
            continue
        seen_claims.add(claim_id)

        urls = []
        titles = []
        roles = []
        excerpts = []
        seen_sources = set()
        for match in matches:
            source_url = match["source_url"].strip()
            evidence = match["matched_evidence"].strip()
            key = (source_url.casefold(), normalize_text(evidence))
            if key in seen_sources:
                continue
            seen_sources.add(key)
            urls.append(source_url)
            roles.append(str(match.get("source_role") or ""))
            titles.append(str(match.get("source_title") or ""))
            excerpts.append(
                f"[{len(excerpts) + 1}] {match.get('source_title') or source_url}\n"
                f"URL: {source_url}\n{evidence}"
            )
        if not urls:
            continue

        source_url = str(
            source_claim.get("source_url")
            or source_claim.get("original_source_url")
            or ""
        ).strip()
        annotation_id = hashlib.sha256(
            f"{analysis_id}\n{claim_id}".encode("utf-8")
        ).hexdigest()
        row = _blank_row(RISK_PACKET_COLUMNS)
        row.update({
            "annotation_id": annotation_id,
            "analysis_id": analysis_id,
            "claim_id": claim_id,
            "company": company,
            "report_id": analysis_id,
            "split_group": normalize_text(company),
            "claim_group": hashlib.sha256(
                normalized_claim.encode("utf-8")
            ).hexdigest(),
            "claim": str(claim_text).strip(),
            "claim_source_title": str(source_claim.get("source_title") or ""),
            "claim_source_url": source_url,
            "claim_source_role": "company_source",
            "evidence_source_titles": "|".join(titles),
            "evidence_source_urls": "|".join(urls),
            "evidence_source_roles": "|".join(roles),
            "evidence_source_url": urls[0],
            "evidence_source_role": roles[0],
            "evidence_excerpt": "\n\n".join(excerpts),
            "evidence_availability": "candidate_pending_review",
            "compatibility_decision": "pending_review",
            "metric_match": "pending_review",
            "unit_match": "pending_review",
            "period_match": "pending_review",
            "scope_match": "pending_review",
            "baseline_match": "pending_review",
            "organizational_boundary_match": "pending_review",
            "methodology_match": "pending_review",
            "performance_status_match": "pending_review",
            "review_status": "pending",
        })
        rows.append(row)
        counts["risk_review_candidates"] += 1
    return rows, dict(counts)


def prepare_review_packet(
    analysis_path: Path,
    output_path: Path,
    task: str,
) -> dict:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing review packet: {output_path}"
        )
    with analysis_path.open("r", encoding="utf-8") as file:
        analysis = json.load(file)
    if not isinstance(analysis, dict):
        raise ValueError("Saved analysis JSON must contain an object.")
    if not analysis.get("analysis_id") or not analysis.get("company"):
        raise ValueError("Saved analysis lacks analysis_id or company.")

    if task == "nli":
        rows, counts = _make_nli_packet(analysis)
        columns = NLI_PACKET_COLUMNS
    else:
        rows, counts = _make_risk_packet(analysis)
        columns = RISK_PACKET_COLUMNS

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "task": task,
        "analysis_id": analysis["analysis_id"],
        "packet": str(output_path),
        "rows": len(rows),
        "counts": counts,
        "blind_review": True,
        "contains_model_predictions_or_features": False,
        "limitations": [
            "Every exported candidate still requires human source eligibility, "
            "metric, unit, reporting period, baseline, organizational boundary, "
            "scope, methodology, and performance-status review.",
            "A pipeline match is a candidate, not validated evidence or a "
            "ground-truth label.",
            "Reviewers should complete labels independently before seeing "
            "model outputs or heuristic features.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a blind CSV review packet without model outputs."
    )
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", choices=("risk", "nli"), required=True)
    args = parser.parse_args(argv)
    try:
        report = prepare_review_packet(args.analysis, args.output, args.task)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
