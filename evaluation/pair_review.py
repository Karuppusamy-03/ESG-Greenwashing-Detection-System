#!/usr/bin/env python3
"""Two-reviewer human labelling workflow for claim-evidence pairs.

This module labels **evidence pairs only**: whether one exact eligible evidence
statement SUPPORTING, CONTRADICTING, UNRELATED to, or UNCERTAIN about one exact
company claim. A pair label is not a company-level greenwashing label, and one
CONTRADICTING pair is not proof of greenwashing.

Workflow (all steps write new files and refuse to overwrite existing ones):

1. make-packets : build one blinded packet per reviewer from a saved analysis,
                   plus a coordinator-only manifest that holds model outputs.
2. validate     : check a returned packet (structure, completeness, label set,
                   rationales, unchanged source text, reviewer identity).
3. reconcile    : compare two validated returns; write agreement statistics,
                   consensus labels and an adjudication queue.
4. finalize     : merge adjudicator decisions into the final label file.
5. evaluate     : metrics on final labels, with NLI predictions taken from the
                   coordinator manifest only.

Reviewers never receive the manifest, each other's packet, or any model output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.prepare_review_packet import (  # noqa: E402
    ALLOWED_EVIDENCE_ROLES,
    _analysis_claims,
    _eligible_matches,
    _valid_http_url,
    normalize_text,
    pair_identifier,
)

LABELS = ("SUPPORTING", "CONTRADICTING", "UNRELATED", "UNCERTAIN")
# Labels that must come with a written rationale.
RATIONALE_REQUIRED = {"CONTRADICTING", "UNCERTAIN"}
MIN_PAIRS_PER_REPORT = 30
MIN_MINORITY_LABEL = 5
MIN_COMPANIES = 10

PACKET_COLUMNS = (
    "pair_id",
    "analysis_id",
    "company",
    "claim",
    "claim_source_title",
    "claim_source_url",
    "claim_publication_date",
    "evidence",
    "evidence_source_title",
    "evidence_publisher",
    "evidence_source_url",
    "evidence_source_role",
    "evidence_publication_date",
    "label",
    "reviewer_id",
    "rationale",
)
# Columns a reviewer must not change; validate() compares them to the packet.
IMMUTABLE_COLUMNS = tuple(
    column for column in PACKET_COLUMNS
    if column not in {"label", "reviewer_id", "rationale"}
)
MANIFEST_COLUMNS = (
    "pair_id",
    "analysis_id",
    "company",
    "claim_group",
    "model_nli_status",
    "model_nli_result",
)
ADJUDICATION_COLUMNS = (
    "pair_id",
    "claim",
    "evidence",
    "evidence_source_url",
    "reviewer_a_id",
    "label_a",
    "rationale_a",
    "reviewer_b_id",
    "label_b",
    "rationale_b",
    "adjudicated_label",
    "adjudicator_id",
    "adjudication_rationale",
)


class ReviewError(ValueError):
    """Raised for any input that must not be used."""


# --------------------------------------------------------------------------- IO

def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ReviewError(f"File not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{k: (v or "") for k, v in row.items()} for row in reader]
        return list(reader.fieldnames or []), rows


def _write_new_csv(path: Path, columns, rows) -> None:
    if path.exists():
        raise ReviewError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel on Windows opens the file with correct characters.
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _write_new_json(path: Path, data: dict) -> None:
    if path.exists():
        raise ReviewError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_analysis(path: Path) -> dict:
    if not path.exists():
        raise ReviewError(f"Saved analysis not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        analysis = json.load(handle)
    if not isinstance(analysis, dict):
        raise ReviewError("Saved analysis must be a JSON object.")
    if not str(analysis.get("analysis_id") or "").strip():
        raise ReviewError("Saved analysis has no analysis_id; it cannot be traced.")
    if not str(analysis.get("company") or "").strip():
        raise ReviewError("Saved analysis has no company name.")
    return analysis


# ------------------------------------------------------------ pair construction

def _nli_lookup(analysis: dict) -> dict[tuple[str, str, str], dict]:
    lookup: dict[tuple[str, str, str], dict] = {}
    for item in analysis.get("nli_results") or []:
        if not isinstance(item, dict):
            continue
        key = (
            normalize_text(item.get("claim")),
            str(item.get("evidence") or "").strip(),
            str(item.get("source_url") or "").strip(),
        )
        lookup[key] = item
    return lookup


def _model_status(item: dict | None) -> tuple[str, str]:
    if item is None:
        return "not_run", ""
    result = str(item.get("nli_result") or "")
    if item.get("scoring_eligible") is False or result.upper() == "UNAVAILABLE":
        return "unavailable", result
    if result in {"ENTAILMENT", "CONTRADICTION", "NEUTRAL"}:
        return "completed", result
    return "unavailable", result


def build_pairs(analysis: dict) -> tuple[list[dict], dict]:
    """Return one record per distinct eligible claim-evidence pair.

    Only pairs whose claim maps to exactly one company-source statement with a
    valid URL, and whose evidence has an allowed external role and a valid URL,
    are kept. Every rejection is counted with its reason.
    """
    analysis_id = str(analysis["analysis_id"]).strip()
    company = str(analysis["company"]).strip()
    claims = _analysis_claims(analysis)
    nli = _nli_lookup(analysis)
    pairs: list[dict] = []
    seen: set[str] = set()
    rejected: Counter = Counter()

    for result in analysis.get("matching_results") or []:
        if not isinstance(result, dict):
            rejected["invalid_match_record"] += 1
            continue
        claim_text = str(result.get("claim") or "").strip()
        sources = claims.get(normalize_text(claim_text), [])
        if not claim_text:
            rejected["empty_claim"] += 1
            continue
        if len(sources) != 1:
            rejected["claim_source_missing_or_ambiguous"] += 1
            continue
        source = sources[0]
        for match in result.get("matches") or []:
            if not isinstance(match, dict):
                rejected["invalid_match"] += 1
                continue
            role = str(match.get("source_role") or "")
            evidence = str(match.get("matched_evidence") or "").strip()
            url = str(match.get("source_url") or "").strip()
            if not evidence:
                rejected["empty_evidence"] += 1
                continue
            if not _valid_http_url(url):
                rejected["evidence_url_missing_or_invalid"] += 1
                continue
            if role.casefold() not in ALLOWED_EVIDENCE_ROLES:
                rejected["evidence_role_not_eligible"] += 1
                continue
            pair_id = pair_identifier(analysis_id, claim_text, evidence, url)
            if pair_id in seen:
                rejected["duplicate_pair"] += 1
                continue
            seen.add(pair_id)
            status, model_result = _model_status(
                nli.get((normalize_text(claim_text), evidence, url))
            )
            pairs.append({
                "pair_id": pair_id,
                "analysis_id": analysis_id,
                "company": company,
                "claim_group": hashlib.sha256(
                    normalize_text(claim_text).encode("utf-8")
                ).hexdigest(),
                "claim": claim_text,
                "claim_source_title": str(source.get("source_title") or ""),
                "claim_source_url": str(
                    source.get("source_url") or source.get("original_source_url") or ""
                ).strip(),
                "claim_publication_date": str(source.get("publication_date") or ""),
                "evidence": evidence,
                "evidence_source_title": str(match.get("source_title") or ""),
                "evidence_publisher": str(match.get("publisher") or ""),
                "evidence_source_url": url,
                "evidence_source_role": role,
                "evidence_publication_date": str(match.get("publication_date") or ""),
                "model_nli_status": status,
                "model_nli_result": model_result,
            })
            rejected["accepted"] += 1
    return pairs, dict(rejected)


# ------------------------------------------------------------- read-only check

def inspect_analysis(analysis_path: Path) -> dict:
    """Read-only summary of whether a saved analysis can produce review packets.

    Writes nothing. Use it before make-packets to confirm the record is real,
    has eligible evidence and valid URLs, and shows NLI status per pair.
    """
    analysis = load_analysis(analysis_path)
    claims = [c for c in (analysis.get("company_claims") or []) if isinstance(c, dict)]
    roles = Counter(str(c.get("source_role") or "missing") for c in claims)
    pairs, rejected = build_pairs(analysis)
    return {
        "file": analysis_path.name,
        "analysis_id": analysis["analysis_id"],
        "company": analysis["company"],
        "company_claims_total": len(claims),
        "company_claims_by_source_role": dict(roles),
        "matching_results": len(analysis.get("matching_results") or []),
        "nli_results": len(analysis.get("nli_results") or []),
        "pairs_accepted_for_review": len(pairs),
        "pairs_by_model_nli_status": dict(Counter(p["model_nli_status"] for p in pairs)),
        "rejections": {k: v for k, v in rejected.items() if k != "accepted"},
        "ready_for_packets": len(pairs) > 0,
        "note": (
            "Only company_source claims with valid URLs enter packets. "
            "Pairs with model_nli_status other than 'completed' are still reviewed; "
            "their model columns are kept only in the coordinator manifest."
        ),
    }


# ------------------------------------------------------------------- step 1

def make_packets(analysis_path: Path, out_dir: Path,
                 reviewer_a: str, reviewer_b: str) -> dict:
    reviewer_a, reviewer_b = reviewer_a.strip(), reviewer_b.strip()
    if not reviewer_a or not reviewer_b:
        raise ReviewError("Two reviewer identifiers are required.")
    if reviewer_a.casefold() == reviewer_b.casefold():
        raise ReviewError("The two reviewers must be different people.")
    analysis = load_analysis(analysis_path)
    pairs, rejected = build_pairs(analysis)
    if not pairs:
        raise ReviewError(
            "No eligible claim-evidence pairs were found in this analysis. "
            f"Rejection counts: {rejected}"
        )
    packets_dir = out_dir / "packets"
    coord_dir = out_dir / "coordinator_only"
    if (packets_dir / "packet_build_report.json").exists() or (coord_dir / "coordinator_manifest.csv").exists():
        raise ReviewError(f"A packet run already exists in {out_dir}; choose a new run folder.")

    def packet_rows():
        for pair in pairs:
            row = {c: pair.get(c, "") for c in PACKET_COLUMNS}
            row.update({"label": "", "reviewer_id": "", "rationale": ""})
            yield row

    # Coordinator-only manifest: model outputs and grouping, never distributed.
    # Reviewers receive only packets/. The manifest lives in coordinator_only/.
    _write_new_csv(coord_dir / "coordinator_manifest.csv", MANIFEST_COLUMNS, pairs)
    for name, reviewer in (("a", reviewer_a), ("b", reviewer_b)):
        _write_new_csv(packets_dir / f"packet_{name}_{_slug(reviewer)}.csv",
                       PACKET_COLUMNS, list(packet_rows()))
    report = {
        "analysis_id": analysis["analysis_id"],
        "company": analysis["company"],
        "pairs_in_packet": len(pairs),
        "reviewers": [reviewer_a, reviewer_b],
        "rejected_or_skipped": rejected,
        "note": (
            "Give each reviewer only their own file from packets/. "
            "coordinator_only/ contains model outputs and must not be sent to reviewers."
        ),
    }
    _write_new_json(packets_dir / "packet_build_report.json", report)
    return report


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "reviewer"


# ------------------------------------------------------------------- step 2

def validate_returned(packet_path: Path, returned_path: Path) -> dict:
    """Validate a returned packet. Raises ReviewError on any hard failure.

    Returns counts. Blank labels are reported as incomplete, not as errors, so a
    reviewer can save partial work.
    """
    packet_cols, packet_rows = _read_csv(packet_path)
    if tuple(packet_cols) != PACKET_COLUMNS:
        raise ReviewError("Packet columns do not match the expected schema.")
    returned_cols, returned = _read_csv(returned_path)
    if tuple(returned_cols) != PACKET_COLUMNS:
        raise ReviewError(
            "Returned file columns were changed. Return the packet unchanged apart "
            "from label, reviewer_id and rationale."
        )
    by_id = {row["pair_id"]: row for row in packet_rows}
    seen: set[str] = set()
    reviewer_ids: set[str] = set()
    counts: Counter = Counter()
    for row in returned:
        pid = row["pair_id"]
        if pid in seen:
            raise ReviewError(f"Duplicate pair_id in returned file: {pid}")
        seen.add(pid)
        if pid not in by_id:
            raise ReviewError(f"pair_id not in this packet (added or altered row): {pid}")
        original = by_id[pid]
        for column in IMMUTABLE_COLUMNS:
            if row[column] != original[column]:
                raise ReviewError(
                    f"Source text was changed for pair {pid} in column '{column}'. "
                    "Reviewers must not edit claims, evidence or URLs."
                )
        label = row["label"].strip().upper()
        if not label:
            counts["unlabelled"] += 1
            continue
        if label not in LABELS:
            raise ReviewError(
                f"Invalid label '{row['label']}' for pair {pid}. Allowed: {', '.join(LABELS)}."
            )
        if not row["reviewer_id"].strip():
            raise ReviewError(f"Labelled pair {pid} has no reviewer_id.")
        if label in RATIONALE_REQUIRED and not row["rationale"].strip():
            raise ReviewError(f"Label {label} for pair {pid} requires a rationale.")
        reviewer_ids.add(row["reviewer_id"].strip())
        counts[label] += 1
    missing = set(by_id) - seen
    if missing:
        raise ReviewError(
            f"{len(missing)} packet pair(s) are missing from the returned file. "
            "Return every row, leaving unfinished labels blank."
        )
    if len(reviewer_ids) > 1:
        raise ReviewError("One returned file contains more than one reviewer_id.")
    counts["rows"] = len(returned)
    counts["complete"] = len(returned) - counts["unlabelled"]
    counts["reviewer_ids"] = sorted(reviewer_ids)
    return dict(counts)


# ------------------------------------------------------------------- step 3

def _load_labels(packet_path: Path, returned_path: Path) -> dict[str, dict]:
    validate_returned(packet_path, returned_path)
    _, rows = _read_csv(returned_path)
    out = {}
    for row in rows:
        label = row["label"].strip().upper()
        if label:
            out[row["pair_id"]] = {
                "label": label,
                "reviewer_id": row["reviewer_id"].strip(),
                "rationale": row["rationale"].strip(),
            }
    return out


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1 - expected)


def reconcile(packet_a: Path, returned_a: Path, packet_b: Path, returned_b: Path,
              manifest_path: Path, out_dir: Path) -> dict:
    labels_a = _load_labels(packet_a, returned_a)
    labels_b = _load_labels(packet_b, returned_b)
    ids_a = {r["reviewer_id"] for r in labels_a.values()}
    ids_b = {r["reviewer_id"] for r in labels_b.values()}
    if ids_a and ids_b and ids_a & ids_b:
        raise ReviewError("Both returned files list the same reviewer_id; reviewers must be independent.")
    _, packet_rows = _read_csv(packet_a)
    _, packet_rows_b = _read_csv(packet_b)
    if {r["pair_id"] for r in packet_rows} != {r["pair_id"] for r in packet_rows_b}:
        raise ReviewError("The two packets do not contain the same pairs.")
    _, manifest = _read_csv(manifest_path)
    manifest_ids = {r["pair_id"] for r in manifest}
    if set(r["pair_id"] for r in packet_rows) - manifest_ids:
        raise ReviewError("Manifest does not cover every packet pair; wrong manifest or analysis.")

    consensus, queue = [], []
    both_pairs = []
    texts = {r["pair_id"]: r for r in packet_rows}
    for pid in (r["pair_id"] for r in packet_rows):
        a, b = labels_a.get(pid), labels_b.get(pid)
        if a and b:
            both_pairs.append((a["label"], b["label"]))
            if a["label"] == b["label"]:
                consensus.append({
                    "pair_id": pid, "final_label": a["label"],
                    "label_source": "agreed",
                    "reviewer_a_id": a["reviewer_id"], "reviewer_b_id": b["reviewer_id"],
                })
                continue
        else:
            # Not labelled by both reviewers: stays open, not in the metrics.
            continue
        queue.append({
            "pair_id": pid,
            "claim": texts[pid]["claim"],
            "evidence": texts[pid]["evidence"],
            "evidence_source_url": texts[pid]["evidence_source_url"],
            "reviewer_a_id": a["reviewer_id"],
            "label_a": a["label"], "rationale_a": a["rationale"],
            "reviewer_b_id": b["reviewer_id"],
            "label_b": b["label"], "rationale_b": b["rationale"],
            "adjudicated_label": "", "adjudicator_id": "", "adjudication_rationale": "",
        })

    unfinished = {
        "pairs_labelled_by_a_only": len(set(labels_a) - set(labels_b)),
        "pairs_labelled_by_b_only": len(set(labels_b) - set(labels_a)),
        "pairs_labelled_by_neither": len(set(texts) - set(labels_a) - set(labels_b)),
    }
    kappa = cohen_kappa([x for x, _ in both_pairs], [y for _, y in both_pairs])
    agreement = (
        sum(x == y for x, y in both_pairs) / len(both_pairs) if both_pairs else None
    )
    report = {
        "pairs_in_packet": len(texts),
        "pairs_labelled_by_both": len(both_pairs),
        "percent_agreement": None if agreement is None else round(agreement, 4),
        "cohens_kappa": None if kappa is None else round(kappa, 4),
        "disagreements_for_adjudication": len(queue),
        "agreed_pairs": len(consensus),
        "unfinished": unfinished,
        "label_pairs_counts": dict(Counter(f"{x}|{y}" for x, y in both_pairs)),
        "note": (
            "Agreement statistics describe reviewer consistency only. "
            "They are not measures of model accuracy."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_new_csv(out_dir / "consensus_labels.csv",
                   ("pair_id", "final_label", "label_source", "reviewer_a_id", "reviewer_b_id"),
                   consensus)
    _write_new_csv(out_dir / "adjudication_queue.csv", ADJUDICATION_COLUMNS, queue)
    _write_new_json(out_dir / "agreement_report.json", report)
    _write_new_json(out_dir / "adjudication_expected_pairs.json",
                    {"pair_ids": sorted(row["pair_id"] for row in queue)})
    return report


# ------------------------------------------------------------------- step 4

def finalize(queue_path: Path, consensus_path: Path, manifest_path: Path,
             out_path: Path) -> dict:
    """Merge adjudicator decisions with agreed labels into final_labels.csv."""
    _, queue = _read_csv(queue_path)
    _, consensus = _read_csv(consensus_path)
    _, manifest = _read_csv(manifest_path)
    manifest_ids = {r["pair_id"] for r in manifest}
    expected_path = consensus_path.parent / "adjudication_expected_pairs.json"
    if not expected_path.exists():
        raise ReviewError(f"Missing {expected_path.name}; run reconcile first.")
    with expected_path.open("r", encoding="utf-8") as handle:
        expected = set(json.load(handle)["pair_ids"])
    queue_ids = [row["pair_id"] for row in queue]
    if len(queue_ids) != len(set(queue_ids)):
        raise ReviewError("Adjudication file contains duplicate pair_id rows.")
    if set(queue_ids) - expected:
        raise ReviewError("Adjudication file contains pairs that were not disagreements.")
    if expected - set(queue_ids):
        raise ReviewError(
            f"{len(expected - set(queue_ids))} disagreement(s) were deleted from the adjudication file. "
            "Return every row from adjudication_queue.csv."
        )
    final, problems = [], []
    seen = set()
    for row in queue:
        pid = row["pair_id"]
        if pid in seen:
            raise ReviewError(f"Duplicate adjudication row: {pid}")
        seen.add(pid)
        if pid not in manifest_ids:
            raise ReviewError(f"Adjudicated pair is not in the manifest: {pid}")
        label = row["adjudicated_label"].strip().upper()
        if not label:
            problems.append(pid)
            continue
        if label not in LABELS:
            raise ReviewError(f"Invalid adjudicated label for {pid}: {row['adjudicated_label']}")
        adjudicator = row["adjudicator_id"].strip()
        if not adjudicator or not row["adjudication_rationale"].strip():
            raise ReviewError(f"Adjudication for {pid} needs adjudicator_id and a rationale.")
        if adjudicator.casefold() in {row["reviewer_a_id"].casefold(), row["reviewer_b_id"].casefold()}:
            raise ReviewError(
                f"Adjudicator for {pid} is one of the two original reviewers; "
                "adjudication must be done by a third person."
            )
        final.append({
            "pair_id": pid, "final_label": label, "label_source": "adjudicated",
            "reviewer_a_id": row["reviewer_a_id"], "reviewer_b_id": row["reviewer_b_id"],
        })
    if problems:
        raise ReviewError(
            f"{len(problems)} disagreement(s) still have no adjudicated_label; "
            "finalize refuses to continue with open items."
        )
    consensus_ids = [row["pair_id"] for row in consensus]
    if len(consensus_ids) != len(set(consensus_ids)):
        raise ReviewError("Consensus file contains duplicate pair_id rows.")
    for row in consensus:
        if row["pair_id"] in seen:
            raise ReviewError(f"Pair appears in both consensus and adjudication: {row['pair_id']}")
        final.append(row)
    report_path = consensus_path.parent / "agreement_report.json"
    unfinished = 0
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as handle:
            unfinished = sum(json.load(handle)["unfinished"].values())
    _write_new_csv(out_path,
                   ("pair_id", "final_label", "label_source", "reviewer_a_id", "reviewer_b_id"),
                   final)
    return {
        "final_labels": len(final),
        "adjudicated": len(seen),
        "pairs_not_labelled_by_both_reviewers_excluded": unfinished,
    }


# ------------------------------------------------------------------- step 5

def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": _r(precision), "recall": _r(recall), "f1": _r(f1)}


def _r(value):
    return None if value is None else round(value, 4)


def _load_runs(run_dirs: list[Path]) -> tuple[list[dict], dict]:
    """Read final_labels.csv and coordinator_only/coordinator_manifest.csv per run."""
    final_rows, manifest = [], {}
    for run in run_dirs:
        _, rows = _read_csv(run / "final_labels.csv")
        _, mani = _read_csv(run / "coordinator_only" / "coordinator_manifest.csv")
        for row in mani:
            if row["pair_id"] in manifest:
                raise ReviewError(f"pair_id appears in more than one run: {row['pair_id']}")
            manifest[row["pair_id"]] = row
        final_rows.extend(rows)
    ids = [r["pair_id"] for r in final_rows]
    if len(ids) != len(set(ids)):
        raise ReviewError("The same pair is labelled in more than one run.")
    return final_rows, manifest


def evaluate(final_path: Path | None = None, manifest_path: Path | None = None,
             out_path: Path | None = None, run_dirs: list[Path] | None = None) -> dict:
    """Metrics on final pair labels versus saved NLI outputs.

    Scope: evidence-pair agreement between human labels and the NLI model. This
    is not company-level greenwashing accuracy and not risk-prediction accuracy.
    """
    if run_dirs:
        final, by_pair = _load_runs(run_dirs)
    else:
        _, final = _read_csv(final_path)
        _, manifest = _read_csv(manifest_path)
        by_pair = {r["pair_id"]: r for r in manifest}
    labels = []
    for row in final:
        pid = row["pair_id"]
        if pid not in by_pair:
            raise ReviewError(f"Final label has no manifest record: {pid}")
        if row["final_label"] not in LABELS:
            raise ReviewError(f"Invalid final label: {row['final_label']}")
        labels.append((row["final_label"], by_pair[pid]))

    distribution = Counter(label for label, _ in labels)
    companies = {m["company"] for _, m in labels}
    nli_coverage = Counter(m["model_nli_status"] for _, m in labels)
    completed = [(lab, m["model_nli_result"]) for lab, m in labels
                 if m["model_nli_status"] == "completed" and lab != "UNCERTAIN"]
    confusion = {lab: Counter() for lab in ("SUPPORTING", "CONTRADICTING", "UNRELATED")}
    for lab, nli in completed:
        confusion[lab][nli] += 1

    def binary(target_label: str, predicted_nli: str) -> dict:
        tp = sum(1 for lab, nli in completed if lab == target_label and nli == predicted_nli)
        fp = sum(1 for lab, nli in completed if lab != target_label and nli == predicted_nli)
        fn = sum(1 for lab, nli in completed if lab == target_label and nli != predicted_nli)
        return _prf(tp, fp, fn)

    n_scored = len(completed)
    minority = min((distribution[l] for l in ("SUPPORTING", "CONTRADICTING")), default=0)
    if (len(labels) < MIN_PAIRS_PER_REPORT or minority < MIN_MINORITY_LABEL
            or len(companies) < MIN_COMPANIES):
        status = "INSUFFICIENT_DATA"
    else:
        status = "REPORTABLE_WITH_CAVEATS"

    result = {
        "runs": len(run_dirs) if run_dirs else 1,
        "scope": "evidence-pair labels versus saved NLI outputs; not company-level greenwashing",
        "reporting_status": status,
        "reporting_requirements": {
            "min_labelled_pairs": MIN_PAIRS_PER_REPORT,
            "min_pairs_per_supporting_and_contradicting_class": MIN_MINORITY_LABEL,
            "min_companies": MIN_COMPANIES,
        },
        "sample_size_final_labels": len(labels),
        "companies": len(companies),
        "class_distribution": {l: distribution.get(l, 0) for l in LABELS},
        "nli_coverage": {k: nli_coverage.get(k, 0)
                         for k in ("completed", "unavailable", "not_run")},
        "pairs_scored_against_nli": n_scored,
        "confusion_human_by_nli": {
            lab: {nli: confusion[lab].get(nli, 0)
                  for nli in ("ENTAILMENT", "NEUTRAL", "CONTRADICTION")}
            for lab in confusion
        },
        "metrics_supporting_vs_nli_entailment": binary("SUPPORTING", "ENTAILMENT"),
        "metrics_contradicting_vs_nli_contradiction": binary("CONTRADICTING", "CONTRADICTION"),
        "not_computed": [
            "company-level greenwashing accuracy (no company-level ground truth)",
            "risk-prediction accuracy (needs a separate reviewed risk protocol)",
            "calibration (needs a reviewed sample large enough to bin confidences)",
            "population generalization (sample is one or more analyses only)",
            "bootstrap intervals (need at least 10 independent companies per label)",
        ],
    }
    if out_path is not None:
        _write_new_json(out_path, result)
    return result


# ------------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-packets")
    p.add_argument("--analysis", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--reviewer-a", required=True)
    p.add_argument("--reviewer-b", required=True)

    p = sub.add_parser("validate")
    p.add_argument("--packet", required=True, type=Path)
    p.add_argument("--returned", required=True, type=Path)

    p = sub.add_parser("reconcile")
    p.add_argument("--packet-a", required=True, type=Path)
    p.add_argument("--returned-a", required=True, type=Path)
    p.add_argument("--packet-b", required=True, type=Path)
    p.add_argument("--returned-b", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)

    p = sub.add_parser("finalize")
    p.add_argument("--queue", required=True, type=Path)
    p.add_argument("--consensus", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("evaluate")
    p.add_argument("--run-dir", type=Path, action="append", default=None,
                   help="Run folder containing final_labels.csv and coordinator_only/. Repeat for several runs.")
    p.add_argument("--final", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("inspect")
    p.add_argument("--analysis", required=True, type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            out = inspect_analysis(args.analysis)
        elif args.command == "make-packets":
            out = make_packets(args.analysis, args.out_dir, args.reviewer_a, args.reviewer_b)
        elif args.command == "validate":
            out = validate_returned(args.packet, args.returned)
        elif args.command == "reconcile":
            out = reconcile(args.packet_a, args.returned_a, args.packet_b,
                            args.returned_b, args.manifest, args.out_dir)
        elif args.command == "finalize":
            out = finalize(args.queue, args.consensus, args.manifest, args.out)
        else:
            if args.run_dir:
                out = evaluate(out_path=args.out, run_dirs=args.run_dir)
            elif args.final and args.manifest:
                out = evaluate(args.final, args.manifest, args.out)
            else:
                raise ReviewError("Give --run-dir (one or more) or both --final and --manifest.")
    except (ReviewError, FileExistsError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    # ASCII-only output: Windows consoles may not render non-ASCII company names.
    print(json.dumps(out, indent=2, sort_keys=True, default=str, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
