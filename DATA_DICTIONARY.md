# Data dictionary

Fields as written by the current code. Source file in parentheses.

## Claim-level matching result (`online_evidence_matches.json`, `results[]`)

| Field | Type | Meaning |
|---|---|---|
| `claim` | string | Extracted ESG claim text. |
| `evidence_status` | string | `potentially_relevant_evidence_found`, `only_weak_matches_found`, or `no_sufficiently_relevant_match` (`online_evidence_matcher.py`). |
| `matches_found` | int | Number of matches kept (max 3). |
| `matches[].matched_evidence` | string | Candidate evidence statement. |
| `matches[].similarity` | float, 4 dp | Cosine similarity from the sentence-transformer. A relatedness measure, not truth. |
| `matches[].match_label` | string | `potentially_relevant` (≥0.55), `weak_match` (0.35–0.55), or `no_sufficiently_relevant_match` (<0.35). |
| `matches[].source_url`, `original_source_url` | string | Source URL and its pre-redirect URL. |
| `matches[].source_role` | string | Role from the source manager. Only external and independent roles are eligible for matching. |
| `matches[].entity_context` | string or null | Company/product the source is attributed to. |
| `matches[].attribution_basis` | string | How the source was attributed to the company. |

## NLI pair result (`online_contradiction_results.json`, `results[]`)

| Field | Type | Meaning |
|---|---|---|
| `claim`, `evidence`, `source_url` | string | Pair identity. |
| `nli_result` | string | `ENTAILMENT`, `NEUTRAL`, `CONTRADICTION`, or `UNAVAILABLE`. |
| `raw_nli_label` | string or null | Label as returned by the model. |
| `confidence` | float or null | Model score. Not calibrated. |
| `compatibility_status` | string | `COMPATIBLE`, `INCONCLUSIVE`, or `INCOMPATIBLE` (set by the measurement gate). |
| `scoring_eligible` | bool | `False` when the pair must not contribute a signal. |
| `nli_failed` | bool | `True` only when the model run itself failed (unusable output). Added by D1 fix. |
| `warning` | string | Human-readable caveat. |

A pair with `nli_failed: True` is **not** a completed outcome. It is never counted as a zero-count result.

## Feature row (`esg_features.csv`, built by `feature_builder.py`)

| Field | Type | Meaning |
|---|---|---|
| `claim_id` | int | 1-based index of the claim in the matching result. |
| `similarity` | float or empty | Maximum similarity over eligible matches. Empty when no eligible match. |
| `evidence_available` | 0/1 | 1 when at least one eligible match exists. |
| `nli_pairs_eligible` | int | Number of distinct eligible matched pairs for the claim (the denominator for coverage). |
| `nli_pairs_completed` | int | Number of those pairs with a completed NLI result. |
| `nli_coverage_complete` | 0/1 or empty | `1` only when every eligible pair has a completed NLI result. `0` means the claim is not scored (D2 policy). Empty when there are no eligible pairs. |
| `nli_entailment`, `nli_contradiction`, `nli_neutral` | int or empty | Counts over completed NLI pairs that belong to this claim's eligible matches (informational when coverage is incomplete). Empty when no completed NLI run exists. Failed (`UNAVAILABLE`) pairs are excluded from counts. |
| `percentage_difference`, `percentage_difference_missing`, `historical_change` | — | Always empty/`1` in the online workflow. Kept for schema compatibility. |

## Risk prediction row (`esg_risk_predictions.csv`, `risk_predictor.py`)

| Field | Type | Meaning |
|---|---|---|
| `risk_score` | float 0–100 or empty | Heuristic review score. Empty when not predicted. |
| `risk_score_status` | string | `HEURISTIC` or `NOT_PREDICTED`. |
| `risk_unavailable_reason` | string | Why not predicted: no eligible evidence; NLI produced no usable result; or only some eligible pairs were assessed (partial coverage). |
| `predicted_risk_class` | 0/1 or empty | `1` when score ≥ 50. Heuristic cutoff, not validated. |
| `demonstration_risk_category` | string | `LOW` (<35), `MEDIUM` (35–70), `HIGH` (≥70), or `NOT_PREDICTED`. |
| `demonstration_risk_probability` | empty | Intentionally empty. A heuristic score is not a probability. |

## Human review packet and labels

Defined by `evaluation/nli_annotation_template.csv` and `evaluation/risk_annotation_template.csv`. Packets omit model outputs, scores, and features. Labels are produced only by human reviewers (see `EVALUATION_PROTOCOL.md`).

## Pair-review workflow files (`evaluation/pair_review.py`)

Run folder layout (`<run>` is the `--out-dir` given to `make-packets`):

| Path | Columns or content | Visible to |
|---|---|---|
| `<run>\packets\packet_a_<reviewer>.csv`, `<run>\packets\packet_b_<reviewer>.csv` | `pair_id, analysis_id, company, claim, claim_source_title, claim_source_url, claim_publication_date, evidence, evidence_source_title, evidence_publisher, evidence_source_url, evidence_source_role, evidence_publication_date, label, reviewer_id, rationale` | that reviewer only |
| `<run>\packets\packet_build_report.json` | run counts and rejection reasons | coordinator |
| `<run>\coordinator_only\coordinator_manifest.csv` | `pair_id, analysis_id, company, claim_group, model_nli_status, model_nli_result` | coordinator only (never sent to reviewers) |
| `<run>\returned_<reviewer>.csv` | same columns as the packet, after labelling | coordinator |
| `<run>\recon\agreement_report.json` | agreement statistics and unfinished counts | coordinator |
| `<run>\recon\consensus_labels.csv` | `pair_id, final_label, label_source, reviewer_a_id, reviewer_b_id` | coordinator |
| `<run>\recon\adjudication_queue.csv` | `pair_id, claim, evidence, evidence_source_url, reviewer_a_id, label_a, rationale_a, reviewer_b_id, label_b, rationale_b, adjudicated_label, adjudicator_id, adjudication_rationale` | adjudicator |
| `<run>\recon\adjudication_expected_pairs.json` | `{"pair_ids": [...]}` listing every disagreement | coordinator (checked by finalize) |
| `<run>\adjudicated_ADJ.csv` | queue format, returned by the adjudicator | coordinator |
| `<run>\final_labels.csv` | `pair_id, final_label, label_source, reviewer_a_id, reviewer_b_id` | coordinator |
| `<results>\evaluation_results.json` | metrics (JSON, ASCII) | coordinator |

`model_nli_status` is `completed`, `unavailable` (saved result failed or was not eligible), or `not_run` (no saved NLI result for the pair). `label_source` is `agreed` or `adjudicated`. Label values: `SUPPORTING`, `CONTRADICTING`, `UNRELATED`, `UNCERTAIN`.
