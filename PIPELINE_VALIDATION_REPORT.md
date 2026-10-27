# Pipeline validation report

Scope: `ESG-Greenwashing-Detection-System-main.zip` (source only), with fixes applied in this pass.

## What the ZIP contains

- Present: source, tests, evaluation scripts, annotation templates, existing audit documents.
- Absent: `data/` (SQLite database and collected documents), `output/` (saved analyses, match and NLI JSON, feature and prediction CSVs), `evaluation/risk_labels.csv`, `evaluation/nli_labels.csv`, and any trained model artifact.

Consequence: no saved company analysis can be replayed, and no human-reviewed data exists in the project.

## Verification results

| Run | Result |
|---|---|
| Baseline, original code, after installing dependencies | 215 run, OK, 1 skipped |
| After D1 fix | 225 run, OK, 1 skipped |
| After D2 policy and regression tests (current) | **233 run, OK, 1 skipped** |
| Compile of all project `.py` files | OK |
| Imports of all listed dependencies | OK |
| `streamlit run app.py --server.headless true` | HTTP 200 |

All unit tests use mocked search and a fake NLI classifier. The real NLI model could not be downloaded in this environment, and live sources were refused, so neither was exercised.

## Stage-by-stage trace

| # | Stage | Module(s) | Status | Evidence |
|---|---|---|---|---|
| 1 | Collection, source provenance | `automatic_collector.py`, `web_search.py`, `website_extractor.py` | NOT EVALUATED (live) | Offline tests pass with mocked results. Live access refused by the sandbox. |
| 2 | Claim extraction | `online_claim_extractor.py`, `claim_extractor.py` | NOT EVALUATED (no real reports) | Offline fixture tests pass. |
| 3 | Company identification | `source_manager.py`, company-identity code | PASS (offline) | `test_company_identity`. |
| 4 | Evidence eligibility and matching | `online_evidence_matcher.py` | PASS (offline) | Field names match the NLI detector and feature builder (`claim`, `matched_evidence`, `source_url`, `scoring_eligible`). Thresholds unchanged. |
| 5 | NLI classification | `contradiction_detector.py` | **D1 fixed** | Failed model outputs are flagged `nli_failed` and `scoring_eligible: False`. Real model not evaluated. |
| 6 | Feature construction | `feature_builder.py` | **D1 and D2 fixed** | Counts only pairs that belong to eligible matches. Adds coverage columns. |
| 7 | Risk prediction | `risk_predictor.py` | **D1 and D2 guarded** | Unassessed states are never scored. |
| 8 | Timeline analysis | `timeline_analyzer.py` | PASS (offline tests) | Existing comparability tests pass. Not re-audited line by line. |
| 9 | Persistence | `database.py`, `analysis_serialization.py` | PASS (offline tests) | `test_analysis_json_persistence`. |
| 10 | UI | `app.py` | PASS (startup only) | HTTP 200. `app.py` is byte-identical to the original; the sidebar is unchanged. Saved-analysis views need `output/`. |

The UI consumes the feature and prediction functions directly (`app.py` calls `build_features_from_results` and `predict_risk_from_features`), so the fixes apply in the app.

## Defects

### D1: failed NLI run scored as a completed zero-count outcome — FIXED

- Reproduced: a claim with one eligible match and a failed pair scored `HEURISTIC`, 10.0, LOW.
- Expected: `NOT_PREDICTED`, reason "Evidence was found, but NLI verification did not produce a usable result".
- Cause: `contradiction_detector` built failed results as `{**pair, "nli_result": "UNAVAILABLE"}`. The pair carried `scoring_eligible: True`. `feature_builder` then counted the row as a completed run with counts `0, 0, 0`.
- Fix: failed outputs carry `scoring_eligible: False` and `nli_failed: True`. The feature builder excludes any result whose `nli_result` is `UNAVAILABLE` from the NLI counts, whatever its flags. It does not exclude the matched evidence.
- Rejected first attempt: marking failed rows only as ineligible removed the evidence, so the claim was reported as having no evidence. Six existing tests failed. The separate `nli_failed` flag distinguishes NLI failures from eligibility exclusions (weak, inconclusive, incompatible pairs), which still remove the match.
- Legacy analyses: saved rows with `UNAVAILABLE` and `scoring_eligible: True` (from before the fix) are also excluded, because the check uses `nli_result`. Test: `LegacySavedFailedRowsTests`.

### D2: partial NLI coverage scored as complete — FIXED (policy)

- Reproduced on the original code: a claim with two eligible matches, only one of which had a completed NLI result, scored `HEURISTIC`, **94.4, HIGH**. The unevaluated match contributed nothing and was not disclosed.
- Policy: a claim is assessed only when every eligible matched evidence pair has a completed NLI result. Otherwise it is `NOT_PREDICTED`, with `risk_unavailable_reason` set to `NLI_PARTIAL_REASON`.
- Why conservative: scoring on a subset would present the score as a verdict on all matched evidence. The score is a review aid, so an unscored claim with its evidence visible is the safer output.
- Implementation:
  - `feature_builder.py`: counts only pairs whose `(claim, evidence, source_url)` key belongs to an eligible match. Adds `nli_pairs_eligible`, `nli_pairs_completed`, `nli_coverage_complete` (1, 0, or empty when there are no eligible matches). Counts are still reported for transparency.
  - `risk_predictor.py`: refuses to score when `nli_coverage_complete == 0`. Zero coverage keeps the existing "NLI did not produce a usable result" reason; partial coverage gets `NLI_PARTIAL_REASON`.
- Evidence is never removed by this policy. Similarity still uses all eligible matches.
- Compatibility: an existing test fixture had NLI rows without `evidence` or `source_url`, so they could not be attributed to any match. The fixture now carries that identity (`tests/test_offline_workflows.py`, `test_risk_prediction_and_explanation_match_the_heuristic`). Real NLI rows always carry it.
- Legacy CSVs without the coverage column keep their previous behaviour. They are not rescored automatically; see `KNOWN_LIMITATIONS.md`.

## Files changed in this pass

- `feature_builder.py` (D2 coverage columns and counting rule; `_pair_key`)
- `risk_predictor.py` (D2 guard, `NLI_PARTIAL_REASON`, `nli_coverage_incomplete`)
- `contradiction_detector.py` (D1 flags; from the previous pass)
- `tests/test_nli_unavailable_scoring.py` (new; 18 tests)
- `tests/test_offline_workflows.py` (one fixture given pair identity)
- Documents: `PIPELINE_VALIDATION_REPORT.md`, `RESEARCH_QUALITY_AUDIT.md`, `EVALUATION_PROTOCOL.md`, `DATA_DICTIONARY.md`, `KNOWN_LIMITATIONS.md`, `COMPLETION_MATRIX.md`

`app.py`, the sidebar, `database.py`, and `evaluation/*` are unchanged.

## What this report does not establish

- No empirical accuracy or agreement. See `EVALUATION_PROTOCOL.md`.
- No live collection or real NLI run.

## Human-review workflow (2026-10-27)

- New module `evaluation/pair_review.py` (make-packets, validate, reconcile, finalize, evaluate). Existing evaluators and label schemas are unchanged.
- Tests: `tests/test_pair_review.py` (22 tests, synthetic fixtures). They cover blinding, rejection of altered or missing source text, invalid labels, rationale rules, duplicate and missing rows, reviewer independence, adjudication gating, adjudicator independence, reporting gates, and metric arithmetic on a hand-checked case.
- The documented CLI sequence was run end to end on a synthetic fixture in a temporary folder. That run is a mechanics check only.
- Not run on real analyses. See `COMPLETION_MATRIX.md`.

## Windows readiness audit (2026-10-27)

Defects found and fixed, each with a regression test in `tests/test_pair_review.py`:

1. The coordinator manifest was written beside the reviewer packets. It is now in `coordinator_only\`, and reviewers receive only `packets\`.
2. `finalize` accepted an adjudication file with rows deleted, so a disagreement could silently drop out. It now checks the file against `adjudication_expected_pairs.json`, written by `reconcile`.
3. `finalize` did not reject duplicate rows in the adjudication file or the consensus file.
4. Console JSON could contain non-ASCII characters from company names. Output is now ASCII-only.
5. `evaluate` accepted one run only, which made the 10-company reporting gate hard to reach. It now accepts several run folders and refuses a pair that appears in more than one.

Checked against the real code path, not only fixtures: a producer-consumer test runs the app's matcher (`match_online_evidence`) and NLI detector (`detect_online_contradictions`) with a fake embedding model and classifier, then passes the saved JSON to the packet builder. It uses synthetic text.

Not verified: PowerShell syntax, and any real saved analysis. Both need the user's Windows run.
