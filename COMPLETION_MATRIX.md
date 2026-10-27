# ESG Greenwashing Detection: Completion matrix (verification run, 2026-10-27)

Status key: **PASS** = run and observed in this environment. **FAIL** = a defect was confirmed and is not yet fixed. **BLOCKED** = cannot be checked here; the reason and the required input are given. **NOT EVALUATED** = not run, or no valid data exists.

Research-quality completion: **NOT ACHIEVED.** Engineering defects found in this pass are fixed and tested. Empirical validation is blocked on human-reviewed labels, and live checks are blocked by the sandbox.

## Scope of this run

- Source: `ESG-Greenwashing-Detection-System-main.zip` (source only). `data/`, `output/`, `evaluation/*labels.csv` and trained model artifacts are absent from the archive.
- Environment: Linux, Python 3.13.17. Packages from `requirements.txt` plus `transformers` and `sentence-transformers`.
- Real NLI model (`cross-encoder/nli-deberta-v3-small`): **cannot be downloaded** here.
- Live web search and company websites: **refused by the sandbox proxy.**

## Test results

| Run | Result |
|---|---|
| Baseline, before any change (`python3 -m unittest discover -s tests -p "test_*.py"`) | 215 run, OK, 1 skipped |
| Previous pass (after D1 fix) | 225 run, OK, 1 skipped |
| Current (after D2 policy and tests) | **233 run, OK, 1 skipped** |
| Focused modules (NLI, risk states, heuristics, evaluation, review workflow, persistence, offline workflows) | 162 run, OK (final run, includes the legacy saved-row test) |
| Skipped test | live workflow test; needs `ESG_LIVE_TESTS=1` and internet |
| Compile (`py_compile` on all project `.py` files) | OK |
| Dependency import check (streamlit, pandas, numpy, sklearn, xgboost, shap, transformers, sentence-transformers, torch, pymupdf, plotly, ddgs, requests, truststore, bs4, matplotlib) | All import |
| App startup (`streamlit run app.py --server.headless true`) | HTTP 200, no traceback in log |

Mocked tests versus real checks: all unit tests use mocked search and a fake NLI classifier. No test used the real NLI model or live sources.

## Requirement matrix

| # | Requirement | Status | Evidence / gap |
|---|---|---|---|
| 0 | Previously reported Windows defects | BLOCKED | No Windows output was supplied. |
| 1 | Live search for unseen companies | BLOCKED | Proxy refuses search and company sites. |
| 2 | Discovered domains stay candidates until verified | PASS (offline) | `test_company_identity`, `test_matcher_excludes_unverified_external_source_role`. |
| 3 | Ambiguous names and subsidiaries | PASS (offline, mocked) | `test_company_identity`. No live Delta search. |
| 4a | Real document collection | BLOCKED | No outbound access to company or report sites. |
| 4b | Extraction, eligibility, matching | PASS (offline fixtures) | Matcher key names verified against NLI detector and feature builder. |
| 4c | NLI execution on real pairs | BLOCKED | Model cannot be downloaded. Offline path verified with a fake classifier. |
| 5a | **Defect D1: failed NLI run scored as completed (LOW)** | **PASS (fixed)** | Reproduced before fix. Fixed in `contradiction_detector.py` and `feature_builder.py`. 6 of 10 D1 regression tests fail on the original code and pass after the fix. |
| 5b | **Defect D2: partial NLI coverage scored as complete (HIGH 94.4 on original code for a claim with one of two matches evaluated)** | **PASS (fixed, policy documented)** | Policy: a claim is scored only when every eligible matched pair has a completed NLI result. Otherwise NOT_PREDICTED with reason `NLI_PARTIAL_REASON`. Evidence is kept. 7 coverage tests (complete, partial, zero, unmatched-row, failed-pair, legacy, order). |
| 5c | Unassessed states never receive a score or LOW class | PASS | `test_risk_unavailable_states`, `test_nli_unavailable_scoring`. |
| 6 | Risk model provenance | PASS for active path; model itself NOT VALIDATED | See `RESEARCH_QUALITY_AUDIT.md` item 6. |
| 7 | Timeline analysis | PASS (offline tests) | Existing comparability tests pass. Not re-audited line by line. |
| 8 | Persistence (SQLite, saved analyses) | PASS (offline) | `test_analysis_json_persistence`. Real `data/` database not present. |
| 9 | Application startup | PASS | HTTP 200. Saved-analysis UI not exercised (needs `output/`). |
| 10 | Evaluation scripts on real labels | PASS (synthetic fixtures); BLOCKED (real labels) | Commands and rules in `EVALUATION_PROTOCOL.md`. Both CLIs exit 1 with a clear message when labels are missing. |
| 11 | Human-reviewed NLI labels | BLOCKED | `evaluation/nli_labels.csv` absent. Requires two independent reviewers plus adjudication. |
| 12 | Human-reviewed risk labels | BLOCKED | `evaluation/risk_labels.csv` absent. |
| 13 | Review packets from real saved analyses | BLOCKED | `output/company_analyses/*.json` absent. See `EVALUATION_PROTOCOL.md` for the exact files needed. |
| 14 | Metrics (precision, recall, F1, accuracy, calibration) | NOT EVALUATED | No valid reviewed data. No metric has been computed. |
| 15 | Error analysis by company, source, claim category | NOT EVALUATED | Depends on 11–13. |

## Overall

Engineering: two confirmed defects (D1, D2) fixed with regression tests. One design gap is documented, not changed: the NLI model itself has not been checked for quality.

Empirical: **BLOCKED.** No human labels, no saved analyses, and no live NLI run exist in the archive or the sandbox.

## Actions needed to finish

1. Supply `output/company_analyses/*.json` (saved analyses from real runs) and, if available, `data/greenwashing_risk.db`. Back up both before any run.
2. On a machine with internet access, run the app on two unseen companies plus Delta Electronics, Delta Air Lines and Delta Dental. Record the discovered domains, eligible sources, NLI status, and risk or inconclusive reason.
3. Produce review packets from those analyses, obtain two independent labels per row, adjudicate, import, and run the evaluators (see `EVALUATION_PROTOCOL.md`).
4. Confirm the two data-dependent tests pass once `data/` and `output/` exist.

## Real-data availability check (2026-10-27)

Checks run in this sandbox:

| Source | Result |
|---|---|
| Upload ZIP (`/mnt/user-data/attachments`) | Source only. No `data/`, `output/`, labels or packets. |
| Public repo `Karuppusamy-03/esg-greenwashing-detection-system` (cloned read-only, commit `49a7e81`) | 54 files, same source-level content. No saved analyses, no databases, no labels, no packets. |
| Local disk search for NLI model cache, `esg_risk_predictions.csv`, project `.db` | None found. |
| `huggingface.co` (NLI model) | Unreachable (no connection). |
| `duckduckgo.com` (live search), `apple.com`, `epa.gov` (company and regulatory sites) | Unreachable. |
| `pypi.org` | Reachable (package installs only). |

Consequence: live collection, the real NLI model, saved company analyses and human-reviewed records are all **BLOCKED**. No real-data metrics exist and none are reported.

## Human-review workflow (2026-10-27)

| Item | Status | Evidence |
|---|---|---|
| Pair-review module (`evaluation/pair_review.py`): blinded packets, manifest, validation, reconciliation, adjudication gate, evaluation | PASS (synthetic tests) | `tests/test_pair_review.py`, 22 tests, OK. CLI run end to end on a synthetic fixture in a temporary folder. |
| Reviewer instructions and four-label definitions (`evaluation/REVIEWER_GUIDE.md`) | DONE (document) | Not yet tested by real reviewers. |
| Real company analyses for packets | BLOCKED | Requires running the app locally with internet access. |
| Two independent human labels on real pairs | BLOCKED | Requires human reviewers. None created. |
| Adjudication on real disagreements | BLOCKED | Depends on real labels. |
| Metrics on real labels | NOT EVALUATED | Reporting gate requires at least 30 pairs, 5 SUPPORTING, 5 CONTRADICTING, 10 companies. |

## Windows readiness for real-data human review (2026-10-27)

| Item | Status | Evidence |
|---|---|---|
| Pair-review CLI commands match `WINDOWS_HUMAN_REVIEW_RUNBOOK.md` | PASS (Python run on Linux) | Full sequence run from the project root on a synthetic run folder: inspect, make-packets, validate, reconcile, finalize, evaluate. |
| PowerShell wrapper lines | NOT EVALUATED | PowerShell is not available in this sandbox. The first Windows run is the check. |
| Reviewer packets exclude model outputs and the manifest | PASS (tests) | Packets live in `packets\`; the manifest lives in `coordinator_only\`. Tests check both. |
| Adjudicator cannot be an original reviewer | PASS (tests) | `finalize` rejects the match. |
| Deleted or added adjudication rows are refused | PASS (tests, new) | `finalize` checks rows against `adjudication_expected_pairs.json`. |
| Non-ASCII company names do not break console output | PASS (test, new) | CLI JSON is ASCII-only. |
| Multiple run folders evaluated together; duplicate pairs refused | PASS (tests, new) | `evaluate --run-dir ...` |
| Producer-consumer check with the real matcher and NLI detector (fake model and classifier, synthetic text) | PASS (test, new) | Confirms the saved-analysis shape the matcher and detector write is what the packet builder reads. |
| Real saved analyses from the app | BLOCKED | Requires the user's local run. |
| Two-reviewer labels, adjudication, metrics on real data | BLOCKED | Requires human reviewers and real analyses. |
