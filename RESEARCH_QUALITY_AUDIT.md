# Research-quality audit

Status of each workstream as of 2026-10-27. Statuses: **PASS**, **FAIL**, **BLOCKED**, **NOT EVALUATED**.

## Summary

Research-quality completion is **not achieved**. Two engineering defects were found and fixed with regression tests. Empirical evaluation is blocked: the archive holds no human-reviewed labels and no saved analyses, and the sandbox cannot run the NLI model or live search.

Passing tests show that the code behaves as specified in its tests. They do not show that the scores are valid.

## Workstreams

| Workstream | Status | Evidence |
|---|---|---|
| Baseline test suite reproducible | PASS | 215 run, OK, 1 skipped (original code). |
| Failed NLI handling (D1) | PASS (fixed) | Reproduced, fixed, tested. 6 D1 tests fail on the original code. |
| Partial NLI coverage (D2) | PASS (fixed, policy documented) | Reproduced on original code: 94.4 HIGH for a partly evaluated claim. Now NOT_PREDICTED with reason. Tests cover complete, partial, zero, unmatched, failed and legacy cases. |
| Unassessed states never scored or LOW | PASS | `test_risk_unavailable_states`, `test_nli_unavailable_scoring`. |
| Source independence and company attribution | PASS (offline) | `test_company_identity`, `test_web_search`. Mocked results only. |
| Live collection and search | BLOCKED | Sandbox proxy refuses search and company sites. |
| Real NLI model | BLOCKED | `cross-encoder/nli-deberta-v3-small` cannot be downloaded. |
| Human-reviewed NLI labels | BLOCKED | `evaluation/nli_labels.csv` absent. |
| Human-reviewed risk labels | BLOCKED | `evaluation/risk_labels.csv` absent. |
| Evaluation scripts consume real labels | PASS (synthetic fixtures) / BLOCKED (real labels) | Commands and rules in `EVALUATION_PROTOCOL.md`. Missing input exits 1 with a clear message. |
| Metrics (precision, recall, F1, accuracy, calibration, error analysis) | NOT EVALUATED | No valid reviewed data. |
| Risk model provenance | PASS for active path; model NOT VALIDATED | See item 6 below. |
| Persistence | PASS (offline) | `test_analysis_json_persistence`. Real `data/` absent. |
| Application startup | PASS | HTTP 200. |
| Full UI on saved analysis | NOT EVALUATED | Needs `output/`. |

## Item 6: provenance of the existing risk model

- `risk_model.py` `train_risk_model()` trains on **six hand-written rows** (the code comment calls it a "tiny demonstration training set"). The labels are the author's judgement. They are not reviewed evidence, and they are not synthetic in the sense of generated data.
- Use in the active app: `app.py` `load_demo_risk_model()` returns `None`. Scoring uses the fixed heuristic in `risk_predictor.py` only.
- Use in explanations: `explain_risk_predictions()` in `explainability.py` describes the heuristic for the app and returns without training anything when a model is `None`. The CLI `explainability.py` trains the six-row model only when the saved predictions are not all `HEURISTIC`, which current runs never produce. Old CSVs with trained-model probabilities could still trigger it.
- Conclusion: no output of the six-row model is shown to users in the current workflow. Its output must not be presented as validated. The fixed heuristic's weights and cutoffs (for example, 50, 35, 70) have no empirical basis in this repository.

## What the project may and may not claim

- May claim: a transparent, deterministic review-support heuristic; NLI outputs are model inferences; unassessed claims are reported as unassessed and never scored; the evaluation workflow exists and is tested on synthetic fixtures.
- May not claim: accuracy, detection rate, generalization, calibration, or that the score or the six-row model is validated.

## Remaining conditions for research-quality completion

1. Two independent human reviewers label real eligible pairs from saved analyses; adjudication is documented.
2. Labels are imported and both evaluators run on them; metrics are recorded from that run only.
3. The partial-coverage policy and the heuristic cutoffs are reviewed against those labels before any claim about them is made.
4. Live collection and NLI runs on previously unseen companies are completed on a machine with internet access.

## Real-data status (2026-10-27)

- Real saved analyses: **BLOCKED.** None in the upload or the public repository. Generating them needs live search, which is unreachable from this sandbox.
- Real NLI execution: **BLOCKED.** The model host is unreachable.
- Human-reviewed records: **BLOCKED.** Requires human reviewers. None were created or invented.
- Real-data metrics: **NOT EVALUATED.**

## Human-review workflow (2026-10-27)

- Workflow implemented and tested on synthetic fixtures: **PASS (mechanics only)**. Not evidence about the system's accuracy.
- Real pair labels from two independent reviewers: **BLOCKED.** Requires real saved analyses, which need the user's local run, and human reviewers. None were created.
- Adjudication and metrics on real labels: **NOT EVALUATED.**
- Company-level greenwashing ground truth: **NOT AVAILABLE** in this workflow. Pair labels cannot establish it.
