# Evaluation protocol

How human-reviewed data must be produced, checked and evaluated. The scripts already exist and are tested on synthetic fixtures. This document does not add metrics or claim results.

**Status: BLOCKED.** No human-reviewed labels, no saved analyses, and no real NLI run exist in this project. No accuracy, precision, recall, F1, or agreement has been measured. Do not report one until the steps below are completed on real data.

## Required local inputs (not in the archive)

| File | Produced by | Needed for |
|---|---|---|
| `output/company_analyses/<company>_<timestamp>.json` | the app (saved company analysis) | review packets, import |
| `evaluation/review_packets/<name>_nli_review.csv` | `prepare_review_packet.py --task nli` | two reviewers' labels |
| `evaluation/review_packets/<name>_risk_review.csv` | `prepare_review_packet.py --task risk` | two reviewers' labels |
| `evaluation/nli_labels.csv` | `import_review_packet.py --task nli` | `evaluate_nli.py` |
| `evaluation/risk_labels.csv` | `import_review_packet.py --task risk` | `evaluate.py` |

If the saved analyses are not supplied, nothing in this document can be run.

## Commands (run from the project root, Windows PowerShell form as in README; use `python3` on Linux)

```powershell
# 1. Blind packets (refuse to overwrite existing output)
.\.venv\Scripts\python.exe -m evaluation.prepare_review_packet --task nli  --analysis .\output\company_analyses\<file>.json --output .\evaluation\review_packets\<name>_nli_review.csv
.\.venv\Scripts\python.exe -m evaluation.prepare_review_packet --task risk --analysis .\output\company_analyses\<file>.json --output .\evaluation\review_packets\<name>_risk_review.csv

# 2. Reviewers complete the packet independently; adjudicator fills adjudicated_label (NLI) or label (risk).

# 3. Import (refuses to overwrite; validates labels; attaches saved model outputs only)
.\.venv\Scripts\python.exe -m evaluation.import_review_packet --task nli  --packet .\evaluation\review_packets\<name>_nli_review.csv  --analysis .\output\company_analyses\<file>.json --output .\evaluation\nli_labels.csv
.\.venv\Scripts\python.exe -m evaluation.import_review_packet --task risk --packet .\evaluation\review_packets\<name>_risk_review.csv --analysis .\output\company_analyses\<file>.json --output .\evaluation\risk_labels.csv

# 4. Evaluate. Both print JSON to stdout; redirect to keep a machine-readable record.
New-Item -ItemType Directory -Force .\evaluation\results | Out-Null
.\.venv\Scripts\python.exe -m evaluation.evaluate_nli .\evaluation\nli_labels.csv  > .\evaluation\results\nli_metrics.json
.\.venv\Scripts\python.exe -m evaluation.evaluate     .\evaluation\risk_labels.csv > .\evaluation\results\risk_metrics.json
```

Both evaluators exit with code 1 and a message when the input is missing. Verified in this environment with no labels present.

## Expected input columns

- NLI (`evaluate_nli.py`, default `evaluation/nli_labels.csv`): the schema in `evaluation/nli_annotation_template.csv`. Required columns include `pair_id`, `company`, `company_group`, `report_id`, `split_group`, `claim_group`, `claim`, `claim_source_url`, `claim_source_role`, `evidence_source_url`, `evidence_source_role`, `evidence_excerpt`, `evidence_availability`, `eligibility_decision`, the eight `*_match` compatibility columns, and the claim and evidence baseline, boundary, methodology and performance-status fields.
- Risk (`evaluate.py`, default `evaluation/risk_labels.csv`): the schema in `evaluation/risk_annotation_template.csv`. Required columns include `annotation_id`, `analysis_id`, `claim_id`, `company`, `report_id`, `claim`, `claim_group`, `split_group`, `claim_source_url`, `evidence_source_url`, `evidence_source_urls`, `evidence_source_role`, `evidence_excerpt`, `compatibility_decision`, the eight `*_match` columns, `label`, `review_status`, and the model feature columns.

## Label sets (validated by the scripts)

- NLI reviewer labels: `entailment`, `neutral`, `contradiction`, `insufficient_evidence`, `uncertain`.
- Risk: `review_concern` (positive), `no_review_concern` (negative). Support labels: `entailment`, `neutral`, `contradiction`, plus reviewer-only `insufficient_evidence` and `uncertain`.
- Any other value fails validation.

## Validation rules enforced by the scripts

- Required columns must all be present; missing columns are an error.
- Evidence source roles must be in the allowed set; URLs must be `http(s)`.
- Compatibility dimensions must be recorded. A missing or ambiguous dimension is inconclusive, not compatible.
- Both reviewers must complete their labels; disagreements need an adjudicated label.
- Each company stays in one split. Exact duplicate claims or pairs are not counted as independent examples.
- Inconclusive and unavailable pairs are kept for coverage but excluded from positive and negative label metrics.
- The risk threshold is chosen on validation groups only; the test groups are not used for selection.

## Outputs

`evaluate_nli.py` (JSON): per-class precision, recall, F1 for `entailment`, `neutral`, `contradiction`; macro-F1; accuracy; confusion matrix; prediction coverage on the held-out test groups; cluster-bootstrap intervals only when at least ten independent held-out company groups support every label.

`evaluate.py` (JSON): positive-class precision, recall, F1; accuracy; a majority-class baseline evaluated on the same test groups; the threshold selected on validation groups; bootstrap intervals under the same ten-group condition.

## Reporting rules

- Model confidence is not a calibrated probability.
- Small or biased samples do not establish population accuracy.
- Synthetic test fixtures must never appear in reported results.
- Keep the raw packets and saved analyses as provenance for any imported dataset.

## Remaining human and live requirements

1. Supply `output/company_analyses/*.json` from real runs (the app needs internet access to produce them; see `KNOWN_LIMITATIONS.md`).
2. Two independent reviewers who did not build the pipeline label each packet row; a documented adjudicator resolves disagreements.
3. Run the commands above and keep the JSON outputs with the packets and saved analyses.
4. Only then record metrics, and state the sample sizes and the number of held-out company groups alongside them.

## Real-data status (2026-10-27)

No real saved analyses are available (see `COMPLETION_MATRIX.md`, "Real-data availability check"). The review packets, labels and metrics are therefore **not produced**. The exact steps to unblock, on a machine with internet access and the full dependencies:

1. Run `streamlit run app.py`, enter each unseen company (for example Procter & Gamble and Johnson & Johnson), and let the analysis finish. Each run writes `output/company_analyses/<company>_<timestamp>.json`. Back up `data/` and `output/` first.
2. Generate packets with the commands above from those JSON files, under new file names.
3. Two reviewers label each packet independently; an adjudicator resolves disagreements.
4. Import and evaluate, then keep the JSON outputs.

## Two-reviewer pair workflow (`evaluation/pair_review.py`) — status: READY, NOT RUN ON REAL DATA

This workflow labels claim-evidence **pairs** with `SUPPORTING`, `CONTRADICTING`, `UNRELATED` or `UNCERTAIN`. It is separate from the NLI and risk evaluators above, whose label sets it does not change. Reviewer instructions: `evaluation/REVIEWER_GUIDE.md`.

### Inputs you must supply (after generating real analyses)

- One saved company analysis: `output\company_analyses\<company>_<id>.json`. It must contain `analysis_id`, `company`, `company_claims`, `matching_results` and `nli_results`, as written by `app.py`. Use one analysis per company run.
- Two reviewers and one adjudicator, none of whom built the pipeline and none of whom is the other reviewer.

### Commands

The complete Windows PowerShell sequence, with every path, is in `WINDOWS_HUMAN_REVIEW_RUNBOOK.md`. Summary of the commands, run from the project root:

```powershell
.\.venv\Scripts\python.exe -m evaluation.pair_review inspect --analysis <saved analysis .json>
.\.venv\Scripts\python.exe -m evaluation.pair_review make-packets --analysis <saved analysis .json> --out-dir <run folder> --reviewer-a R1 --reviewer-b R2
.\.venv\Scripts\python.exe -m evaluation.pair_review validate --packet <run folder>\packets\packet_a_R1.csv --returned <run folder>\returned_R1.csv
.\.venv\Scripts\python.exe -m evaluation.pair_review validate --packet <run folder>\packets\packet_b_R2.csv --returned <run folder>\returned_R2.csv
.\.venv\Scripts\python.exe -m evaluation.pair_review reconcile --packet-a <run folder>\packets\packet_a_R1.csv --returned-a <run folder>\returned_R1.csv --packet-b <run folder>\packets\packet_b_R2.csv --returned-b <run folder>\returned_R2.csv --manifest <run folder>\coordinator_only\coordinator_manifest.csv --out-dir <run folder>\recon
.\.venv\Scripts\python.exe -m evaluation.pair_review finalize --queue <run folder>\adjudicated_ADJ.csv --consensus <run folder>\recon\consensus_labels.csv --manifest <run folder>\coordinator_only\coordinator_manifest.csv --out <run folder>\final_labels.csv
.\.venv\Scripts\python.exe -m evaluation.pair_review evaluate --run-dir <run folder> [--run-dir <another run folder> ...] --out <results folder>\evaluation_results.json
```

Layout of a run folder (`<run folder>`): `packets\` (reviewers receive only their own file), `coordinator_only\` (manifest with model outputs, never sent), `returned_*.csv`, `recon\` (agreement, consensus, queue), `adjudicated_ADJ.csv`, `final_labels.csv`.

### Validation rules enforced

- Packet and returned file must have the exact column schema; changing columns is rejected.
- Every packet row must be returned exactly once; no added, missing or duplicated `pair_id`.
- Source text (claim, evidence, URLs, titles, dates, IDs) must be unchanged, column by column.
- Labels must be one of the four; `CONTRADICTING` and `UNCERTAIN` require a rationale; a labelled row requires `reviewer_id`.
- One reviewer identity per returned file; the two reviewers must differ.
- Adjudicator must not be either original reviewer; every disagreement needs an adjudicated label and rationale before `finalize` runs.
- Pair IDs must exist in the coordinator manifest; a manifest from a different analysis is rejected.
- Only pairs labelled by both reviewers, or adjudicated, enter final labels.

### Reporting status

`evaluate` reports `INSUFFICIENT_DATA` unless there are at least 30 final labelled pairs, at least 5 `SUPPORTING` and 5 `CONTRADICTING` pairs, and at least 10 companies. Even when reportable, results carry the caveat that they describe evidence-pair agreement only.

### What the metrics mean

- Reviewer agreement (percent agreement and Cohen's kappa), from `reconcile`: consistency of human labels only. Not model accuracy.
- Evidence-pair agreement with NLI (precision, recall, F1 for `SUPPORTING` against `ENTAILMENT` and `CONTRADICTING` against `CONTRADICTION`; confusion table of human label by NLI result). `UNCERTAIN` pairs and pairs without a completed NLI result are excluded and counted separately.
- Not computed (needs data not in this workflow): company-level greenwashing accuracy, risk-prediction accuracy, calibration, population generalization, and bootstrap intervals (need at least 10 independent companies per label).
