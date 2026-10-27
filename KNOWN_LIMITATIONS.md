# Known limitations

## Evaluation and validity

- **No empirical validation.** No human-reviewed labels exist in this project. No accuracy, precision, recall, F1, or agreement has been measured.
- **Heuristic score.** The weights and cutoffs in `risk_predictor.py` (class cutoff 50; categories LOW below 35, MEDIUM 35–70, HIGH from 70) are not justified by data.
- **Six-row model.** `risk_model.py` trains on six hand-written rows for demonstration. Its output is not used for scoring and must not be presented as validated.
- **Scores are not calibrated.** Neither the heuristic score nor NLI confidence is a probability.
- **Synthetic fixtures are software tests only.** They are not evidence about real companies.

## Pipeline

- **Partial NLI coverage is not scored (D2, fixed by policy).** A claim is assessed only when every eligible matched pair has a completed NLI result. This is conservative: claims with some unassessed evidence are left unscored, so the output has more `NOT_PREDICTED` rows than before.
- **Earlier saved CSVs are not re-scored.** Risk CSVs written before this fix (`output/esg_risk_predictions.csv`) and feature files without the coverage columns keep their old values. Re-run `feature_builder.py` and `risk_predictor.py` to apply the policy. Failed NLI rows inside older saved analyses are excluded automatically, because the check uses `nli_result`.
- **Similarity is textual only.** High similarity does not show that evidence supports a claim. The score uses the maximum similarity over eligible matches.
- **NLI model load failure stops the stage.** If `cross-encoder/nli-deberta-v3-small` cannot be loaded, `contradiction_detector` raises a `RuntimeError`. Pairs are marked `UNAVAILABLE` only when the model runs and returns an unusable output for that pair.
- **Company disambiguation is lexical.** `Delta Electronics` is accepted as Delta. Subsidiary relationships are not checked against any corporate registry.
- **Source independence is attribution-based.** A different domain does not prove independence.
- **Measurement compatibility is conservative.** Missing period, scope, unit, or boundary data makes a pair inconclusive. Valid evidence can be excluded this way.

## Live operation (not verified here)

- Live search, company-site collection, and the NLI model download were blocked in this sandbox. The live path is verified only through mocked tests.
- The README recommends a live check on two unseen companies, run on the user's machine.

## Environment

- The unit suite needs `transformers` and `sentence-transformers` installed (six test modules import them).
- Two data-dependent tests need the local `data/` and `output/` folders. They fail with "Missing saved fixture" without them.
- Test runs create an empty `data/collected_documents/` directory. It is gitignored and was removed from the delivered copy.

## Real-data status (2026-10-27)

- No real company analyses, NLI runs or human labels exist in the upload or the public repository.
- The sandbox cannot reach the NLI model host, live search, or company websites. Only package indexes are reachable.
- Every number in this project comes from mocked or synthetic tests. None is a measure of real-world performance.

## Human-review workflow (2026-10-27)

- The pair-review workflow has been tested only on synthetic fixtures (`tests/test_pair_review.py`). Its labels and metrics from those fixtures are software tests, not research results.
- Pair labels are evidence-level judgements. They do not establish company-level greenwashing, and one CONTRADICTING pair is not proof of greenwashing.
- Reviewer independence is enforced only by the workflow: distinct identifiers, separate packets, no model columns in packets. It cannot verify that two people did not discuss the pairs.
- Reviewers may open linked sources; the workflow cannot check what they read.
- Reviewer labels are not validated against any external ground truth.
