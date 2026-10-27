# Audit report

## Summary
The project originally used a six-row demo classifier in `risk_model.py` and `risk_predictor.py` to generate risk scores from a tiny handcrafted dataset. That caused nearly all claims to be treated as high risk regardless of the actual evidence state. The project was also missing a documented evaluation entry point and lacked an explicit NLI-failure path.

## Issue discovered

### 1. Unreliable six-row classifier controlled real risk scoring
- Affected files: `risk_model.py`, `risk_predictor.py`, `app.py`, `explainability.py`
- Cause: The app used a classifier trained on six manually written examples. This model was not statistically valid, not calibrated, and not suitable for real greenwashing review. It could rank near-zero similarity/no-evidence cases as high risk.
- Fix implemented: The active scoring path now uses a deterministic, transparent heuristic that combines `similarity`, `evidence_available`, `nli_entailment`, `nli_contradiction`, and `nli_neutral` with explicit safety checks and normalized inputs. The score is bounded to 0-100 and documented as a heuristic, not a validated classifier.
- Test used to verify: `python -m unittest discover -s tests -p "test_*.py" -q` and `tests/test_risk_heuristics.py`
- Remaining limitation: It is a review support score, not a validated or trained greenwashing classifier.

### 2. NLI failures could silently create misleading outputs
- Affected files: `contradiction_detector.py`
- Cause: The NLI pipeline sometimes returned unexpected structures or failed to load a model, and the code did not clearly preserve unavailable results.
- Fix implemented: The detector now marks unusable outputs as `UNAVAILABLE`, raises useful errors for model-load problems, and keeps the evidence quality and source provenance intact.
- Test used to verify: the NLI regression in `tests/test_risk_heuristics.py` and the offline pipeline tests.
- Remaining limitation: Offline or missing-model execution must still be treated as a limited analysis, not as factual proof.

### 3. Historical timeline validation was under-documented and easy to misuse
- Affected files: `timeline_analyzer.py`, `app.py`
- Cause: The charting logic had comparability checks, but it was not clearly described as requiring the same metric, unit, and boundary across multiple years.
- Fix implemented: The code preserves explicit reporting years, excludes targets and incompatible units/boundaries, and does not fabricate missing data. The UI and docs now explain this limitation clearly.
- Test used to verify: the timeline tests contained in `tests/test_offline_workflows.py`.
- Remaining limitation: Some companies legitimately have insufficient historical data; no chart is shown in those cases.

### 4. Evaluation path was incomplete
- Affected files: new `evaluation/evaluate.py`
- Cause: The workspace did not contain a real evaluation script or a documented `labels.csv` workflow.
- Fix implemented: A new evaluation script validates the `labels.csv` input, checks class balance and duplicates, runs the same feature-building and risk pipeline, and emits a threshold sweep and confusion metrics with warnings for limited data.
- Test used to verify: direct script execution and syntax checks.
- Remaining limitation: A valid labeled dataset is required to produce meaningful metrics.
