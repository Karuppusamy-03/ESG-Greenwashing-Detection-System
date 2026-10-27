# ESG Greenwashing Detection

This project is an experimental pipeline for reviewing ESG claims using public or collected source documents, candidate evidence matching, semantic similarity, NLI inference, feature engineering, heuristic risk scoring, explanation output, and timeline analysis. It has not established scientific prediction accuracy and is not a production compliance system.

## Project overview

The workflow is:

1. Collect company documents.
2. Extract environmental claims.
3. Match those claims against candidate source text.
4. Run NLI classification on matched claim-evidence pairs.
5. Build features from similarity and NLI outputs.
6. Compute a heuristic review score between 0 and 100 only when an eligible evidence match exists.
7. Display explanations and timeline evidence.
8. Save analysis results to SQLite and output JSON/CSV artifacts.

This project is designed for B.Tech-style demonstration work and is intentionally transparent about the limits of its evidence and scoring.

PDF collection is bounded to 150 pages and 200,000 characters of primary text. If that limit is reached, it may append up to 75,000 characters from short windows around explicitly headed assurance or review statements, including bounded excerpts of headed measurement tables. Attributed assurance sections are retained as third-party evidence rather than company claims. Embedded verifier statements retain the company-hosted report URL; naming a verifier in a report is not the same as independently retrieving that statement from the verifier's own site, and measurement compatibility is still required before NLI or scoring.

## Required Python version

The project is intended for Python 3.11 to 3.13 where compatible dependency versions are available. This environment was verified on Python 3.13 in the workspace.

## Installation (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If internet access is unavailable, the dependency installation may be blocked; in that case, the project can still be inspected and tested only with the available environment.

## Launch the Streamlit app

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The app is a demonstration workflow and is not a regulated greenwashing detection system.

## Run tests and compilation checks

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -c "from pathlib import Path; files=[p for p in Path('.').rglob('*.py') if not {'.venv','data','output','__pycache__'}.intersection(p.parts)]; [compile(p.read_text(encoding='utf-8-sig'), str(p), 'exec') for p in files]; print(f'Compiled {len(files)} Python files in memory')"
```

Focused test files can be selected by passing their paths after `pytest`.

## Human-reviewed data collection and evaluation

There is no reviewed evaluation dataset in this project. `evaluation/risk_labels.csv` and `evaluation/nli_labels.csv` are intentionally absent; the six demonstration training rows in `risk_model.py` are not evidence or evaluation data. No model accuracy, greenwashing detection rate, or generalization claim is established.

### Screened public data resources

No screened public resource was adopted as an ESG risk benchmark:

- [GreenwashingDetectionDataset](https://github.com/DizzyPanda1/GreenwashingDetectionDataset) has a very small set of mixed claims/accusations, with incomplete provenance and unclear label consistency. Although its repository README mentions MIT, GitHub did not show a formal repository license and rights for underlying third-party material were not established. It was not used.
- [ClimateFever](https://huggingface.co/datasets/tdiggelm/climate_fever) contains climate-related claim/evidence annotations (including support, refutation, and insufficient-information distinctions), but its license is listed as unknown and its domain is not company ESG reporting. It was not used as an ESG-risk benchmark.

These are screening decisions, not claims that no other public research data exists. Any future reuse must check actual data, label definitions, licensing, provenance, and suitability first. NLI entailment/refutation labels, unsupported-claim labels, and corporate greenwashing labels are different tasks and must not be conflated.

### Prepare blind review packets

Use an existing saved company analysis as the source of candidate rows. Replace the example path with a real saved JSON file and choose a new output path; packet creation refuses to overwrite an existing file.

```powershell
.\.venv\Scripts\python.exe -m evaluation.prepare_review_packet `
  --task nli `
  --analysis .\output\company_analyses\example.json `
  --output .\evaluation\review_packets\example_nli_review.csv

.\.venv\Scripts\python.exe -m evaluation.prepare_review_packet `
  --task risk `
  --analysis .\output\company_analyses\example.json `
  --output .\evaluation\review_packets\example_risk_review.csv
```

Packets contain candidate source text and provenance but omit model predictions, confidence values, heuristic features, and risk scores. Pipeline-matched evidence is marked `candidate_pending_review`; it is not automatically deemed independent, compatible, or true. Reviewers must verify the source and company attribution, eligibility, metric, units, period, boundary/scope, and whether the source describes an actual outcome versus a target, contract, estimate, or forecast. Fill the packet’s fields and have two reviewers assess rows independently before adjudication.

For NLI, reviewer labels mean that the exact eligible evidence **entails**, is **neutral** toward, or **contradicts** the exact claim. Neutral means the evidence is not materially supportive or contradictory; it must not be used for absent, inaccessible, ineligible, or incompatible evidence. Mark uncertainty or insufficient information explicitly, resolve disagreements with a documented adjudicator, and do not count unresolved cases as reviewed labels.

For risk review, `review_concern` and `no_review_concern` describe whether an independent human reviewer recommends additional investigation of the claim against the documented eligible evidence. They are not ground-truth greenwashing labels. Reviewers must not see model scores, features, or NLI outputs while assigning labels. Record individual rationales and adjudication notes, including uncertainty and disagreements.

For quantitative pairs, reviewers must record and compare the reporting period, metric, units, emissions scope, organizational boundary, methodology, and reported/estimated/target/contract status. Record the baseline year on both sides when a baseline is cited; mark it not applicable on both sides only when neither statement makes a baseline comparison. A missing or ambiguous dimension is inconclusive, not compatible. Do not assign entailment, neutral, contradiction, review-concern, or no-review-concern labels to unavailable, ineligible, incompatible, or unresolved pairs. Qualitative NLI pairs may mark every measurement dimension not applicable only after the reviewers confirm that the pair is genuinely qualitative.

The automatic gate recognizes a limited set of explicit organizational-boundary and methodology terms; it is not a substitute for human review of the full source and reporting-period definition. It detects explicit fiscal-year and quarter markers but does not parse exact fiscal-year start/end dates or every possible subannual period. Where the source does not establish the reporting period, methodology, organizational boundary, or other required context, retain the pair as inconclusive rather than filling gaps by inference.

### Import and evaluate reviewed data

```powershell
.\.venv\Scripts\python.exe -m evaluation.import_review_packet `
  --task nli `
  --packet .\evaluation\review_packets\example_nli_review.csv `
  --analysis .\output\company_analyses\example.json `
  --output .\evaluation\nli_labels.csv

.\.venv\Scripts\python.exe -m evaluation.import_review_packet `
  --task risk `
  --packet .\evaluation\review_packets\example_risk_review.csv `
  --analysis .\output\company_analyses\example.json `
  --output .\evaluation\risk_labels.csv

.\.venv\Scripts\python.exe .\evaluation\evaluate_nli.py
.\.venv\Scripts\python.exe .\evaluation\evaluate.py
```

Import refuses to overwrite an existing dataset. It validates the completed human annotations, then attaches only the saved pipeline’s corresponding model outputs/features; it never derives labels from those outputs. NLI rows without a matching saved model result are marked `unavailable` and count against prediction coverage. The risk evaluator also refuses rows without eligible evidence, independently adjudicated risk/support labels, provenance, and compatible evidence. Keep the raw review packets and saved analyses as provenance for any imported dataset.

Both evaluators validate source roles, URLs, eligibility, all recorded compatibility dimensions, review completion, group leakage, and label definitions. They keep each company in a single split; exact duplicate claims/pairs cannot be counted as independent examples. Risk evaluation uses a fixed heuristic and selects its cutoff on validation groups only; NLI evaluation scores already-produced model predictions without training or tuning. Both use deterministic grouped 60/20/20-style splits and compare with a majority-class baseline. The held-out test groups must remain untouched during annotation-policy changes, threshold selection, and model iteration. Include adjudicated positive, negative, and inconclusive examples in the review pool; inconclusive cases must be retained for coverage/error analysis but excluded from positive/negative label metrics. The same company, report, near-duplicate claim, and reused evidence excerpt must not leak across splits. NLI reports per-class precision/recall/F1, macro-F1, accuracy, confusion matrix, and prediction coverage when the held-out predictions exist. Cluster-bootstrap intervals are omitted unless at least ten independent held-out company groups support every label. Small or biased samples do not establish population accuracy. Model confidence is not calibrated probability.

The blank `evaluation/risk_annotation_template.csv` and `evaluation/nli_annotation_template.csv` document the full post-review schemas. Use generated blind packets for annotation rather than showing reviewers the model-related columns in those schemas.

## Company identity, aliases and disambiguation

Company search is generic. No per-company table is required for new names.

- **Name-derived aliases.** Legal suffixes and a leading "the" are removed. The compact form (`procter & gamble` -> `proctergamble`) and, for multi-word names, the initialism (`pg`, `ibm`) are derived from the entered name. Aliases are used only to match domain labels; they are not ownership claims.
- **Official-domain discovery.** A host becomes a candidate when its registered label matches a derived alias and at least one collected result on that host names the company without an ambiguous continuation. Government hosts and common hosting/social/aggregator domains are never candidates. Discovered domains are labelled `Potential company source (discovered domain)` and marked `candidate_unverified`. The label `Official company source` is reserved for the configured seed table.
- **Follow-up queries.** Each discovered domain triggers one site-restricted query (`site:<domain> "<name>" ...`). These queries and their results keep full query provenance in diagnostics.
- **Disambiguation.** A name followed by an unfamiliar capitalised term is not accepted as the company. Example: `Delta` in `Delta Air Lines` is rejected with reason `ambiguous_company_reference`. Accepted continuations are corporate descriptors (e.g. `Electronics`, `Holdings`, `Group`), ESG/report vocabulary, lower-case prose, and sentences with a subsidiary cue (`subsidiary of`, `owned by`, `division of`, ...).
- **Known limits.** Disambiguation is lexical. `Delta Electronics` is accepted as Delta because `Electronics` is treated as a descriptor of the same group. A sentence that contains a subsidiary cue is accepted even if the cue refers to a different relationship. Resolving such cases needs external corporate-registry data, which this project does not include.

## Running the test suite

The project uses the standard library `unittest` runner. Running `python -m pytest` also works where pytest is installed, but the commands below do not require it.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Tests are deterministic and mocked by default: they do not contact websites, search providers, or model servers. Some tests depend on saved fixtures in `data\collected_documents`. Those files are gitignored and are not part of a source ZIP, so those tests fail with "Missing saved fixture" until your own `data\` folder is present. The `DashboardTests` class uses Streamlit's `AppTest` and needs the packages in `requirements.txt` installed.

Live checks are **not** part of the unit suite. To verify live search and real NLI, run the Streamlit app against two previously unseen companies and inspect the saved analysis (see "Live verification" below).

## Live verification (run on your own machine)

Live checks require internet access and the full dependency set (including `torch` and `transformers`, which download the NLI model on first use).

1. Start the app: `.\.venv\Scripts\Activate.ps1` then `streamlit run app.py`.
2. Enter an unseen company name, for example `Procter & Gamble` and then `Johnson & Johnson`.
3. Open the saved analysis and check the diagnostics: `name_derived_aliases`, `official_domain_discovery`, per-query results, and rejected results with reasons.
4. Confirm that evidence counted as independent does not come from the company's own domains, and that stages report COMPLETED, PARTIAL, FAILED, or NOT_PREDICTED accurately.

## How the heuristic risk score is calculated

The score is intentionally transparent and deterministic. Its active heuristic uses:

- `similarity`: textual similarity to the candidate evidence source
- `evidence_available`: whether a relevant evidence candidate was found
- `nli_entailment`: count of entailment outputs
- `nli_contradiction`: count of contradiction outputs
- `nli_neutral`: count of neutral outputs

Percentage differences and historical changes are unavailable in this workflow and do not affect the score.

### Unassessed risk claims

Each claim has one of three risk states. A claim with no eligible external evidence is `NOT_PREDICTED` with the reason "No eligible external evidence matched this claim." A claim with evidence but no usable NLI result is also `NOT_PREDICTED`, with the reason "Evidence was found, but NLI verification did not produce a usable result, so the claim is unassessed." Neither state receives a risk score, a predicted class, or a LOW/MEDIUM/HIGH category. Unassessed claims are never treated as support, contradiction, proof of greenwashing, or proof of low risk. A claim whose NLI run completed with zero counts is a real outcome and is scored by the heuristic.

Saved analyses record the reason in the `risk_unavailable_reason` field. Analyses saved before this field existed remain readable; their unassessed rows simply show no recorded reason.

Saved analyses are written through `analysis_serialization.py`, which converts filesystem paths to strings, normalizes NumPy and pandas values, rejects values it cannot save safely, and replaces the JSON file atomically so a failed write never corrupts an existing valid analysis.

The score is bounded to 0-100 and is displayed as a heuristic review indicator. The application and standalone predictor abstain (`NOT_PREDICTED`, with no score/class) when no eligible evidence match exists. The score is not a probability, validated greenwashing classifier, or finding of wrongdoing. Explanations describe the rule applied and the available signals; they are not causal feature effects.

## Interpreting similarity and NLI outputs

- Similarity is a textual relatedness measure, not proof that a claim is true.
- NLI output is a model inference, not proof of greenwashing.
- High similarity alone does not prove support.
- An NLI contradiction label is an experimental model inference that requires human review; it is not proof of falsehood or greenwashing.
- Missing evidence is indeterminate and must not be treated as neutral, contradiction, or no concern.

## Collection diagnostics and source roles

Each new company search collection JSON includes a `diagnostics` object with attempted search queries, per-query raw and accepted result counts, rejected search results with URLs and reasons, search failures, download and extraction outcomes, HTTP status when available, original/final URLs, and source-role counts. If a company-specific query returns an environmental result whose title/snippet omits the company, the result is marked `pending_page_content` and collected only as a bounded attribution candidate. The extracted page must independently mention the full requested company in relevant context before its statements can enter analysis; unconfirmed candidates are retained with their query, URLs, attribution basis, and rejection reason. Company analysis JSON includes pipeline counts for relevant documents, extracted statements, evidence candidates, matched pairs, NLI results, and rejected statements/documents with reasons. Empty search results and failed searches are recorded separately. `analyze_collected_evidence.py` likewise retains source URLs and rejection reasons in its candidate review output while matching only eligible external-source records.

Source roles distinguish company sources, possible company-domain matches, configured regulatory domains, configured environmental sources, explicitly attributed independent third-party sources, rehosted reports, and other sources whose independence or regulatory status is unverified. A government top-level domain alone does not establish regulatory status; unconfigured government hosts remain available for provenance and review but are not used as external matching candidates. A different domain alone does not establish independence, and a name-like domain match is not treated as verified company ownership. Unverified or rehosted material is retained for provenance and review but is not used as an external matching candidate. These labels describe source attribution, not source truth or scientific reliability. Access restrictions such as HTTP 403 are reported, not bypassed.

## How to interpret company timelines

Timeline charts only appear when there are at least two comparable measurements from distinct years with the same metric, unit, and reporting boundary.

Rules:

- Targets, commitments, and forecasts are kept separate from achieved performance.
- Metric, reporting year, unit, emissions scope, organizational boundary, recognized methodology, and applicable baseline must agree before a quantitative pair is marked compatible.
- Quantitative comparisons require a compatible reported-versus-estimated status; missing status is inconclusive. Emissions-reduction percentages require an explicit matching baseline year.
- Different reporting boundaries such as Scope 1 and Scope 2 are not mixed without explicit justification.
- Missing years are not filled with invented measurements.
- Duplicate or conflicting measurements are flagged instead of silently selected.
- A proven mismatch is incompatible; missing metadata is inconclusive and is not fed into NLI or risk scoring.

## Troubleshooting

### Missing models or offline execution

The NLI model is `cross-encoder/nli-deberta-v3-small`. If model download or inference fails, the project reports the failure clearly instead of producing a fake low-risk or high-risk score.

### Network access issues

Web search and downstream scraping depend on internet access. Collector
requests use the operating system's trusted certificate store with normal TLS
verification enabled, including for `robots.txt`. On Windows, install project
dependencies with `python -m pip install -r requirements.txt` to enable that
certificate integration. If an organization or antivirus product inspects
HTTPS, its legitimate root certificate must be installed in the Windows
trusted root store by the responsible administrator/vendor; do not disable
certificate verification or bypass robots rules.

## Known limitations

- No validated accuracy exists until a suitably labeled dataset is evaluated.
- The risk score is a heuristic, not a trained or validated greenwashing classifier.
- NLI output is a model inference, not proof of greenwashing.
- Similarity is not proof that evidence supports a claim.
- Historical trends require comparable reported measurements.
- Targets are not achieved performance.
- Company name matching is lexical. Same-name companies and some subsidiaries cannot be separated without external corporate data.
- Aliases are derived from the entered name and from domain labels; alias discovery from page text is not implemented.
- Internet-dependent workflows may not work offline.
- The project remains a demo workflow intended for academic or internal review, not a production compliance system.

## Verification status (2026-10-26)

- No source code was changed in this pass. No defect was reproduced with evidence that could be fixed safely, so no regression tests were added.
- Offline suite (`python -m unittest discover -s tests -p "test_*.py"`, real torch and sentence-transformers installed): 191 tests, 186 pass, 1 skipped (the live workflow test, enabled only with `ESG_LIVE_TESTS=1`), 4 fail. The 4 failures require the local `data/collected_documents/` and `output/company_analyses/` files, which are gitignored and were not provided. They are not code defects.
- Not verified in this environment, because it had no internet access: live web search for unseen companies, real document collection, the live NLI model download, and real-evidence risk scoring. Run `.\run_all_tests.ps1 -Live` on Windows with internet access to verify them.
- See `COMPLETION_MATRIX.md` for the per-requirement status.
