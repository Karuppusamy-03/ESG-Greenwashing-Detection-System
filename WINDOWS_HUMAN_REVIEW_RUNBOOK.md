# Windows runbook: real-company analyses and two-reviewer pair labelling

Status: **NOT RUN ON REAL DATA.** The Python commands have been run on synthetic data in a Linux sandbox. The PowerShell wrapper lines (variables, `Get-ChildItem`, `ConvertFrom-Json`) have **not** been executed on Windows, because PowerShell is not available in that sandbox. Treat your first run as the check, and send me any output that differs.

Work only in the project folder. Never delete `data\`, `output\`, or any file in a run folder that has `final_labels.csv`.

---

## 0. Prerequisites (run once)

Open **Windows PowerShell** and go to the project folder (the one containing `app.py`):

```powershell
cd "C:\path\to\ESG-Greenwashing-Detection-System-main"
```

Check Python (3.11, 3.12 or 3.13 is required):

```powershell
py -3.13 --version
```

If that fails, try `py -3.12 --version` or `py -3.11 --version`, and use the version that works in place of `3.13` below.

Create the virtual environment and install dependencies (torch is large; expect a long install):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check the imports used by the app and the review workflow:

```powershell
.\.venv\Scripts\python.exe -c "import streamlit, pandas, sklearn, transformers, sentence_transformers, torch, fitz, ddgs, bs4, requests; print('dependencies OK')"
```

Run the full test suite (all mocked; no internet needed):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Expected final line: `OK (skipped=1)`. The one skip is the live test, which needs `ESG_LIVE_TESTS=1` and internet.

If you see `ModuleNotFoundError: No module named 'evaluation'`, you are not in the project folder. Check with `Get-Location`.

---

## 1. Back up existing data (before any analysis)

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
New-Item -ItemType Directory -Force ".\backups\$stamp" | Out-Null
if (Test-Path ".\data")   { Copy-Item ".\data"   ".\backups\$stamp\data"   -Recurse }
if (Test-Path ".\output") { Copy-Item ".\output" ".\backups\$stamp\output" -Recurse }
```

Existing data is never deleted by the workflow. The backup is an extra safety copy.

---

## 2. Start the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Leave this window open. Open the URL it prints (normally `http://localhost:8501`) in your browser.

The first analysis downloads two models from Hugging Face (the NLI model `cross-encoder/nli-deberta-v3-small` and the embedding model `all-MiniLM-L6-v2`). This needs internet access and can take several minutes. Later runs reuse the cached models.

---

## 3. Run analyses for at least 10 distinct companies

For each company:

1. In the browser, open the company analysis page of the app and enter one company name. Use a different company each time.
2. Wait until every stage has finished. Stages that report PARTIAL or FAILED are still usable for review, but note them in your run log.
3. Confirm the analysis was saved (see step 4). Each finished analysis writes one file under `output\company_analyses\`.

Keep a short log (a text file you create, for example `C:\...\review_log.txt`) listing: company name, date, file name, and any stage that was PARTIAL or FAILED.

Do not change thresholds or matching settings to get more matches. Do not edit saved analyses.

---

## 4. Locate and check the saved analyses

List the saved files:

```powershell
Get-ChildItem ".\output\company_analyses\*.json" | Sort-Object LastWriteTime | Format-Table Name, Length, LastWriteTime -AutoSize
```

Each file is named `<company>_<16 hex characters>.json`. The company name inside the file is the authority, not the file name.

---

## 5. Pair-review workflow (one run folder per analysis)

Use one run folder per company analysis: `run01`, `run02`, and so on. Reviewer identifiers used below are `R1` and `R2`; the adjudicator is `ADJ`. You may use other identifiers, but they must be different people, and the same identifiers must be used in every step of a run.

### 5.1 Choose the analysis and check it (read-only)

Set `$company` to the exact name you entered in the app for this run. The script finds the saved file whose `company` field matches it. Stop if no file or more than one file matches; do not guess.

```powershell
$py = ".\.venv\Scripts\python.exe"
$run = ".\evaluation\pair_review\run01"
$company = "Procter & Gamble"
$matches = @(Get-ChildItem ".\output\company_analyses\*.json" | Where-Object { (Get-Content $_.FullName -Raw | ConvertFrom-Json).company -eq $company })
$matches.Count
$analysis = $matches[0].FullName
$analysis
& $py -m evaluation.pair_review inspect --analysis $analysis
```

If `$matches.Count` is not `1`, run the listing in step 4 and check the file by hand before continuing. For the first run, the example company name is only an example: replace it with your own company name.

`inspect` writes nothing. Check its output:

- `ready_for_packets` must be `true`. If it is `false`, do not continue with this analysis. Choose another company (see "Common errors").
- `company` must be the company you entered.
- `pairs_by_model_nli_status` shows how many pairs have a completed NLI result (`completed`), an unavailable result (`unavailable`), or no NLI result (`not_run`). Reviewers still label every pair; the model status is hidden from them.
- `rejections` lists pairs dropped and why. These are not errors.

### 5.2 Create the blinded packets

```powershell
& $py -m evaluation.pair_review make-packets --analysis $analysis --out-dir $run --reviewer-a R1 --reviewer-b R2
```

Expected output: `pairs_in_packet` and a rejection summary. Files created:

- `$run\packets\packet_a_R1.csv`: send to reviewer R1 only.
- `$run\packets\packet_b_R2.csv`: send to reviewer R2 only.
- `$run\packets\packet_build_report.json`: counts for the run. Safe to share with the coordinator.
- `$run\coordinator_only\coordinator_manifest.csv`: **model outputs. Never send this to a reviewer.**

Reviewers must not receive `coordinator_only\`, the other reviewer's file, or the `recon\` folder.

### 5.3 Reviewers label their packet

Send each reviewer their own packet and `evaluation\REVIEWER_GUIDE.md`. The reviewer:

1. Opens the packet in Excel or Notepad.
2. Fills only `label`, `reviewer_id` and `rationale`.
3. Saves it as CSV (UTF-8) with a new name.

When the reviewer returns the file, save it in the run folder:

- R1's returned file as `$run\returned_R1.csv`
- R2's returned file as `$run\returned_R2.csv`

Each reviewer works alone and does not see the other's labels.

### 5.4 Validate each returned file

```powershell
& $py -m evaluation.pair_review validate --packet "$run\packets\packet_a_R1.csv" --returned "$run\returned_R1.csv"
& $py -m evaluation.pair_review validate --packet "$run\packets\packet_b_R2.csv" --returned "$run\returned_R2.csv"
```

Expected output: counts including `complete`, `unlabelled`, and `reviewer_ids`. A blank label counts as unlabelled, not as an error. Any error stops the step; see "Common errors". Do not edit the returned file by hand to pass validation.

### 5.5 Reconcile the two reviewers

```powershell
& $py -m evaluation.pair_review reconcile --packet-a "$run\packets\packet_a_R1.csv" --returned-a "$run\returned_R1.csv" --packet-b "$run\packets\packet_b_R2.csv" --returned-b "$run\returned_R2.csv" --manifest "$run\coordinator_only\coordinator_manifest.csv" --out-dir "$run\recon"
```

Files created in `$run\recon\`:

- `agreement_report.json`: pairs labelled by both reviewers, percent agreement, Cohen's kappa, disagreements, and any pairs labelled by only one reviewer.
- `consensus_labels.csv`: pairs where both reviewers agreed.
- `adjudication_queue.csv`: pairs where they disagreed. Give this to the adjudicator.
- `adjudication_expected_pairs.json`: the list of disagreements. Do not edit it.

### 5.6 Adjudicate disagreements

Send `$run\recon\adjudication_queue.csv` to the adjudicator (`ADJ`), who must be a third person and not R1 or R2. The adjudicator fills `adjudicated_label` (one of SUPPORTING, CONTRADICTING, UNRELATED, UNCERTAIN), `adjudicator_id`, and `adjudication_rationale`, and returns the file. Save it as:

```powershell
# Save the adjudicator's returned file as:  $run\adjudicated_ADJ.csv
```

The adjudicator must return **every row** of the queue, even if the queue has no rows (in that case, the file is just the header).

### 5.7 Finalize the run

```powershell
& $py -m evaluation.pair_review finalize --queue "$run\adjudicated_ADJ.csv" --consensus "$run\recon\consensus_labels.csv" --manifest "$run\coordinator_only\coordinator_manifest.csv" --out "$run\final_labels.csv"
```

Expected output: `final_labels`, `adjudicated`, and `pairs_not_labelled_by_both_reviewers_excluded`. The last number counts pairs that were left with only one reviewer's label. Those pairs are not in the metrics. Record the number in your log.

Repeat sections 5.1 to 5.7 for each company, with `run02`, `run03`, and so on. Update `$analysis` and `$run` each time.

### 5.8 Evaluate all runs together

When at least 10 runs have `final_labels.csv`:

```powershell
New-Item -ItemType Directory -Force ".\evaluation\pair_review\results" | Out-Null
& $py -m evaluation.pair_review evaluate --run-dir ".\evaluation\pair_review\run01" --run-dir ".\evaluation\pair_review\run02" --run-dir ".\evaluation\pair_review\run03" --run-dir ".\evaluation\pair_review\run04" --run-dir ".\evaluation\pair_review\run05" --run-dir ".\evaluation\pair_review\run06" --run-dir ".\evaluation\pair_review\run07" --run-dir ".\evaluation\pair_review\run08" --run-dir ".\evaluation\pair_review\run09" --run-dir ".\evaluation\pair_review\run10" --out ".\evaluation\pair_review\results\evaluation_results.json"
```

Include every run folder you completed. The output file is written once and never overwritten; to re-evaluate, write to a new file name.

Read `reporting_status` in the JSON:

- `INSUFFICIENT_DATA`: the sample is below the reporting gate (30 pairs, 5 SUPPORTING, 5 CONTRADICTING, 10 companies). Report the numbers as descriptive only.
- `REPORTABLE_WITH_CAVEATS`: report the metrics with the caveats in the file. They describe evidence-pair agreement with the NLI model. They are not company-level greenwashing accuracy.

---

## 6. Expected files after each step (summary)

| Step | Command | Files created |
|---|---|---|
| 5.1 | inspect | none (read-only) |
| 5.2 | make-packets | `packets\packet_a_R1.csv`, `packets\packet_b_R2.csv`, `packets\packet_build_report.json`, `coordinator_only\coordinator_manifest.csv` |
| 5.3 | (reviewers) | `returned_R1.csv`, `returned_R2.csv` placed by you |
| 5.4 | validate | none (prints counts) |
| 5.5 | reconcile | `recon\agreement_report.json`, `recon\consensus_labels.csv`, `recon\adjudication_queue.csv`, `recon\adjudication_expected_pairs.json` |
| 5.6 | (adjudicator) | `adjudicated_ADJ.csv` placed by you |
| 5.7 | finalize | `final_labels.csv` |
| 5.8 | evaluate | `evaluation\pair_review\results\evaluation_results.json` |

---

## 7. Common errors and safe recovery

| Message | Meaning | What to do |
|---|---|---|
| `ModuleNotFoundError: No module named 'evaluation'` | Not in the project folder | `cd` to the folder containing `app.py`. |
| `Saved analysis has no analysis_id` | File was not written by the app, or was edited | Choose another file. Do not add an ID by hand. |
| `No eligible claim-evidence pairs were found` | No company claim with a valid source, or no eligible external evidence | Record the company as "no usable pairs" in your log and choose another company. Do not lower thresholds. |
| `Source text was changed ... column 'X'` | A reviewer edited a source column | Send the reviewer the original packet file again and ask them to change only `label`, `reviewer_id`, `rationale`. |
| `Invalid label` | Typo in a label | Correct the label to one of the four allowed values. |
| `requires a rationale` | CONTRADICTING or UNCERTAIN without a rationale | The reviewer adds a short rationale. |
| `missing from the returned file` / `Duplicate pair_id` | Rows deleted, added or duplicated | Ask for the file to be returned with every row, in the original order. |
| `Refusing to overwrite existing file` | A file already exists | Use a new run folder or a new output file name. Never delete an existing result to make a command run. |
| `A packet run already exists` | `make-packets` was run twice in the same folder | Use a new run folder. |
| `Both returned files list the same reviewer_id` / `must be independent` | One person labelled both packets | Use two different reviewers. |
| `Adjudicator for ... is one of the two original reviewers` | Adjudicator is R1 or R2 | Use a third person. |
| `N disagreement(s) were deleted from the adjudication file` | Adjudicator removed rows | Send the unchanged `recon\adjudication_queue.csv` again. |
| `Missing adjudication_expected_pairs.json` | `finalize` was given a folder without reconcile output | Run `reconcile` first, and pass its `recon\consensus_labels.csv`. |
| `pair_id appears in more than one run` | The same analysis was used in two run folders | Remove the duplicate run from the `evaluate` command. Do not delete its files. |
| Non-ASCII characters look wrong in a file | Excel saved the CSV in ANSI encoding | Re-save as "CSV UTF-8". Validation will reject any changed source text, so you will find out. |

If a run becomes inconsistent and you are not sure why, stop, keep the folder unchanged, and send me the run folder's `packet_build_report.json`, `recon\agreement_report.json` and the error text. Do not delete anything.

---

## 8. Checklist: confirm the evidence is genuine (before and during labelling)

Do this for a sample of pairs in each run, and for any pair a reviewer labels SUPPORTING or CONTRADICTING:

1. The claim text in the packet appears on the company source page at `claim_source_url`. Open the link, press Ctrl+F, and search for the first 8 words of the claim.
2. The evidence excerpt appears on `evidence_source_url`. Open the link, search for the first 8 words of the excerpt.
3. The evidence URL starts with `https://`. Reject links that are redirects, aggregators, or copies on a different site from the named publisher.
4. The publisher is not the company itself (check `evidence_publisher` and the domain). Evidence from the company's own site is not independent.
5. The publication date (`evidence_publication_date`) is present and matches the period the evidence describes. If it is missing, use UNCERTAIN.
6. If a page cannot be opened, label from the packet text only and say so in the rationale, or use UNCERTAIN.
7. Record any pair that fails these checks as UNCERTAIN with a rationale. Do not delete it.

`inspect` checks the record's structure only. It cannot confirm that a page says what the saved record says. Steps 1 to 7 are the human check.

---

## 9. What is still blocked

- **Real company analyses:** blocked until you run section 3 on your computer.
- **Two independent human labels for each pair:** blocked until reviewers complete section 5.3.
- **Adjudication:** blocked until disagreements exist and a third person completes section 5.6.
- **Metrics on real data:** blocked until section 5.8 runs on at least 10 companies and the sample passes the reporting gate.
- **Company-level greenwashing accuracy, risk-prediction accuracy, calibration:** not computed by this workflow. They need different ground truth.
