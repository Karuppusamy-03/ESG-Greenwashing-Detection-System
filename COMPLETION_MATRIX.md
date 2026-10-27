# ESG Greenwashing Detection: Completion Matrix (verification run, 2026-10-26)

## Scope and limits of this run

- **Source reviewed:** `a54aa25d-V1_ESG-Greenwashing-Detection-updated-completed.zip` (V1). The file named `...-completed-v2.zip` was **not present** in the environment.
- **Windows test results:** **not provided.** No Windows output, logs, or `run_all_tests.ps1` results were in the archive, so no Windows-reported defect was reviewed.
- **Environment:** Linux sandbox (Python 3.13, Chromium via Playwright). Only package indexes and the local proxy were reachable. HuggingFace, DuckDuckGo, and company websites returned 403/connect refusals.
- **Existing data:** `data/` (SQLite database, collected documents) and `output/company_analyses/` (saved analyses) are gitignored and **absent from the archive**. No database or saved analysis was read or modified.
- **Project files:** No project source was changed. No defect was reproduced that could be fixed with evidence, so no regression tests were added.

## Test results

| Suite | Result |
|---|---|
| Offline unittest suite (`tests/`, 191 tests, real torch + sentence-transformers installed) | 186 pass, 1 skipped (live, by design), 4 fail |
| The 4 failures | All need the user's local `data/` and `output/` folders (`apple_20261011_024917.json`, `tesla_20261010_182212.json`, `apple_3f8c735de19795cb.json`). Environmental, not code defects. They should pass on your machine if those files exist. |
| Live workflow test (`test_live_workflows`) | Skipped (requires `ESG_LIVE_TESTS=1` and internet) |

## Requirement matrix

Status key: **Verified** = run and observed here. **Implemented** = code and offline tests exist, but not exercised live. **Blocked** = cannot be checked in this environment. **Incomplete** = missing or failing.

| # | Requirement | Status | Evidence / gap |
|---|---|---|---|
| 0 | Fix defects shown by Windows results | **Blocked** | Windows results were not provided. Send them and the v2 zip. |
| 1 | Dynamic search for unseen companies with real web search | **Blocked** | Live search returns `403 Forbidden` from the proxy. The app degrades to "Search unavailable" with no results (verified). Needs a Windows run with internet access. |
| 2 | Discovered domains stay candidates until identity/ownership is established | **Verified in code, offline tests** | `discover_official_domains` returns `candidate_unverified`. Discovered domains get role `external_source_unverified` and domain-name matches get `potential_company_source_unverified`. Neither role is in the matcher's eligible set (`online_evidence_matcher.py`). Covered by `test_company_identity` and `test_matcher_excludes_unverified_external_source_role`, both passing. |
| 3 | Ambiguous names and subsidiaries (Delta Electronics, Air Lines, Dental) | **Implemented, offline tests only** | Bare "Delta" is rejected when followed by Air Lines or Dental, and discovery stops on ambiguous continuations. These pass with mocked results. No live Delta search was possible. |
| 4a | Real document collection | **Blocked** | No outbound access to company or report sites. |
| 4b | Extraction, evidence eligibility, matching | **Implemented, offline tests pass** | Eligibility and matching tests pass with fixtures. Not run on real documents. |
| 4c | NLI execution where eligible evidence exists | **Blocked (live model)** | `cross-encoder/nli-deberta-v3-small` cannot be downloaded here. The offline NLI path was verified: with the model unavailable, a pair becomes `UNAVAILABLE`, `scoring_eligible=False`, with an explanation. No crash. |
| 5 | Risk prediction and inconclusive handling without forcing scores | **Verified offline** | Heuristic tests pass (`test_risk_heuristics`, `test_risk_visualization`). Inconclusive path verified as above. Score is not forced when evidence is missing. Real-evidence scores not verified. |
| 6 | Real Streamlit dashboard and existing UI | **Partly verified** | App starts headless (HTTP 200). Chromium loads "Dashboard Overview" and the "Company ESG Analysis" page, with sidebar intact and no exceptions. Verified on Linux only. The saved-analysis view needs the user's `output/` data. |
| 7 | Existing database and saved analyses, no rewriting of history | **Verified on a copy** | Migration adds columns (`ADD COLUMN`) and never drops or rewrites. Test on a copy with 2 legacy rows: both rows preserved. Saved analyses use `os.link`, which refuses to overwrite. The real `data/` database was **not** touched. |
| 8 | Fix safely fixable defects and add regression tests | **No defect reproduced** | The only failures trace to missing data or the environment. No code changed. |
| 9 | Completion matrix | **Done** (this file) | Based on the results above, not Windows results. |

## Overall status

**Not fully complete.** Dynamic live search, real document collection, the live NLI model, and the Windows results remain **blocked** here. Items 2, 5, and 7 are verified at the offline or copy level, and item 6 is partly verified on Linux.

## Actions needed to finish

1. **Upload the v2 zip and the Windows test output** (`run_all_tests.ps1` console output or logs). I will then fix any demonstrated defects and add regression tests.
2. **Run on your Windows machine:** `.\run_all_tests.ps1 -Live` with internet access, so the live workflow test, real web search, and the NLI model download run. Send back the output.
3. **Run the app** (`streamlit run app.py`) on two previously unseen companies, plus Delta Electronics, Delta Air Lines, and Delta Dental. Record for each: discovered domains and their status, which sources were eligible, whether NLI ran, and the risk output or the inconclusive reason.
4. **Confirm the 4 data-dependent tests** pass on your machine. They need `data/collected_documents/` and `output/company_analyses/` to exist. Nothing needs to be rewritten.
5. **Backups:** Before any run that writes to `data/`, copy `data/greenwashing_risk.db` and `output/` somewhere safe. Nothing destructive was done in this session.
