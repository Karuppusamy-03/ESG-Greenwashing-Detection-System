# Reviewer guide: claim-evidence pair labels

You will label pairs. Each pair is one **company claim** and one **evidence sentence** from an external source. You are judging whether the evidence sentence bears on the claim. You are **not** judging whether the company is greenwashing. Never write that conclusion anywhere.

## What you receive

- One packet file (CSV) for you alone. Open it in Excel, or in a text editor.
- Do not look for, request, or open another reviewer's file.
- Do not edit the `pair_id`, `claim`, `evidence`, `*_url`, `*_title`, `*_publisher`, `*_date`, `analysis_id`, or `company` columns. The workflow rejects any change to them.

## Columns you fill in

| Column | What to enter |
|---|---|
| `label` | One of `SUPPORTING`, `CONTRADICTING`, `UNRELATED`, `UNCERTAIN`. Leave blank if you have not decided. |
| `reviewer_id` | Your identifier, exactly as you were given it. |
| `rationale` | Required for `CONTRADICTING` and `UNCERTAIN`. One or two sentences naming the specific detail (year, unit, scope, wording) that decided it. Recommended for all labels. |

Save the file under a new name, for example `returned_<your id>.csv`, keeping the CSV format and UTF-8.

## The four labels

**SUPPORTING**: read on its own, the evidence states the same measurable fact or commitment as the claim, for the same company, the same reporting period, the same metric, the same unit, and the same scope and boundary. Confirmation of the claim's specific wording counts.

**CONTRADICTING**: read on its own, the evidence states a *different* value, outcome or status for the same company, metric, period, unit, scope and boundary. Example: the claim says emissions fell 10%; the evidence says they rose 5% in the same year and scope. A mismatch in period, unit or scope alone is **not** contradiction. Use UNRELATED or UNCERTAIN instead.

**UNRELATED**: the evidence is about a different company or entity, a different topic or metric, or it does not address the claim. Shared words such as "emissions" or "renewable" alone do not make a pair related. A sentence that does not mention the claim's subject at all is UNRELATED, not CONTRADICTING.

**UNCERTAIN**: the evidence may bear on the claim, but you cannot decide. Use it when context is missing, wording is ambiguous, or the source excerpt is truncated.

## Objective rules

1. **Judge the excerpt against the claim.** Use only the `evidence` text and the claim text in the packet. Opening `evidence_source_url` to read the surrounding text is allowed. Record in `rationale` if you did so. If the link cannot be opened, you may still label the pair from the excerpt; if the excerpt is not enough, use UNCERTAIN.
2. **Same entity.** If the evidence is attributed to a subsidiary, a different company with a similar name, or the company is not named, use UNCERTAIN unless the text clearly ties it to the same company. Do not infer ownership from your own knowledge.
3. **Dates.** Compare the year or period stated in the evidence with the one in the claim. A different year for the same metric is a mismatch of period, so it is UNCERTAIN (or UNRELATED if the topic differs), never CONTRADICTING. If the evidence has no date, and the claim depends on one, use UNCERTAIN.
4. **Units.** Different units (tonnes against megatonnes; MWh against GWh; percentage against absolute value) are not compared by you. Use UNCERTAIN unless the evidence itself states the conversion or the same unit.
5. **Targets and forecasts.** A target, commitment, or forecast does not support or contradict an achieved result. Claim "we cut emissions 10%" with evidence "we aim to cut 10% by 2030" is UNRELATED or UNCERTAIN, never SUPPORTING.
6. **Baselines and scope.** If the claim cites a baseline year or scope (for example Scope 1 only) and the evidence differs, use UNCERTAIN.
7. **Unsupported claims.** If the evidence says nothing about the claim, use UNRELATED. Absence of evidence is not contradiction.
8. **Ambiguous wording.** If two readings are both reasonable, use UNCERTAIN and say which two.
9. **No outside knowledge.** Do not use what you know about the company or the world beyond the packet and the linked source.
10. **Do not score the company.** Pair labels never mean the company is or is not greenwashing.

## Worked examples

| Claim | Evidence (excerpt) | Label | Why |
|---|---|---|---|
| We cut Scope 1 emissions 10% in 2024. | Scope 1 emissions fell 10% in 2024, per the verifier. | SUPPORTING | Same metric, period, scope, and fact. |
| We cut Scope 1 emissions 10% in 2024. | Scope 1 emissions rose 5% in 2024. | CONTRADICTING | Same metric, period, scope; opposite outcome. |
| We cut Scope 1 emissions 10% in 2024. | Scope 1 emissions fell 10% in 2023. | UNCERTAIN | Same metric, different year. Period mismatch is not contradiction. |
| We source 50% renewable power. | We aim to source 50% renewable power by 2030. | UNRELATED | Target, not an achieved result. |
| We source 50% renewable power. | Our subsidiary in Germany sources 80% renewable power. | UNCERTAIN | Subsidiary scope is unclear. |
| We source 50% renewable power. | Our packaging is 100% recyclable. | UNRELATED | Different topic. |

## Disagreements

You do not need to resolve disagreements with the other reviewer. The coordinator sends disagreements to a third adjudicator. Just label what you see, with a rationale.

## Common mistakes

- Labelling CONTRADICTING because the year, unit or scope differs. Use UNCERTAIN.
- Labelling SUPPORTING from a target or commitment.
- Writing a company-level verdict in `rationale`.
- Editing the claim or evidence text to make it fit.
- Leaving `reviewer_id` blank on a labelled row. The row is rejected.
