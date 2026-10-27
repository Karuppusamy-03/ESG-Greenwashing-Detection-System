import math
import os
import pandas as pd

from risk_model import FEATURE_COLUMNS


def _as_float(value, default=0.0):
    if value is None or value is pd.NaT:
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _to_bool_flag(value):
    if value is None or value is pd.NaT:
        return 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return 1
        if lowered in {"0", "false", "no", "n", "off", ""}:
            return 0
    try:
        return 1 if bool(float(value)) else 0
    except (TypeError, ValueError):
        return 0


NLI_COLUMNS = ("nli_entailment", "nli_contradiction", "nli_neutral")

NO_EVIDENCE_REASON = "No eligible external evidence matched this claim."
NLI_PARTIAL_REASON = (
    "Only some eligible evidence for this claim received a completed NLI "
    "result, so the claim is unassessed rather than scored on partial evidence."
)
NLI_UNAVAILABLE_REASON = (
    "Evidence was found, but NLI verification did not produce a usable "
    "result, so the claim is unassessed."
)


def nli_result_missing(row):
    """True when no NLI counts were recorded for the row at all.

    A completed NLI run stores integer counts (including zero). A missing run
    stores no value in any NLI column. The two must not be confused.
    """
    return all(
        pd.isna(row.get(column)) if column in row else True
        for column in NLI_COLUMNS
    )


def nli_coverage_incomplete(row):
    """True when some eligible evidence pair for the claim has no completed NLI.

    Older feature files lack the coverage column. For those rows this returns
    False, so their behaviour is unchanged; they are not newly rejected.
    """
    if "nli_coverage_complete" not in row:
        return False
    value = pd.to_numeric(row.get("nli_coverage_complete"), errors="coerce")
    return bool(pd.notna(value) and int(value) == 0)


def _heuristic_risk_score(row):
    # This score is a transparent heuristic for review support, not a validated
    # greenwashing classifier. It is intentionally deterministic and traceable to
    # the input features. Missing evidence is not assigned a score. Low
    # similarity, neutral NLI, and unavailable NLI outputs do not count as proof
    # of greenwashing.
    similarity = min(max(_as_float(row.get("similarity"), 0.0), 0.0), 1.0)
    evidence_available = _to_bool_flag(row.get("evidence_available"))
    nli_entailment = max(0, int(_as_float(row.get("nli_entailment"), 0.0)))
    nli_contradiction = max(0, int(_as_float(row.get("nli_contradiction"), 0.0)))
    nli_neutral = max(0, int(_as_float(row.get("nli_neutral"), 0.0)))

    if evidence_available == 0:
        return None

    # Evidence without a completed NLI result is unassessed. A completed run
    # with zero counts is a real outcome and is still scored below.
    if nli_result_missing(row):
        return None

    # Partial NLI coverage (D2): scoring on a subset of the eligible evidence
    # would silently treat unassessed pairs as if they had been evaluated.
    if nli_coverage_incomplete(row):
        return None

    if nli_contradiction > 0:
        return min(100.0, 72.0 + similarity * 18.0 + nli_contradiction * 8.0)

    if nli_entailment > 0:
        return max(0.0, min(30.0, 8.0 + (1.0 - similarity) * 18.0))

    if nli_neutral > 0:
        return min(55.0, 30.0 + similarity * 20.0 + nli_neutral * 8.0)

    # If no valid NLI signal is available, keep the heuristic in a low-risk
    # band rather than treating missing evidence as proof of greenwashing.
    base = 10.0 + max(0.0, (0.75 - similarity) * 20.0)
    return min(30.0, base)


def predict_risk_from_features(features, model=None):
    """Return a deterministic heuristic risk score for the supplied feature rows."""

    if features.empty:
        return features.copy()

    required_columns = ["claim_id"]
    missing = [
        column
        for column in required_columns
        if column not in features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing risk feature columns: {', '.join(missing)}"
        )

    result = features.copy()
    for column in FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["evidence_available"] = result["evidence_available"].map(
        _to_bool_flag
    )

    if "percentage_difference_missing" not in result.columns:
        result["percentage_difference_missing"] = (
            result["percentage_difference"].isna().astype(int)
        )
    else:
        percentage_missing = result.get("percentage_difference_missing")
        result["percentage_difference_missing"] = pd.to_numeric(
            percentage_missing,
            errors="coerce",
        ).fillna(result["percentage_difference"].isna().astype(int))

    eligible = result["evidence_available"].eq(1)
    result["risk_score"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Float64",
    )
    if eligible.any():
        result.loc[eligible, "risk_score"] = result.loc[eligible].apply(
            _heuristic_risk_score,
            axis=1,
        )
    # A row is scored only when the heuristic produced a number. Evidence with
    # no completed NLI run yields NA and must stay unassessed: it is never
    # given a status, class, or category of LOW, MEDIUM, or HIGH.
    scored = eligible & result["risk_score"].notna()
    result["risk_score_status"] = "NOT_PREDICTED"
    result.loc[scored, "risk_score_status"] = "HEURISTIC"
    result["risk_unavailable_reason"] = ""
    result.loc[~eligible, "risk_unavailable_reason"] = NO_EVIDENCE_REASON
    result.loc[eligible & ~scored, "risk_unavailable_reason"] = (
        NLI_UNAVAILABLE_REASON
    )
    if "nli_coverage_complete" in result.columns:
        partial = eligible & ~scored & (
            pd.to_numeric(result["nli_coverage_complete"], errors="coerce") == 0
        ) & result[list(NLI_COLUMNS)].notna().any(axis=1)
        result.loc[partial, "risk_unavailable_reason"] = NLI_PARTIAL_REASON
    result["predicted_risk_class"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )
    scored_scores = result.loc[scored, "risk_score"].astype("float64")
    result.loc[scored, "predicted_risk_class"] = (
        scored_scores >= 50
    ).astype("int64").to_numpy()
    # Keep the legacy field empty for existing CSV/database consumers. A
    # heuristic score is not a calibrated probability.
    result["demonstration_risk_probability"] = pd.NA
    result["demonstration_risk_category"] = [
        (
            "HIGH" if score >= 70
            else "MEDIUM" if score >= 35
            else "LOW"
        )
        if is_scored
        else "NOT_PREDICTED"
        for score, is_scored in zip(
            result["risk_score"],
            scored,
        )
    ]
    return result


def predict_risk():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(
        base_dir, "output", "esg_features.csv"
    )

    output_file = os.path.join(
        base_dir, "output", "esg_risk_predictions.csv"
    )

    if not os.path.exists(input_file):
        print("ERROR: Feature file not found.")
        print("Run python feature_builder.py first.")
        return

    df = pd.read_csv(input_file)

    if df.empty:
        print("ERROR: No feature records found.")
        return

    required_columns = ["claim_id"] + FEATURE_COLUMNS
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        print("ERROR: Missing columns:", missing)
        print("Run python feature_builder.py first.")
        return

    df = predict_risk_from_features(df)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print("=" * 65)
    print("ESG GREENWASHING RISK PREDICTION")
    print("=" * 65)

    for _, row in df.iterrows():
        print("\nClaim ID:", row["claim_id"])
        if row["risk_score_status"] != "HEURISTIC":
            print(
                "Prediction: NOT PREDICTED — no eligible evidence match."
            )
            continue
        print("Predicted class:", row["predicted_risk_class"])
        print(f"Heuristic review score: {row['risk_score']:.1f}/100")
        print("Category:", row["demonstration_risk_category"])

    print("\nResults saved to:", output_file)
    print(
        "\nWARNING: Scores are experimental investigation signals, "
        "not probabilities or proof of greenwashing."
    )


if __name__ == "__main__":
    predict_risk()
