import os
import numpy as np
import pandas as pd

from risk_model import train_risk_model, FEATURE_COLUMNS


def _heuristic_explanations(predictions, company_name):
    explanations = []
    for _, row in predictions.reset_index(drop=True).iterrows():
        def number(feature):
            value = pd.to_numeric(row.get(feature), errors="coerce")
            return float(value) if pd.notna(value) else 0.0

        evidence_available = number("evidence_available") > 0
        contradictions = int(number("nli_contradiction"))
        entailments = int(number("nli_entailment"))
        neutrals = int(number("nli_neutral"))

        if not evidence_available:
            feature = "evidence_available"
            explanation = (
                "No matched external evidence was available. Missing "
                "evidence is not a negative finding or proof of wrongdoing."
            )
        elif contradictions:
            feature = "nli_contradiction"
            explanation = (
                f"The NLI model returned {contradictions} contradiction "
                "signal(s). Manually check source context; this is not proof "
                "that the claim is false."
            )
        elif entailments:
            feature = "nli_entailment"
            explanation = (
                f"The NLI model returned {entailments} entailment signal(s). "
                "This is not independent verification of the claim."
            )
        elif neutrals:
            feature = "nli_neutral"
            explanation = (
                f"The NLI model returned {neutrals} neutral signal(s); the "
                "evidence does not clearly support or contradict the claim."
            )
        else:
            feature = "nli_contradiction"
            explanation = (
                "No usable NLI classification was available. The heuristic "
                "score is not evidence of support or contradiction."
            )

        value = row.get(feature)
        explanations.append({
            "company": company_name,
            "claim_id": int(row["claim_id"]),
            "claim": str(row["claim"]),
            "demonstration_risk_category": str(
                row["demonstration_risk_category"]
            ),
            "risk_score": float(row["risk_score"]),
            "feature": feature,
            "feature_value": float(value) if pd.notna(value) else None,
            "explanation": explanation,
        })
    return explanations


def explain_risk_predictions(predictions, model, company_name):
    """Explain the active heuristic without inventing model contributions."""

    if predictions.empty:
        return []

    if model is not None:
        raise ValueError(
            "Heuristic risk scores cannot be explained with a separate "
            "trained model."
        )

    required_columns = [
        "claim_id",
        "claim",
        *FEATURE_COLUMNS,
        "demonstration_risk_category",
        "risk_score",
    ]
    missing = [
        column for column in required_columns
        if column not in predictions.columns
    ]
    if missing:
        raise ValueError(
            f"Missing explanation columns: {', '.join(missing)}"
        )

    # Only scored claims have a heuristic score to explain. Unassessed claims
    # (no eligible evidence, or evidence without a completed NLI run) are
    # skipped here rather than cast to a number.
    scored = predictions[
        predictions["risk_score"].notna()
        & predictions["demonstration_risk_category"].isin(
            {"LOW", "MEDIUM", "HIGH"}
        )
    ]
    if scored.empty:
        return []
    return _heuristic_explanations(scored, company_name)


def explain_predictions():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    feature_file = os.path.join(
        base_dir, "output", "esg_features.csv"
    )

    prediction_file = os.path.join(
        base_dir, "output", "esg_risk_predictions.csv"
    )

    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(feature_file):
        print("ERROR: Feature file not found.")
        print("Run feature_builder.py first.")
        return

    if not os.path.exists(prediction_file):
        print("ERROR: Risk prediction file not found.")
        print("Run risk_predictor.py first.")
        return

    df = pd.read_csv(feature_file)
    predictions_df = pd.read_csv(prediction_file)

    if (
        "risk_score_status" in predictions_df.columns
        and predictions_df["risk_score_status"].eq("HEURISTIC").all()
    ):
        print(
            "The active risk workflow produces heuristic scores, not "
            "probabilities. Use the risk explanations in the app; no SHAP "
            "model explanation was generated."
        )
        return

    import matplotlib.pyplot as plt
    import shap

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("ERROR: Missing feature columns:", missing_columns)
        return

    if len(df) != len(predictions_df):
        print("ERROR: Feature and prediction row counts differ.")
        print("Run risk_predictor.py again before explaining.")
        return

    X = df[FEATURE_COLUMNS].copy()
    X = X.fillna(0)

    # Use exactly the same shared model as risk_predictor.py.
    model = train_risk_model()

    # Confirm the shared model reproduces the saved predictions.
    calculated_probabilities = model.predict_proba(X)[:, 1]

    if "demonstration_risk_probability" not in predictions_df.columns:
        print("ERROR: Risk probability column is missing.")
        return

    saved_probabilities = predictions_df[
        "demonstration_risk_probability"
    ].to_numpy()

    if not all(
        abs(float(a) - float(b)) < 1e-5
        for a, b in zip(
            calculated_probabilities,
            saved_probabilities,
        )
    ):
        print("WARNING: Saved predictions do not match the current model.")
        print("Run risk_predictor.py again before explaining.")

    # Calculate SHAP contributions for the shared model.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print("=" * 65)
    print("ESG GREENWASHING RISK - EXPLAINABILITY")
    print("=" * 65)

    for i, row in enumerate(shap_values):
        if "claim_id" in df.columns:
            claim_id = df.iloc[i]["claim_id"]
        else:
            claim_id = i + 1

        print(f"\nClaim ID: {claim_id}")

        contributions = sorted(
            zip(FEATURE_COLUMNS, row),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )

        print("Top three model feature contributions:")

        for feature, value in contributions[:3]:
            print(f"  {feature}: {float(value):.4f}")

        if "demonstration_risk_category" in predictions_df.columns:
            category = predictions_df.iloc[i][
                "demonstration_risk_category"
            ]
            print(f"Demonstration risk category: {category}")

    # Save SHAP summary plot.
    plot_file = os.path.join(
        output_dir, "shap_summary.png"
    )

    shap.summary_plot(
        shap_values,
        X,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\nSHAP summary plot saved to:")
    print(plot_file)

    print("\n" + "=" * 65)
    print("EXPLAINABILITY ANALYSIS COMPLETED")
    print("=" * 65)

    print(
        "\nIMPORTANT LIMITATION:"
        "\nThis model uses six manually labelled demonstration examples."
        "\nSHAP explains how features influence this model's output."
        "\nIt does not establish whether a company has engaged in"
        "\ngreenwashing. All risk predictions are unvalidated."
    )


if __name__ == "__main__":
    explain_predictions()