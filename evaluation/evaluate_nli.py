#!/usr/bin/env python3
"""Evaluate fixed NLI predictions against adjudicated, compatible ESG pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABELS = ("entailment", "neutral", "contradiction")
ALLOWED_REVIEWER_LABELS = set(LABELS) | {
    "insufficient_evidence",
    "uncertain",
}
ALLOWED_EVIDENCE_ROLES = {
    "external_source",
    "external_regulatory_evidence",
    "configured_environmental_source",
    "independent_third_party_source",
}
REQUIRED_COLUMNS = {
    "pair_id",
    "company",
    "company_group",
    "report_id",
    "split_group",
    "claim_group",
    "claim",
    "claim_source_url",
    "claim_source_role",
    "evidence_source_title",
    "evidence_publisher",
    "evidence_source_url",
    "evidence_source_role",
    "evidence_excerpt",
    "evidence_availability",
    "eligibility_decision",
    "metric_match",
    "unit_match",
    "period_match",
    "scope_match",
    "baseline_match",
    "organizational_boundary_match",
    "methodology_match",
    "performance_status_match",
    "claim_baseline_year",
    "claim_organizational_boundary",
    "claim_methodology",
    "claim_performance_status",
    "evidence_baseline_year",
    "evidence_organizational_boundary",
    "evidence_methodology",
    "evidence_performance_status",
    "reviewer_1_id",
    "reviewer_1_label",
    "reviewer_1_notes",
    "reviewer_2_id",
    "reviewer_2_label",
    "reviewer_2_notes",
    "adjudicated_label",
    "adjudicator_id",
    "review_status",
    "review_notes",
    "adjudication_notes",
    "model_name",
    "model_prediction",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "nli_labels.csv"
SPLIT_SEED = 42
BOOTSTRAP_RESAMPLES = 1000
MIN_BOOTSTRAP_GROUPS_PER_LABEL = 10


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
            and (parsed.port is None or 1 <= parsed.port <= 65535)
        )
    except (TypeError, ValueError):
        return False


def load_nli_annotations(csv_path: Path) -> pd.DataFrame:
    """Load only independently adjudicated eligible pairs for NLI evaluation."""

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"NLI dataset not found: {csv_path}. "
            "No NLI performance evaluation was run."
        )
    frame = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "NLI dataset is missing required columns: "
            + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("NLI dataset contains no annotated pairs.")

    text_columns = (
        "pair_id",
        "company",
        "company_group",
        "report_id",
        "split_group",
        "claim_group",
        "claim",
        "claim_source_url",
        "claim_source_role",
        "evidence_source_title",
        "evidence_publisher",
        "evidence_source_url",
        "evidence_source_role",
        "evidence_excerpt",
        "evidence_availability",
        "eligibility_decision",
        "metric_match",
        "unit_match",
        "period_match",
        "scope_match",
        "baseline_match",
        "organizational_boundary_match",
        "methodology_match",
        "performance_status_match",
        "claim_baseline_year",
        "claim_organizational_boundary",
        "claim_methodology",
        "claim_performance_status",
        "evidence_baseline_year",
        "evidence_organizational_boundary",
        "evidence_methodology",
        "evidence_performance_status",
        "reviewer_1_id",
        "reviewer_1_label",
        "reviewer_1_notes",
        "reviewer_2_id",
        "reviewer_2_label",
        "reviewer_2_notes",
        "adjudicated_label",
        "adjudicator_id",
        "review_status",
        "review_notes",
        "model_name",
        "model_prediction",
    )
    for column in text_columns:
        frame[column] = frame[column].map(
            lambda value: value.strip() if isinstance(value, str) else ""
        )
        if frame[column].eq("").any():
            rows = (frame.index[frame[column].eq("")] + 2).tolist()
            raise ValueError(
                f"Empty values in {column!r} at CSV rows: {rows}."
            )
    same_reviewer = frame["reviewer_1_id"] == frame["reviewer_2_id"]
    if same_reviewer.any():
        rows = (frame.index[same_reviewer] + 2).tolist()
        raise ValueError(
            "reviewer_1_id and reviewer_2_id must identify distinct reviewers; "
            f"CSV rows: {rows}."
        )
    reused_adjudicator = frame["adjudicator_id"].isin(
        frame["reviewer_1_id"]
    ) | frame["adjudicator_id"].isin(frame["reviewer_2_id"])
    if reused_adjudicator.any():
        rows = (frame.index[reused_adjudicator] + 2).tolist()
        raise ValueError(
            "adjudicator_id must be distinct from both reviewers; "
            f"CSV rows: {rows}."
        )

    for column in ("claim_source_url", "evidence_source_url"):
        invalid = ~frame[column].map(_is_http_url)
        if invalid.any():
            rows = (frame.index[invalid] + 2).tolist()
            raise ValueError(
                "Source URLs must be absolute HTTP(S) URLs; "
                f"invalid CSV rows: {rows}."
            )

    for column in (
        "claim_source_role",
        "evidence_source_role",
        "evidence_availability",
        "eligibility_decision",
        "metric_match",
        "unit_match",
        "period_match",
        "scope_match",
        "baseline_match",
        "organizational_boundary_match",
        "methodology_match",
        "performance_status_match",
        "reviewer_1_label",
        "reviewer_2_label",
        "adjudicated_label",
        "review_status",
        "model_prediction",
    ):
        frame[column] = frame[column].str.casefold()

    if not frame["claim_source_role"].eq("company_source").all():
        raise ValueError("NLI pairs require a company-authored claim source.")
    invalid_roles = sorted(
        set(frame["evidence_source_role"]) - ALLOWED_EVIDENCE_ROLES
    )
    if invalid_roles:
        raise ValueError(
            "NLI pairs require an eligible external evidence role; found: "
            + ", ".join(invalid_roles)
        )
    if not frame["evidence_availability"].eq("available_eligible").all():
        raise ValueError(
            "NLI evaluation requires available eligible evidence; "
            "missing evidence is not a neutral or contradiction label."
        )
    if not frame["eligibility_decision"].eq("eligible").all():
        raise ValueError("Only independently reviewed eligible pairs can be evaluated.")
    compatibility_values = {"compatible", "not_applicable"}
    compatibility_columns = (
        "metric_match",
        "unit_match",
        "period_match",
        "scope_match",
        "baseline_match",
        "organizational_boundary_match",
        "methodology_match",
        "performance_status_match",
    )
    for column in compatibility_columns:
        invalid = sorted(set(frame[column]) - compatibility_values)
        if invalid:
            raise ValueError(
                f"{column} must be compatible or not_applicable; found: "
                + ", ".join(invalid)
            )
    baseline_year_pattern = re.compile(r"(?:19|20)\d{2}")
    baseline_years = []
    for column in ("claim_baseline_year", "evidence_baseline_year"):
        baseline_years.append(
            frame[column].str.fullmatch(baseline_year_pattern).fillna(False)
        )
    baseline_not_applicable = (
        frame["claim_baseline_year"].str.casefold().eq("not applicable")
        & frame["evidence_baseline_year"].str.casefold().eq("not applicable")
    )
    baseline_equal = (
        frame["claim_baseline_year"].str.casefold()
        == frame["evidence_baseline_year"].str.casefold()
    ) & baseline_years[0] & baseline_years[1]
    baseline_valid = (
        baseline_not_applicable
        & frame["baseline_match"].eq("not_applicable")
    ) | (
        baseline_equal
        & frame["baseline_match"].eq("compatible")
    )
    if not baseline_valid.all():
        rows = (frame.index[~baseline_valid] + 2).tolist()
        raise ValueError(
            "baseline_match must reflect identical reviewed baseline years "
            "or not applicable on both sides; CSV rows: "
            f"{rows}."
        )
    non_baseline_matches = (
        "metric_match",
        "unit_match",
        "period_match",
        "scope_match",
        "organizational_boundary_match",
        "methodology_match",
        "performance_status_match",
    )
    qualitative_pairs = frame[list(non_baseline_matches)].eq(
        "not_applicable"
    ).all(axis=1)
    invalid_quantitative = ~qualitative_pairs & (
        frame[list(non_baseline_matches)].ne("compatible").any(axis=1)
    )
    invalid_baseline = (
        ~qualitative_pairs
        & ~frame["baseline_match"].isin({"compatible", "not_applicable"})
    ) | (
        qualitative_pairs
        & frame["baseline_match"].ne("not_applicable")
    )
    if invalid_quantitative.any() or invalid_baseline.any():
        rows = (
            frame.index[invalid_quantitative | invalid_baseline] + 2
        ).tolist()
        raise ValueError(
            "NLI pairs must be compatible on every applicable measurement "
            "dimension; use not_applicable for all measurement dimensions "
            "only when the pair is qualitative. CSV rows: "
            f"{rows}."
        )

    if not frame["review_status"].eq("adjudicated").all():
        raise ValueError(
            "Only rows with review_status='adjudicated' can be evaluated."
        )
    for column in ("reviewer_1_label", "reviewer_2_label"):
        invalid = sorted(set(frame[column]) - ALLOWED_REVIEWER_LABELS)
        if invalid:
            raise ValueError(
                f"{column} contains invalid labels: " + ", ".join(invalid)
            )
    invalid = sorted(set(frame["adjudicated_label"]) - set(LABELS))
    if invalid:
        raise ValueError(
            "adjudicated_label must be entailment, neutral, or contradiction; "
            + ", ".join(invalid)
        )
    invalid = sorted(set(frame["model_prediction"]) - (set(LABELS) | {"unavailable"}))
    if invalid:
        raise ValueError(
            "model_prediction must be entailment, neutral, contradiction, "
            "or unavailable; found: " + ", ".join(invalid)
        )

    disagreement = (
        frame["reviewer_1_label"] != frame["reviewer_2_label"]
    ) | (
        frame["adjudicated_label"] != frame["reviewer_1_label"]
    ) | (
        frame["adjudicated_label"] != frame["reviewer_2_label"]
    )
    notes = frame.get(
        "adjudication_notes",
        pd.Series("", index=frame.index),
    )
    missing_notes = notes.map(
        lambda value: not isinstance(value, str) or not value.strip()
    ) & disagreement
    if missing_notes.any():
        rows = (frame.index[missing_notes] + 2).tolist()
        raise ValueError(
            f"Disagreements require adjudication_notes; CSV rows: {rows}."
        )

    for column in ("model_confidence",):
        if column in frame:
            converted = pd.to_numeric(frame[column], errors="coerce")
            invalid = frame[column].notna() & converted.isna()
            if invalid.any():
                raise ValueError(f"{column} must be numeric when provided.")
            non_finite = converted.dropna().map(
                lambda value: not math.isfinite(float(value))
            )
            if non_finite.any() or not converted.dropna().between(0, 1).all():
                raise ValueError(f"{column} must be between 0 and 1.")
            frame[column] = converted

    if frame["pair_id"].duplicated().any():
        raise ValueError("pair_id values must be unique.")
    frame["_normalized_claim"] = frame["claim"].str.casefold().str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    frame["_normalized_evidence"] = frame["evidence_excerpt"].str.casefold().str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    if frame.duplicated(
        ["_normalized_claim", "evidence_source_url", "_normalized_evidence"]
    ).any():
        raise ValueError("Duplicate claim-evidence pairs must not be counted twice.")
    inconsistent_claim_groups = (
        frame.groupby("claim_group")["split_group"].nunique() > 1
    )
    if inconsistent_claim_groups.any():
        raise ValueError("All rows in a claim_group must share one split_group.")
    company_group_counts = frame.groupby("company")["split_group"].nunique()
    if (company_group_counts > 1).any():
        raise ValueError(
            "All reports for a company must share one split_group to keep "
            "companies out of both training and test data."
        )
    group_company_counts = frame.groupby("company_group")["split_group"].nunique()
    if (group_company_counts > 1).any():
        raise ValueError(
            "Each company_group must remain in a single split_group."
        )
    split_company_groups = frame.groupby("split_group")["company_group"].nunique()
    if (split_company_groups > 1).any():
        raise ValueError(
            "A split_group cannot combine distinct company_group values."
        )
    return frame.reset_index(drop=True)


def _group_splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups_per_label = frame.groupby("adjudicated_label")["split_group"].nunique()
    insufficient = {
        label: int(groups_per_label.get(label, 0))
        for label in LABELS
        if groups_per_label.get(label, 0) < 5
    }
    if insufficient:
        raise ValueError(
            "At least five distinct split_group values per NLI label are "
            "required for a stratified 60/20/20 split; found "
            + ", ".join(f"{label}={count}" for label, count in insufficient.items())
        )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SPLIT_SEED,
    )
    folds = [
        set(frame.iloc[test_indices]["split_group"])
        for _, test_indices in splitter.split(
            frame,
            frame["adjudicated_label"],
            groups=frame["split_group"],
        )
    ]
    test_groups = folds[0]
    validation_groups = folds[1]
    train_groups = set(frame["split_group"]) - test_groups - validation_groups
    group_sets = {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups,
    }
    result = {
        name: frame[frame["split_group"].isin(group_ids)].copy()
        for name, group_ids in group_sets.items()
    }
    for name, split in result.items():
        if set(split["adjudicated_label"]) != set(LABELS):
            raise ValueError(
                f"The {name} split does not contain all three NLI labels; "
                "collect more independently grouped examples."
            )
    return result


def _metrics(actual: list[str], predicted: list[str]) -> dict:
    matrix = [
        [
            sum(a == actual_label and p == predicted_label
                for a, p in zip(actual, predicted))
            for predicted_label in LABELS
        ]
        for actual_label in LABELS
    ]
    per_class = {}
    for index, label in enumerate(LABELS):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(LABELS)) if row != index)
        false_negative = sum(matrix[index][column] for column in range(len(LABELS)) if column != index)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            ),
            "support": sum(value == label for value in actual),
        }
    total = len(actual)
    return {
        "accuracy": (
            sum(matrix[index][index] for index in range(len(LABELS))) / total
            if total else 0.0
        ),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(LABELS),
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows_actual_columns_predicted": matrix,
        },
        "per_class": per_class,
    }


def _test_confidence_intervals(frame: pd.DataFrame) -> dict:
    group_counts = frame.groupby("adjudicated_label")["split_group"].nunique()
    insufficient = {
        label: int(group_counts.get(label, 0))
        for label in LABELS
        if group_counts.get(label, 0) < MIN_BOOTSTRAP_GROUPS_PER_LABEL
    }
    if insufficient:
        return {
            "status": "not_reported",
            "reason": (
                f"At least {MIN_BOOTSTRAP_GROUPS_PER_LABEL} independent held-out "
                "company groups per label are required."
            ),
            "company_groups_per_label": {
                label: int(group_counts.get(label, 0)) for label in LABELS
            },
        }

    rows_by_group = {
        group: group_frame
        for group, group_frame in frame.groupby("split_group", sort=True)
    }
    groups = list(rows_by_group)
    rng = random.Random(SPLIT_SEED)
    accuracy_values = []
    macro_f1_values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_groups = rng.choices(groups, k=len(groups))
        sampled_rows = pd.concat(
            [rows_by_group[group] for group in sampled_groups],
            ignore_index=True,
        )
        metrics = _metrics(
            sampled_rows["adjudicated_label"].tolist(),
            sampled_rows["model_prediction"].tolist(),
        )
        accuracy_values.append(metrics["accuracy"])
        macro_f1_values.append(metrics["macro_f1"])

    def interval(values: list[float]) -> list[float]:
        ordered = sorted(values)
        return [
            ordered[int(0.025 * (len(ordered) - 1))],
            ordered[int(0.975 * (len(ordered) - 1))],
        ]

    return {
        "status": "reported",
        "method": "percentile cluster bootstrap resampling company groups",
        "level": 0.95,
        "resamples": BOOTSTRAP_RESAMPLES,
        "company_groups_per_label": {
            label: int(group_counts.get(label, 0)) for label in LABELS
        },
        "accuracy": interval(accuracy_values),
        "macro_f1": interval(macro_f1_values),
    }


def evaluate_dataset(frame: pd.DataFrame) -> dict:
    """Evaluate fixed predictions once on held-out company groups."""

    splits = _group_splits(frame)
    training_labels = splits["train"]["adjudicated_label"]
    majority_label = training_labels.value_counts().sort_values(
        ascending=False
    ).index[0]
    test = splits["test"]
    available_test = test[test["model_prediction"] != "unavailable"]
    actual = available_test["adjudicated_label"].tolist()
    predicted = available_test["model_prediction"].tolist()
    report = {
        "evaluation_type": (
            "fixed NLI predictions compared with blind, adjudicated human "
            "labels; no threshold tuning or model fitting"
        ),
        "split_seed": SPLIT_SEED,
        "label_definition": (
            "Human judgment of whether the cited evidence entails, is neutral "
            "toward, or contradicts the exact claim; not a greenwashing label."
        ),
        "reviewer_agreement": {
            "exact_agreement": float(
                (
                    frame["reviewer_1_label"]
                    == frame["reviewer_2_label"]
                ).mean()
            ),
            "disagreement_rows": int(
                (
                    frame["reviewer_1_label"]
                    != frame["reviewer_2_label"]
                ).sum()
            ),
            "rows": len(frame),
        },
        "split_summary": {
            name: {
                "rows": len(split),
                "company_groups": int(split["split_group"].nunique()),
                "label_counts": {
                    label: int(count)
                    for label, count in split["adjudicated_label"].value_counts().items()
                },
            }
            for name, split in splits.items()
        },
        "test_prediction_coverage": (
            len(available_test) / len(test) if len(test) else 0.0
        ),
        "test_unavailable_predictions": int(
            test["model_prediction"].eq("unavailable").sum()
        ),
        "limitations": [
            "The held-out set is grouped by company; this estimates "
            "generalization only to companies represented by the sampling frame.",
            "NLI labels evaluate evidence relation, not greenwashing or risk concern.",
            "Model confidence is not calibrated probability.",
            "This report is valid only for reviewed, eligible, compatible pairs.",
        ],
    }
    if actual:
        report["majority_class_baseline_label"] = majority_label
        report["majority_class_baseline"] = _metrics(
            actual,
            [majority_label] * len(actual),
        )
        report["test_metrics"] = _metrics(actual, predicted)
        report["test_confidence_intervals"] = _test_confidence_intervals(
            available_test
        )
    else:
        report["majority_class_baseline_label"] = majority_label
        report["majority_class_baseline"] = None
        report["test_metrics"] = None
        report["test_metrics_status"] = (
            "No test pair had an available model prediction; no NLI metrics "
            "were computed."
        )
        report["test_confidence_intervals"] = {
            "status": "not_reported",
            "reason": "No available model predictions in held-out test data.",
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate NLI predictions on independently adjudicated ESG evidence pairs."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"NLI annotation CSV (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args(argv)
    try:
        frame = load_nli_annotations(args.dataset)
        report = evaluate_dataset(frame)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
