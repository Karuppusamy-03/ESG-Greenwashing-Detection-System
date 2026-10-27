#!/usr/bin/env python3
"""Evaluate the active heuristic with independently reviewed feature labels."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "risk_labels.csv"
POSITIVE_LABEL = "review_concern"
NEGATIVE_LABEL = "no_review_concern"
ALLOWED_LABELS = {POSITIVE_LABEL, NEGATIVE_LABEL}
SPLIT_SEED = 42
THRESHOLDS = tuple(range(0, 101, 5))

sys.path.insert(0, str(PROJECT_ROOT))

from risk_model import FEATURE_COLUMNS
from risk_predictor import predict_risk_from_features

REQUIRED_COLUMNS = {
    "annotation_id",
    "analysis_id",
    "claim_id",
    "company",
    "report_id",
    "claim",
    "claim_group",
    "split_group",
    "claim_source_url",
    "claim_source_role",
    "evidence_source_url",
    "evidence_source_urls",
    "evidence_source_role",
    "evidence_excerpt",
    "evidence_availability",
    "compatibility_decision",
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
    "support_label",
    "review_notes",
    "reviewer_1_id",
    "reviewer_1_risk_label",
    "reviewer_1_support_label",
    "reviewer_1_notes",
    "reviewer_2_id",
    "reviewer_2_risk_label",
    "reviewer_2_support_label",
    "reviewer_2_notes",
    "adjudicator_id",
    "review_status",
    "adjudication_notes",
    "label",
    *FEATURE_COLUMNS,
}
ALLOWED_SUPPORT_LABELS = {"entailment", "neutral", "contradiction"}
ALLOWED_REVIEWER_SUPPORT_LABELS = (
    ALLOWED_SUPPORT_LABELS | {"insufficient_evidence", "uncertain"}
)
ALLOWED_EVIDENCE_ROLES = {
    "external_source",
    "external_regulatory_evidence",
    "configured_environmental_source",
    "independent_third_party_source",
}
BOOTSTRAP_RESAMPLES = 1000
MIN_BOOTSTRAP_GROUPS_PER_LABEL = 10


def _normalized_claim(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def load_evaluation_data(csv_path: Path) -> pd.DataFrame:
    """Load reviewed rows and reject labels/features that cannot be audited."""

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {csv_path}. "
            "No labeled evaluation was run."
        )

    frame = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "Evaluation dataset is missing required columns: "
            + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("Evaluation dataset contains no labeled rows.")

    text_columns = (
        "annotation_id",
        "analysis_id",
        "claim_id",
        "company",
        "report_id",
        "claim",
        "claim_group",
        "split_group",
        "claim_source_url",
        "claim_source_role",
        "evidence_source_url",
        "evidence_source_urls",
        "evidence_source_role",
        "evidence_excerpt",
        "evidence_availability",
        "compatibility_decision",
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
        "support_label",
        "review_notes",
        "reviewer_1_id",
        "reviewer_1_risk_label",
        "reviewer_1_support_label",
        "reviewer_1_notes",
        "reviewer_2_id",
        "reviewer_2_risk_label",
        "reviewer_2_support_label",
        "reviewer_2_notes",
        "adjudicator_id",
        "review_status",
        "label",
    )
    for column in ("annotation_id", "analysis_id", "claim_id", "report_id"):
        frame[column] = frame[column].map(
            lambda value: (
                ""
                if pd.isna(value)
                else str(value).strip()
            )
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

    def is_http_url(value: str) -> bool:
        try:
            parsed = urlparse(value)
            return (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and not parsed.username
                and not parsed.password
                and (parsed.port is None or 1 <= parsed.port <= 65535)
            )
        except (ValueError, TypeError):
            return False

    url_columns = ("claim_source_url", "evidence_source_url")
    invalid_urls = pd.Series(False, index=frame.index)
    for column in url_columns:
        invalid_urls |= ~frame[column].map(is_http_url)
    if invalid_urls.any():
        rows = (frame.index[invalid_urls] + 2).tolist()
        raise ValueError(f"Source URLs must be absolute HTTP(S) URLs; invalid CSV rows: {rows}.")

    invalid_evidence_urls = frame["evidence_source_urls"].map(
        lambda value: not all(
            is_http_url(url.strip())
            for url in value.split("|")
            if url.strip()
        )
    )
    if invalid_evidence_urls.any():
        rows = (frame.index[invalid_evidence_urls] + 2).tolist()
        raise ValueError(
            "evidence_source_urls must contain absolute HTTP(S) URLs "
            f"separated by '|'; invalid CSV rows: {rows}."
        )

    frame["label"] = frame["label"].str.casefold()
    invalid_labels = sorted(set(frame["label"]) - ALLOWED_LABELS)
    if invalid_labels:
        raise ValueError(
            "Labels must be exactly 'review_concern' or "
            "'no_review_concern'; found: " + ", ".join(invalid_labels)
        )
    for column in ("reviewer_1_risk_label", "reviewer_2_risk_label"):
        frame[column] = frame[column].str.casefold()
        invalid = sorted(set(frame[column]) - ALLOWED_LABELS)
        if invalid:
            raise ValueError(
                f"{column} contains invalid labels: " + ", ".join(invalid)
            )
    for column in (
        "reviewer_1_support_label",
        "reviewer_2_support_label",
    ):
        frame[column] = frame[column].str.casefold()
        invalid = sorted(
            set(frame[column]) - ALLOWED_REVIEWER_SUPPORT_LABELS
        )
        if invalid:
            raise ValueError(
                f"{column} contains invalid labels: " + ", ".join(invalid)
            )
    frame["support_label"] = frame["support_label"].str.casefold()
    invalid_support_labels = sorted(
        set(frame["support_label"]) - ALLOWED_SUPPORT_LABELS
    )
    if invalid_support_labels:
        raise ValueError(
            "support_label must be an adjudicated entailment, neutral, or "
            "contradiction label; found: "
            + ", ".join(invalid_support_labels)
        )
    if not frame["review_status"].str.casefold().eq("adjudicated").all():
        raise ValueError(
            "Only rows with review_status='adjudicated' can be evaluated."
        )
    frame["review_status"] = frame["review_status"].str.casefold()
    if not frame["claim_source_role"].str.casefold().eq("company_source").all():
        raise ValueError(
            "Risk labels require a verified company-authored claim source."
        )
    frame["claim_source_role"] = frame["claim_source_role"].str.casefold()
    frame["evidence_source_role"] = frame["evidence_source_role"].str.casefold()
    invalid_roles = sorted(
        set(frame["evidence_source_role"]) - ALLOWED_EVIDENCE_ROLES
    )
    if invalid_roles:
        raise ValueError(
            "Risk labels require an eligible external evidence role; found: "
            + ", ".join(invalid_roles)
        )
    if "evidence_source_roles" in frame:
        invalid_role_rows = []
        for index, value in frame["evidence_source_roles"].items():
            roles = [
                role.strip().casefold()
                for role in str(value or "").split("|")
                if role.strip()
            ]
            urls = [
                url.strip()
                for url in str(frame.at[index, "evidence_source_urls"]).split("|")
                if url.strip()
            ]
            if (
                not roles
                or len(roles) != len(urls)
                or any(role not in ALLOWED_EVIDENCE_ROLES for role in roles)
            ):
                invalid_role_rows.append(index + 2)
        if invalid_role_rows:
            raise ValueError(
                "evidence_source_roles must list one eligible external role "
                "per evidence_source_urls entry, separated by '|'; invalid "
                f"CSV rows: {invalid_role_rows}."
            )
    if not frame["evidence_availability"].str.casefold().eq(
        "available_eligible"
    ).all():
        raise ValueError(
            "Risk evaluation requires available eligible evidence; "
            "missing, inaccessible, ineligible, or uncertain evidence "
            "cannot be a negative label."
        )
    frame["evidence_availability"] = (
        frame["evidence_availability"].str.casefold()
    )
    allowed_compatibility = {"compatible", "qualitative_reviewed"}
    frame["compatibility_decision"] = (
        frame["compatibility_decision"].str.casefold()
    )
    invalid_compatibility = sorted(
        set(frame["compatibility_decision"]) - allowed_compatibility
    )
    if invalid_compatibility:
        raise ValueError(
            "Risk labels require reviewed compatible evidence; found: "
            + ", ".join(invalid_compatibility)
        )
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
    allowed_dimension_values = {"compatible", "not_applicable"}
    for column in compatibility_columns:
        frame[column] = frame[column].str.casefold()
        invalid = sorted(set(frame[column]) - allowed_dimension_values)
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
    qualitative_rows = frame["compatibility_decision"].eq(
        "qualitative_reviewed"
    )
    invalid_qualitative = qualitative_rows & (
        frame[list(non_baseline_matches)].ne("not_applicable").any(axis=1)
        | frame["baseline_match"].ne("not_applicable")
    )
    quantitative_rows = ~qualitative_rows
    invalid_quantitative = quantitative_rows & (
        frame[list(non_baseline_matches)].ne("compatible").any(axis=1)
        | ~frame["baseline_match"].isin({"compatible", "not_applicable"})
    )
    if invalid_qualitative.any() or invalid_quantitative.any():
        rows = (
            frame.index[invalid_qualitative | invalid_quantitative] + 2
        ).tolist()
        raise ValueError(
            "Risk rows must be compatible on every applicable measurement "
            "dimension; qualitative_reviewed rows must mark every dimension "
            "not_applicable. CSV rows: "
            f"{rows}."
        )

    disagreement = (
        frame["reviewer_1_risk_label"] != frame["reviewer_2_risk_label"]
    ) | (
        frame["reviewer_1_support_label"]
        != frame["reviewer_2_support_label"]
    ) | (
        frame["label"] != frame["reviewer_1_risk_label"]
    ) | (
        frame["label"] != frame["reviewer_2_risk_label"]
    ) | (
        frame["support_label"] != frame["reviewer_1_support_label"]
    ) | (
        frame["support_label"] != frame["reviewer_2_support_label"]
    )
    missing_adjudication_notes = (
        frame["adjudication_notes"].map(
            lambda value: not isinstance(value, str) or not value.strip()
        )
        & disagreement
    )
    if missing_adjudication_notes.any():
        rows = (
            frame.index[missing_adjudication_notes] + 2
        ).tolist()
        raise ValueError(
            "Disagreements require adjudication_notes; CSV rows: "
            f"{rows}."
        )

    frame["_normalized_claim"] = frame["claim"].map(_normalized_claim)
    duplicate_claims = frame.groupby("_normalized_claim", dropna=False)
    if (duplicate_claims["claim_group"].nunique() > 1).any():
        raise ValueError(
            "Exact duplicate claims must share one claim_group so they "
            "cannot leak across data splits."
        )
    if (duplicate_claims["split_group"].nunique() > 1).any():
        raise ValueError(
            "Exact duplicate claims must share one split_group so they "
            "cannot leak across data splits."
        )
    if (duplicate_claims["label"].nunique() > 1).any():
        raise ValueError(
            "Exact duplicate claims have conflicting human labels."
        )

    inconsistent_split_groups = (
        frame.groupby("claim_group")["split_group"].nunique() > 1
    )
    if inconsistent_split_groups.any():
        groups = inconsistent_split_groups[
            inconsistent_split_groups
        ].index.tolist()
        raise ValueError(
            "All rows in a claim_group must have one split_group; "
            "conflicting groups: " + ", ".join(groups)
        )

    inconsistent_group_labels = (
        frame.groupby("claim_group")["label"].nunique() > 1
    )
    if inconsistent_group_labels.any():
        groups = inconsistent_group_labels[
            inconsistent_group_labels
        ].index.tolist()
        raise ValueError(
            "All rows in a claim_group must have one adjudicated label; "
            "conflicting groups: " + ", ".join(groups)
        )
    company_group_counts = frame.groupby("company")["split_group"].nunique()
    if (company_group_counts > 1).any():
        companies = company_group_counts[
            company_group_counts > 1
        ].index.tolist()
        raise ValueError(
            "All reports for a company must share one split_group to keep "
            "companies out of both training and test data: "
            + ", ".join(companies)
        )
    if frame["annotation_id"].duplicated().any():
        duplicates = frame.loc[
            frame["annotation_id"].duplicated(keep=False),
            "annotation_id",
        ].tolist()
        raise ValueError(
            "annotation_id values must be unique: "
            + ", ".join(duplicates)
        )

    for column in FEATURE_COLUMNS:
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & converted.isna()
        if invalid.any():
            rows = (frame.index[invalid] + 2).tolist()
            raise ValueError(
                f"Non-numeric values in feature {column!r} at CSV rows: "
                f"{rows}."
            )
        non_finite = converted.dropna().map(
            lambda value: not math.isfinite(float(value))
        )
        if non_finite.any():
            raise ValueError(f"Feature {column!r} contains non-finite values.")
        frame[column] = converted

    if frame["similarity"].isna().any() or not frame["similarity"].between(
        0, 1
    ).all():
        raise ValueError("similarity must be present and between 0 and 1.")
    if not frame["evidence_available"].isin([0, 1]).all():
        raise ValueError("evidence_available must contain only 0 or 1.")
    if not frame["evidence_available"].eq(1).all():
        raise ValueError(
            "Risk-score evaluation requires an eligible evidence match "
            "for every row; missing evidence is not a negative label."
        )
    for column in ("nli_entailment", "nli_contradiction", "nli_neutral"):
        values = frame[column].dropna()
        if (values < 0).any() or ((values % 1) != 0).any():
            raise ValueError(f"{column} must contain non-negative counts.")

    return frame.reset_index(drop=True)


def _stratified_group_splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Make deterministic approximate 60/20/20 splits by source group."""

    groups = (
        frame[["split_group", "label"]]
        .drop_duplicates()
        .sort_values("split_group")
        .reset_index(drop=True)
    )
    class_counts = (
        frame.groupby("label")["split_group"].nunique()
    )
    insufficient = {
        label: int(class_counts.get(label, 0))
        for label in sorted(ALLOWED_LABELS)
        if class_counts.get(label, 0) < 5
    }
    if insufficient:
        raise ValueError(
            "At least five distinct split_group values per label are "
            "required for a stratified 60/20/20 split; found "
            + ", ".join(
                f"{label}={count}"
                for label, count in insufficient.items()
            )
            + "."
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
            frame["label"],
            groups=frame["split_group"],
        )
    ]
    test_groups = folds[0]
    validation_groups = folds[1]
    train_groups = set(frame["split_group"]) - test_groups - validation_groups

    split_groups = {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups,
    }
    for name, split_group_ids in split_groups.items():
        labels = set(
            frame.loc[
                frame["split_group"].isin(split_group_ids),
                "label",
            ]
        )
        if labels != ALLOWED_LABELS:
            raise ValueError(
                "The grouped split does not contain both labels in "
                f"{name}; collect more independent split groups per label."
            )

    return {
        "train": frame[frame["split_group"].isin(train_groups)].copy(),
        "validation": frame[
            frame["split_group"].isin(validation_groups)
        ].copy(),
        "test": frame[frame["split_group"].isin(test_groups)].copy(),
    }


def _metrics(actual: list[int], predicted: list[int]) -> dict:
    true_negative = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    false_positive = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    false_negative = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    true_positive = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))

    per_class = {}
    for label, name in ((0, NEGATIVE_LABEL), (1, POSITIVE_LABEL)):
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "support": sum(value == label for value in actual),
        }

    positive = per_class[POSITIVE_LABEL]
    return {
        "accuracy": (
            (true_positive + true_negative) / len(actual) if actual else 0.0
        ),
        "precision": positive["precision"],
        "recall": positive["recall"],
        "f1": positive["f1"],
        "macro_f1": sum(item["f1"] for item in per_class.values()) / 2,
        "confusion_matrix": {
            "labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
            "rows_actual_columns_predicted": [
                [true_negative, false_positive],
                [false_negative, true_positive],
            ],
        },
        "per_class": per_class,
    }


def _risk_scores(frame: pd.DataFrame) -> list[float]:
    features = frame[FEATURE_COLUMNS].copy()
    features.insert(0, "claim_id", range(1, len(features) + 1))
    predictions = predict_risk_from_features(features)
    return predictions["risk_score"].astype(float).tolist()


def _cluster_bootstrap_intervals(
    test: pd.DataFrame,
    predictions: list[int],
) -> dict:
    """Estimate cluster-bootstrap intervals only with enough independent groups."""

    group_labels = test.groupby("split_group")["label"].agg(
        lambda values: set(values)
    )
    groups_per_label = {
        label: int(
            sum(label in labels for labels in group_labels)
        )
        for label in sorted(ALLOWED_LABELS)
    }
    if min(groups_per_label.values()) < MIN_BOOTSTRAP_GROUPS_PER_LABEL:
        return {
            "status": "not_reported",
            "reason": (
                "Cluster-bootstrap intervals require at least "
                f"{MIN_BOOTSTRAP_GROUPS_PER_LABEL} independent test groups "
                "containing each label."
            ),
            "test_groups_per_label": groups_per_label,
        }

    prediction_by_index = dict(zip(test.index, predictions))
    indices_by_group = {
        group: list(indices)
        for group, indices in test.groupby("split_group").groups.items()
    }
    group_names = list(indices_by_group)
    generator = random.Random(SPLIT_SEED)
    samples = {"accuracy": [], "macro_f1": [], "f1": []}
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_groups = [
            generator.choice(group_names)
            for _ in group_names
        ]
        sampled_indices = [
            index
            for group in sampled_groups
            for index in indices_by_group[group]
        ]
        actual = [
            int(test.at[index, "label"] == POSITIVE_LABEL)
            for index in sampled_indices
        ]
        predicted = [
            prediction_by_index[index]
            for index in sampled_indices
        ]
        metrics = _metrics(actual, predicted)
        for metric in samples:
            samples[metric].append(metrics[metric])

    intervals = {}
    for metric, values in samples.items():
        values.sort()
        intervals[metric] = {
            "lower_95": values[int(0.025 * (len(values) - 1))],
            "upper_95": values[int(0.975 * (len(values) - 1))],
        }
    return {
        "status": "reported",
        "method": "95% percentile cluster bootstrap by split_group",
        "resamples": BOOTSTRAP_RESAMPLES,
        "test_groups_per_label": groups_per_label,
        "metrics": intervals,
    }


def evaluate_dataset(frame: pd.DataFrame) -> dict:
    """Tune only on validation rows and report once on untouched test groups."""

    splits = _stratified_group_splits(frame)
    y_by_label = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}
    train_labels = [y_by_label[label] for label in splits["train"]["label"]]
    majority_label = int(sum(train_labels) > len(train_labels) / 2)
    validation_labels = [
        y_by_label[label] for label in splits["validation"]["label"]
    ]
    validation_scores = _risk_scores(splits["validation"])

    validation_candidates = []
    for threshold in THRESHOLDS:
        predicted = [int(score >= threshold) for score in validation_scores]
        candidate = _metrics(validation_labels, predicted)
        validation_candidates.append(
            {"threshold": threshold, "metrics": candidate}
        )
    selected = max(
        validation_candidates,
        key=lambda item: (
            item["metrics"]["f1"],
            item["metrics"]["precision"],
            item["threshold"],
        ),
    )

    test = splits["test"]
    test_labels = [y_by_label[label] for label in test["label"]]
    test_scores = _risk_scores(test)
    test_predictions = [
        int(score >= selected["threshold"]) for score in test_scores
    ]
    baseline_predictions = [majority_label] * len(test_labels)
    test_metrics = _metrics(test_labels, test_predictions)

    split_summary = {}
    for name, split in splits.items():
        split_summary[name] = {
            "rows": len(split),
            "claim_groups": int(split["claim_group"].nunique()),
            "split_groups": int(split["split_group"].nunique()),
            "label_counts": {
                label: int(count)
                for label, count in split["label"].value_counts().items()
            },
        }

    return {
        "evaluation_type": (
            "fixed heuristic score with validation-selected cutoff; "
            "no learned model was fit"
        ),
        "prediction_target": (
            "Adjudicated human recommendation for further review against "
            "the supplied evidence; not a ground-truth greenwashing label."
        ),
        "positive_label": POSITIVE_LABEL,
        "split_seed": SPLIT_SEED,
        "reviewer_agreement": {
            column: {
                "exact_agreement": float(
                    (frame[first] == frame[second]).mean()
                ),
                "disagreement_rows": int(
                    (frame[first] != frame[second]).sum()
                ),
                "rows": len(frame),
            }
            for column, first, second in (
                (
                    "risk_label",
                    "reviewer_1_risk_label",
                    "reviewer_2_risk_label",
                ),
                (
                    "support_label",
                    "reviewer_1_support_label",
                    "reviewer_2_support_label",
                ),
            )
        },
        "split_summary": split_summary,
        "validation_threshold": selected["threshold"],
        "validation_metrics": selected["metrics"],
        "test_metrics": test_metrics,
        "test_confidence_intervals": _cluster_bootstrap_intervals(
            test,
            test_predictions,
        ),
        "majority_class_baseline": {
            "predicted_label": (
                POSITIVE_LABEL if majority_label else NEGATIVE_LABEL
            ),
            "test_metrics": _metrics(test_labels, baseline_predictions),
        },
        "limitations": [
            "Results are valid only for independently reviewed, "
            "provenance-backed labels and features.",
            "Threshold selection uses validation groups only; the test "
            "split is used once for final reporting.",
            "This evaluates the deterministic heuristic, not scientific "
            "greenwashing accuracy.",
            "NLI requires a separate human-labeled claim-evidence pair "
            "benchmark and is not evaluated here.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the fixed experimental risk heuristic using "
            "human-reviewed, provenance-backed features."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"CSV dataset path (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args(argv)

    try:
        frame = load_evaluation_data(args.dataset)
        report = evaluate_dataset(frame)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
