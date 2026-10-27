#!/usr/bin/env python3
"""Import completed blind review packets with fixed, saved model outputs."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import load_evaluation_data
from evaluation.evaluate_nli import load_nli_annotations
from evaluation.prepare_review_packet import pair_identifier
from risk_model import FEATURE_COLUMNS

MODEL_OUTPUT_COLUMNS = {
    "model_name",
    "model_prediction",
    "model_confidence",
}


def _read_inputs(packet_path: Path, analysis_path: Path):
    packet = pd.read_csv(packet_path, dtype=str, keep_default_na=False)
    with analysis_path.open("r", encoding="utf-8") as file:
        analysis = json.load(file)
    if not isinstance(analysis, dict):
        raise ValueError("Saved analysis JSON must contain an object.")
    analysis_id = str(analysis.get("analysis_id") or "")
    if not analysis_id:
        raise ValueError("Saved analysis has no analysis_id.")
    if not packet.empty and set(packet["analysis_id"]) != {analysis_id}:
        raise ValueError(
            "The review packet analysis_id does not match the supplied analysis."
        )
    return packet, analysis


def _write_exclusive_validated(
    frame: pd.DataFrame,
    output_path: Path,
    validation,
) -> None:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing labels: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="esg-label-validation-",
        dir=output_path.parent,
    ) as directory:
        temporary_csv = Path(directory) / "candidate.csv"
        frame.to_csv(temporary_csv, index=False)
        validation(temporary_csv)
    with output_path.open("x", encoding="utf-8-sig", newline="") as file:
        frame.to_csv(file, index=False)


def import_risk_reviews(
    packet_path: Path,
    analysis_path: Path,
    output_path: Path,
) -> dict:
    packet, analysis = _read_inputs(packet_path, analysis_path)
    prohibited = (set(FEATURE_COLUMNS) | MODEL_OUTPUT_COLUMNS) & set(packet.columns)
    if prohibited:
        raise ValueError(
            "Blind risk-review input must not contain model features or "
            "outputs: " + ", ".join(sorted(prohibited))
        )
    if packet.empty:
        raise ValueError("Risk review packet contains no rows.")

    features = {}
    for item in analysis.get("features") or []:
        if not isinstance(item, dict) or item.get("claim_id") is None:
            continue
        claim_id = str(item["claim_id"])
        if claim_id in features:
            raise ValueError(f"Analysis has duplicate feature claim_id {claim_id}.")
        features[claim_id] = item

    rows = []
    for row_number, source_row in packet.iterrows():
        claim_id = source_row.get("claim_id", "")
        feature = features.get(str(claim_id))
        if feature is None:
            raise ValueError(
                f"No saved feature record for claim_id {claim_id!r} "
                f"(CSV row {row_number + 2})."
            )
        if str(feature.get("claim") or "").strip() != str(
            source_row.get("claim") or ""
        ).strip():
            raise ValueError(
                f"Claim text mismatch for claim_id {claim_id!r} "
                f"(CSV row {row_number + 2})."
            )
        row = source_row.to_dict()
        for column in FEATURE_COLUMNS:
            row[column] = feature.get(column)
        rows.append(row)
    frame = pd.DataFrame(rows)
    _write_exclusive_validated(
        frame,
        output_path,
        load_evaluation_data,
    )
    return {
        "task": "risk",
        "rows_imported": len(frame),
        "output": str(output_path),
        "validated": True,
        "model_outputs_exposed_during_review": False,
    }


def _nli_predictions(analysis: dict) -> dict[str, dict]:
    from contradiction_detector import MODEL_NAME

    analysis_id = str(analysis["analysis_id"])
    predictions = {}
    for item in analysis.get("nli_results") or []:
        if not isinstance(item, dict):
            continue
        claim = item.get("claim")
        evidence = item.get("evidence")
        evidence_url = item.get("source_url")
        if not all(isinstance(value, str) and value.strip() for value in (
            claim,
            evidence,
            evidence_url,
        )):
            continue
        pair_id = pair_identifier(
            analysis_id,
            claim,
            evidence,
            evidence_url,
        )
        prediction = str(
            item.get("raw_nli_label")
            or item.get("nli_result")
            or "unavailable"
        ).casefold().strip()
        if prediction not in {
            "entailment",
            "neutral",
            "contradiction",
            "unavailable",
        }:
            raise ValueError(
                f"Unsupported saved NLI prediction {prediction!r}."
            )
        result = {
            "model_name": str(item.get("model_name") or MODEL_NAME),
            "model_prediction": prediction,
            "model_confidence": item.get("confidence"),
        }
        previous = predictions.get(pair_id)
        if previous is not None and previous != result:
            raise ValueError(
                f"Conflicting saved NLI outputs for pair_id {pair_id}."
            )
        predictions[pair_id] = result
    return predictions


def import_nli_reviews(
    packet_path: Path,
    analysis_path: Path,
    output_path: Path,
) -> dict:
    packet, analysis = _read_inputs(packet_path, analysis_path)
    prohibited = MODEL_OUTPUT_COLUMNS & set(packet.columns)
    if prohibited:
        raise ValueError(
            "Blind NLI-review input must not contain model outputs: "
            + ", ".join(sorted(prohibited))
        )
    if packet.empty:
        raise ValueError("NLI review packet contains no rows.")

    predictions = _nli_predictions(analysis)
    from contradiction_detector import MODEL_NAME

    frame = packet.copy()
    frame["model_name"] = [
        predictions.get(pair_id, {}).get("model_name", MODEL_NAME)
        for pair_id in frame["pair_id"]
    ]
    frame["model_prediction"] = [
        predictions.get(pair_id, {}).get("model_prediction", "unavailable")
        for pair_id in frame["pair_id"]
    ]
    frame["model_confidence"] = [
        predictions.get(pair_id, {}).get("model_confidence")
        for pair_id in frame["pair_id"]
    ]
    unavailable = int(frame["model_prediction"].eq("unavailable").sum())
    _write_exclusive_validated(
        frame,
        output_path,
        load_nli_annotations,
    )
    return {
        "task": "nli",
        "rows_imported": len(frame),
        "model_predictions_available": len(frame) - unavailable,
        "model_predictions_unavailable": unavailable,
        "output": str(output_path),
        "validated": True,
        "model_outputs_exposed_during_review": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Import completed human-review annotations and merge saved model "
            "features only after blind review."
        )
    )
    parser.add_argument("--task", choices=("risk", "nli"), required=True)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.task == "risk":
            report = import_risk_reviews(
                args.packet,
                args.analysis,
                args.output,
            )
        else:
            report = import_nli_reviews(
                args.packet,
                args.analysis,
                args.output,
            )
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
