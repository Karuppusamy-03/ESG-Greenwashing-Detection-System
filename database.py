# database.py

from contextlib import closing
import hashlib
import json
import os
import sqlite3

import pandas as pd


def initialize_database(database_path=None):
    """Create the SQLite database and risk predictions table."""

    if database_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        database_path = os.path.join(
            base_dir,
            "data",
            "greenwashing_risk.db",
        )

    database_path = os.fspath(database_path)
    os.makedirs(os.path.dirname(os.path.abspath(database_path)), exist_ok=True)

    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS risk_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER,
                similarity REAL,
                evidence_available INTEGER,
                nli_entailment REAL,
                nli_contradiction REAL,
                nli_neutral REAL,
                percentage_difference REAL,
                historical_change REAL,
                risk_score REAL,
                predicted_risk_class INTEGER,
                demonstration_risk_probability REAL,
                demonstration_risk_category TEXT,
                company_name TEXT,
                analysis_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(risk_predictions)"
            )
        }

        for column in ("company_name", "analysis_id", "risk_score"):
            if column not in columns:
                column_type = "REAL" if column == "risk_score" else "TEXT"
                connection.execute(
                    f"ALTER TABLE risk_predictions ADD COLUMN {column} {column_type}"
                )

        connection.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_risk_predictions_company_analysis_claim
            ON risk_predictions (company_name, analysis_id, claim_id)
            WHERE company_name IS NOT NULL AND analysis_id IS NOT NULL
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS environmental_timeline (
                record_id TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                analysis_id TEXT NOT NULL,
                reporting_year INTEGER,
                timeline_year INTEGER NOT NULL,
                year_basis TEXT NOT NULL,
                target_year INTEGER,
                metric TEXT NOT NULL,
                metric_value REAL,
                unit TEXT,
                boundary TEXT,
                performance_type TEXT,
                statement TEXT NOT NULL,
                previous_reported_performance TEXT,
                change_over_time TEXT,
                evidence_status TEXT,
                source_name TEXT,
                source_url TEXT,
                publication_date TEXT,
                collected_at TEXT,
                supporting_evidence TEXT,
                supporting_evidence_source TEXT,
                supporting_evidence_url TEXT,
                supporting_evidence_status TEXT,
                supporting_evidence_nli TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        timeline_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(environmental_timeline)"
            )
        }
        if "target_year" not in timeline_columns:
            connection.execute(
                "ALTER TABLE environmental_timeline ADD COLUMN target_year INTEGER"
            )
        if "timeline_year" not in timeline_columns:
            connection.execute(
                "ALTER TABLE environmental_timeline ADD COLUMN timeline_year INTEGER"
            )
        if "year_basis" not in timeline_columns:
            connection.execute(
                "ALTER TABLE environmental_timeline ADD COLUMN year_basis TEXT"
            )
        for column in (
            "supporting_evidence",
            "supporting_evidence_source",
            "supporting_evidence_url",
            "supporting_evidence_status",
            "supporting_evidence_nli",
        ):
            if column not in timeline_columns:
                connection.execute(
                    f"ALTER TABLE environmental_timeline ADD COLUMN {column} TEXT"
                )

    print("Database initialized successfully.")
    print("Database location:", database_path)


def save_timeline_records(
    company_name,
    analysis_id,
    records,
    database_path=None,
):
    """Persist source-linked timeline records without duplicating a run."""

    if not isinstance(company_name, str) or not company_name.strip():
        raise ValueError("company_name must be a non-empty string.")
    if not isinstance(analysis_id, str) or not analysis_id.strip():
        raise ValueError("analysis_id must be a non-empty string.")
    if not isinstance(records, list):
        raise TypeError("records must be a list.")

    if database_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        database_path = os.path.join(
            base_dir,
            "data",
            "greenwashing_risk.db",
        )

    database_path = os.fspath(database_path)
    initialize_database(database_path)
    persisted_records = []

    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each timeline record must be a dictionary.")
        required_fields = (
            "reporting_year",
            "timeline_year",
            "year_basis",
            "metric",
            "statement",
        )
        missing = [
            field for field in required_fields if field not in record
        ]
        if missing:
            raise ValueError(
                f"Timeline record is missing fields: {', '.join(missing)}"
            )

        identity = {
            "company_name": company_name,
            "analysis_id": analysis_id,
            "record": record,
        }
        record_id = hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        persisted_records.append((
            record_id,
            company_name,
            analysis_id,
            (
                int(record["reporting_year"])
                if record.get("reporting_year") is not None
                else None
            ),
            int(record["timeline_year"]),
            str(record["year_basis"]),
            record.get("target_year"),
            str(record["metric"]),
            record.get("value"),
            record.get("unit"),
            record.get("boundary"),
            record.get("performance_type"),
            str(record["statement"]),
            record.get("previous_reported_performance"),
            record.get("change_over_time"),
            record.get("evidence_status"),
            record.get("source_name"),
            record.get("source_url"),
            record.get("publication_date"),
            record.get("collected_at"),
            record.get("supporting_evidence"),
            record.get("supporting_evidence_source"),
            record.get("supporting_evidence_url"),
            record.get("supporting_evidence_status"),
            record.get("supporting_evidence_nli"),
        ))

    if not persisted_records:
        return 0

    with closing(sqlite3.connect(database_path)) as connection, connection:
        before = connection.total_changes
        connection.executemany("""
            INSERT OR IGNORE INTO environmental_timeline (
                record_id,
                company_name,
                analysis_id,
                reporting_year,
                timeline_year,
                year_basis,
                target_year,
                metric,
                metric_value,
                unit,
                boundary,
                performance_type,
                statement,
                previous_reported_performance,
                change_over_time,
                evidence_status,
                source_name,
                source_url,
                publication_date,
                collected_at,
                supporting_evidence,
                supporting_evidence_source,
                supporting_evidence_url,
                supporting_evidence_status,
                supporting_evidence_nli
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, persisted_records)
        return connection.total_changes - before


def save_company_predictions(
    company_name,
    analysis_id,
    predictions,
    database_path=None,
):
    """Save eligible company-scoped risk predictions without duplicating a run."""

    if predictions.empty:
        return 0

    required_columns = [
        "claim_id",
        "similarity",
        "evidence_available",
        "nli_entailment",
        "nli_contradiction",
        "nli_neutral",
        "percentage_difference",
        "historical_change",
        "risk_score",
        "predicted_risk_class",
        "demonstration_risk_probability",
        "demonstration_risk_category",
    ]
    missing = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing:
        raise ValueError(
            f"Missing prediction columns: {', '.join(missing)}"
        )

    eligible = (
        predictions["evidence_available"].eq(1)
        & predictions["risk_score"].notna()
        & predictions["predicted_risk_class"].notna()
        & predictions["demonstration_risk_category"].isin(
            {"LOW", "MEDIUM", "HIGH"}
        )
    )
    if "risk_score_status" in predictions.columns:
        eligible &= predictions["risk_score_status"].eq("HEURISTIC")
    predictions = predictions.loc[eligible]
    if predictions.empty:
        return 0

    if database_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        database_path = os.path.join(
            base_dir,
            "data",
            "greenwashing_risk.db",
        )

    database_path = os.fspath(database_path)
    initialize_database(database_path)

    rows = predictions[required_columns].astype(object)
    rows = rows.where(pd.notna(rows), None)
    records = []

    for values in rows.itertuples(index=False, name=None):
        native_values = tuple(
            value.item() if hasattr(value, "item") else value
            for value in values
        )
        records.append((
            *native_values,
            company_name,
            analysis_id,
        ))

    with closing(sqlite3.connect(database_path)) as connection, connection:
        before = connection.total_changes
        connection.executemany("""
            INSERT OR IGNORE INTO risk_predictions (
                claim_id,
                similarity,
                evidence_available,
                nli_entailment,
                nli_contradiction,
                nli_neutral,
                percentage_difference,
                historical_change,
                risk_score,
                predicted_risk_class,
                demonstration_risk_probability,
                demonstration_risk_category,
                company_name,
                analysis_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)

        return connection.total_changes - before


def save_predictions():
    """Save prediction CSV records to the SQLite database."""

    base_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(
        base_dir, "output", "esg_risk_predictions.csv"
    )

    database_path = os.path.join(
        base_dir, "data", "greenwashing_risk.db"
    )

    if not os.path.exists(csv_path):
        print("ERROR: Prediction CSV not found.")
        print("Run risk_predictor.py first.")
        return

    df = pd.read_csv(csv_path)

    required_columns = [
        "claim_id",
        "similarity",
        "evidence_available",
        "nli_entailment",
        "nli_contradiction",
        "nli_neutral",
        "percentage_difference",
        "historical_change",
        "risk_score",
        "predicted_risk_class",
        "demonstration_risk_probability",
        "demonstration_risk_category",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print("ERROR: Missing CSV columns:", missing)
        return

    # Convert missing numeric values to SQL NULL.
    df = df[required_columns].astype(object)
    df = df.where(pd.notna(df), None)

    records = [
        tuple(row)
        for row in df.itertuples(index=False, name=None)
    ]

    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.executemany("""
            INSERT INTO risk_predictions (
                claim_id,
                similarity,
                evidence_available,
                nli_entailment,
                nli_contradiction,
                nli_neutral,
                percentage_difference,
                historical_change,
                risk_score,
                predicted_risk_class,
                demonstration_risk_probability,
                demonstration_risk_category
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)

    print(f"Saved {len(records)} prediction records.")
    print("Database location:", database_path)


def show_predictions():
    """Display records stored in SQLite."""

    base_dir = os.path.dirname(os.path.abspath(__file__))

    database_path = os.path.join(
        base_dir, "data", "greenwashing_risk.db"
    )

    with closing(sqlite3.connect(database_path)) as connection, connection:
        df = pd.read_sql_query(
            "SELECT * FROM risk_predictions ORDER BY id",
            connection,
        )

    print("\nSTORED ESG RISK PREDICTIONS")
    print("=" * 65)

    if df.empty:
        print("No predictions have been saved yet.")
    else:
        print(df.to_string(index=False))

    print("\nTotal database records:", len(df))


if __name__ == "__main__":
    initialize_database()
    save_predictions()
    show_predictions()