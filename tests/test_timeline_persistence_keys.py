"""Regression: internal timeline keys must never reach persisted records.

Persistence hashes each whole record into record_id. An internal key (for example
_value_position) changes that hash and would make IDs depend on implementation
detail. SYNTHETIC inputs; uses a temporary SQLite file, never the user's database."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import database
import timeline_analyzer as ta

DOCS = [{
    "status": "success", "title": "Synthetic report", "publisher": "Synthetic Co",
    "publication_date": "2025-01-01", "source_url": "https://company.example/r",
    "text": (
        "Scope 1 emissions reached 500 t in 2024 and are projected to reach 400 t in 2026. "
        "Emissions fell 10% in 2024. We aim to cut emissions to 300 t by 2030."
    ),
}]


class TimelineKeyTests(unittest.TestCase):
    def test_no_internal_keys_in_timeline_output(self):
        for record in ta.analyze_timeline(DOCS):
            internal = [key for key in record if key.startswith("_")]
            self.assertEqual(internal, [], f"internal keys leaked: {internal}")

    def test_persisted_timeline_rows_store_no_internal_keys(self):
        records = ta.analyze_timeline(DOCS)
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "synthetic.sqlite3"
            database.initialize_database(database_path)
            database.save_timeline_records("Synthetic Co", "syn-analysis", records, database_path=database_path)
            with closing(sqlite3.connect(database_path)) as connection:
                rows = connection.execute("SELECT COUNT(*) FROM environmental_timeline").fetchone()[0]
        self.assertEqual(rows, len(records))


if __name__ == "__main__":
    unittest.main()
