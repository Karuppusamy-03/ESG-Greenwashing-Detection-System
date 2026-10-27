"""Regression tests for saved-analysis JSON persistence.

All files are written to temporary directories. These tests never touch the
project's real database or saved analyses. The payloads are synthetic.
"""

import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

from analysis_serialization import (
    UnsupportedJSONValueError,
    dumps_analysis_json,
    to_json_safe,
    write_json_atomically,
)


SYNTHETIC_OUTPUT = PureWindowsPath(
    r"C:\Synthetic\ESG-Greenwashing\output\collection_fixture.json"
)


def synthetic_analysis_payload():
    """A fictional analysis payload that mirrors the real structure."""
    return {
        "company": "synthetic-co",
        "analysis_id": "a" * 64,
        "collection": {
            "output_file": SYNTHETIC_OUTPUT,
            "supplemental_output_file": None,
            "documents": [
                {
                    "source_url": "https://example.invalid/report",
                    "status": "ok",
                    "publication_date": "2024-01-01",
                }
            ],
            "diagnostics": {
                "claim_specific_search": {
                    "output_file": SYNTHETIC_OUTPUT,
                    "documents_collected": 1,
                }
            },
        },
        "company_claims": [{"text": "Fictional claim.", "year": 2024}],
        "features": [{"similarity": np.float64(0.42), "evidence_available": np.int64(1)}],
        "matching_results": [],
        "nli_results": [],
    }


class WindowsPathSerializationTests(unittest.TestCase):
    def test_original_windows_path_error_is_fixed(self):
        """Reproduce the reported defect, then confirm the fix resolves it."""
        payload = {"collection": {"output_file": SYNTHETIC_OUTPUT}}

        # Pre-fix behaviour: the standard encoder rejects the path.
        with self.assertRaises(TypeError) as context:
            json.dumps(payload)
        self.assertIn("not JSON serializable", str(context.exception))

        # Post-fix behaviour: the same payload is saved and reloads cleanly.
        text = dumps_analysis_json(payload)
        reloaded = json.loads(text)
        self.assertEqual(reloaded["collection"]["output_file"], str(SYNTHETIC_OUTPUT))

    def test_nested_paths_inside_dicts_lists_and_results_are_converted(self):
        payload = {
            "outer": [{"inner": {"deep": [SYNTHETIC_OUTPUT, (SYNTHETIC_OUTPUT,)]}}]
        }
        converted = to_json_safe(payload)
        expected = str(SYNTHETIC_OUTPUT)
        self.assertEqual(converted["outer"][0]["inner"]["deep"], [expected, [expected]])

    def test_normal_analysis_payload_round_trips_with_important_fields(self):
        payload = synthetic_analysis_payload()
        text = dumps_analysis_json(payload)
        reloaded = json.loads(text)

        self.assertEqual(reloaded["company"], "synthetic-co")
        self.assertEqual(reloaded["analysis_id"], "a" * 64)
        self.assertEqual(
            reloaded["collection"]["documents"][0]["source_url"],
            "https://example.invalid/report",
        )
        self.assertEqual(
            reloaded["collection"]["output_file"], str(SYNTHETIC_OUTPUT)
        )
        self.assertAlmostEqual(reloaded["features"][0]["similarity"], 0.42)
        self.assertEqual(reloaded["features"][0]["evidence_available"], 1)
        self.assertIsNone(reloaded["collection"]["supplemental_output_file"])


class UnsupportedValueTests(unittest.TestCase):
    def test_unknown_object_raises_clearly_instead_of_silent_stringify(self):
        class Opaque:
            pass

        with self.assertRaises(UnsupportedJSONValueError) as context:
            to_json_safe({"result": {"data": Opaque()}})
        self.assertIn("$.result.data", str(context.exception))
        self.assertIn("Opaque", str(context.exception))

    def test_non_finite_float_is_rejected(self):
        with self.assertRaises(UnsupportedJSONValueError):
            to_json_safe({"score": math.nan})
        with self.assertRaises(UnsupportedJSONValueError):
            to_json_safe({"score": math.inf})

    def test_non_string_dict_key_is_rejected(self):
        with self.assertRaises(UnsupportedJSONValueError):
            to_json_safe({1: "value"})

    def test_pandas_missing_values_become_null(self):
        converted = to_json_safe(
            {"when": pd.NaT, "amount": pd.NA, "stamp": pd.Timestamp("2024-05-01")}
        )
        self.assertIsNone(converted["when"])
        self.assertIsNone(converted["amount"])
        self.assertEqual(converted["stamp"], "2024-05-01T00:00:00")


class AtomicWriteTests(unittest.TestCase):
    def test_write_creates_missing_parent_directories_and_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "company_analyses" / "synthetic.json"
            write_json_atomically(target, synthetic_analysis_payload())

            self.assertTrue(target.exists())
            reloaded = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(reloaded["company"], "synthetic-co")
            self.assertEqual(
                reloaded["collection"]["output_file"], str(SYNTHETIC_OUTPUT)
            )

    def test_failed_write_does_not_corrupt_existing_valid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "synthetic.json"
            write_json_atomically(target, {"analysis_id": "original"})
            original_bytes = target.read_bytes()

            class Opaque:
                pass

            with self.assertRaises(UnsupportedJSONValueError):
                write_json_atomically(target, {"analysis_id": "bad", "x": Opaque()})

            self.assertEqual(target.read_bytes(), original_bytes)
            self.assertEqual(json.loads(target.read_text())["analysis_id"], "original")
            leftovers = [p for p in Path(directory).iterdir() if p.suffix == ".tmp"]
            self.assertEqual(leftovers, [])

    def test_failed_write_leaves_no_partial_file_when_target_is_new(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "never_created.json"

            class Opaque:
                pass

            with self.assertRaises(UnsupportedJSONValueError):
                write_json_atomically(target, {"x": Opaque()})

            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])


class PersistenceStatusTests(unittest.TestCase):
    """SQLite and JSON outcomes must be reportable independently."""

    def test_sqlite_success_with_json_failure_is_distinguishable(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "synthetic.db"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE analyses (analysis_id TEXT PRIMARY KEY)")
            connection.execute("INSERT INTO analyses VALUES ('a' * 64)")
            connection.commit()
            connection.close()

            class Opaque:
                pass

            json_outcome = None
            try:
                write_json_atomically(
                    Path(directory) / "analysis.json", {"x": Opaque()}
                )
            except UnsupportedJSONValueError as error:
                json_outcome = f"{type(error).__name__}: {error}"

            check = sqlite3.connect(database)
            try:
                rows = check.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            finally:
                check.close()

            self.assertEqual(rows, 1)
            self.assertIsNotNone(json_outcome)
            self.assertFalse((Path(directory) / "analysis.json").exists())


if __name__ == "__main__":
    unittest.main()
