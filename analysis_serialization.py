"""JSON serialization boundary for saved company analyses.

The analysis payload contains values that the standard json module cannot
encode directly, most notably pathlib paths (for example the WindowsPath
stored in collection["output_file"]). This module converts those values at a
single, explicit boundary and fails loudly on anything it does not recognise,
so that data is never silently discarded or stringified by accident.
"""

import json
import math
import os
from pathlib import Path, PurePath

import numpy as np
import pandas as pd


class UnsupportedJSONValueError(TypeError):
    """Raised when a payload contains a value that cannot be saved safely."""


def to_json_safe(value, _path="$"):
    """Return a copy of ``value`` made only of JSON-encodable types.

    Supported conversions:
    - ``pathlib`` paths (``PurePath``, including ``WindowsPath``) -> ``str``
    - NumPy scalars -> Python ``int``/``float``/``bool``
    - pandas ``Timestamp``/``Timedelta`` -> ISO-8601 ``str``
    - ``None`` and pandas missing values (``NaT``, ``NA``) -> ``None``
    - ``dict`` keys must be strings; tuples and sets of supported values are
      converted to lists

    Anything else raises ``UnsupportedJSONValueError`` with the location of the
    offending value, rather than being silently converted.
    """
    if value is None or isinstance(value, (bool, str)):
        return value

    if isinstance(value, PurePath):
        return str(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnsupportedJSONValueError(
                f"Non-finite number at {_path} cannot be saved as JSON: {value!r}"
            )
        return value

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        if pd.isna(value):
            return None
        return value.isoformat()

    if value is pd.NaT or value is pd.NA:
        return None

    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise UnsupportedJSONValueError(
                    f"Non-string dictionary key at {_path}: {key!r}"
                )
            converted[key] = to_json_safe(item, f"{_path}.{key}")
        return converted

    if isinstance(value, (list, tuple)):
        return [
            to_json_safe(item, f"{_path}[{index}]")
            for index, item in enumerate(value)
        ]

    if hasattr(value, "item") and np.ndim(value) == 0:
        # Other NumPy-like 0-dimensional scalars.
        return to_json_safe(value.item(), _path)

    raise UnsupportedJSONValueError(
        f"Unsupported value of type {type(value).__name__} at {_path}; "
        "refusing to save it rather than silently converting it."
    )


def dumps_analysis_json(payload, **kwargs):
    """Serialize ``payload`` to a JSON string after making it JSON-safe."""
    options = {"ensure_ascii": False, "indent": 2, "allow_nan": False}
    options.update(kwargs)
    return json.dumps(to_json_safe(payload), **options)


def write_json_atomically(path, payload, **kwargs):
    """Write ``payload`` to ``path`` without ever leaving a partial file.

    The data is first written to a temporary file in the same directory,
    flushed to disk, and only then moved over ``path`` with ``os.replace``.
    If anything fails before the replace, the existing file at ``path`` (if
    any) is left untouched and the temporary file is removed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = dumps_analysis_json(payload, **kwargs)

    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return path
