from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtester.io.serialization import to_jsonable, write_json


def test_to_jsonable_handles_numpy_and_timestamps():
    payload = {
        "n": np.int64(3),
        "f": np.float64(1.5),
        "ts": pd.Timestamp("2024-01-02"),
        "arr": np.array([1, 2]),
        "nested": {"k": np.float32(0.5)},
    }
    out = to_jsonable(payload)
    json.dumps(out)  # must not raise


def test_write_json_roundtrip(tmp_path: Path):
    p = tmp_path / "out.json"
    write_json(p, {"a": 1, "ts": pd.Timestamp("2024-01-02")})
    data = json.loads(p.read_text())
    assert data["a"] == 1
    assert data["ts"].startswith("2024-01-02")
