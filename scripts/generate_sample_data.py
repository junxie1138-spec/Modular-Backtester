"""Write deterministic synthetic OHLCV CSVs for SPY and AAPL.

Run from the repo root:
    python scripts/generate_sample_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures.synthetic import make_ohlcv


def main() -> None:
    out = REPO_ROOT / "data" / "synth"
    out.mkdir(parents=True, exist_ok=True)

    spy = make_ohlcv(n=3000, seed=1, start="2013-01-02", start_price=140.0)
    aapl = make_ohlcv(n=3000, seed=2, start="2013-01-02", start_price=18.0)

    spy.to_csv(out / "SPY.csv", index_label="date")
    aapl.to_csv(out / "AAPL.csv", index_label="date")

    print(f"Wrote {out / 'SPY.csv'} ({len(spy)} rows)")
    print(f"Wrote {out / 'AAPL.csv'} ({len(aapl)} rows)")


if __name__ == "__main__":
    main()
