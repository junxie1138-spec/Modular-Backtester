from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# Hard-coded golden captured from the pre-trailing-stop sma_cross_spy run.
# Update ONLY when an intentional simulator change requires a new baseline.
EXPECTED = {
    "total_return": 0.3509706358824263,
    "sharpe": 0.30048402218924875,
    "max_drawdown": -0.36916662882487716,
    "n_trades": 65,
    "final_equity": 135097.06358824263,
}


def test_sma_cross_spy_unchanged_with_trailing_disabled(tmp_path: Path):
    """Acceptance criterion 2: with trailing-stop fields absent (or None),
    the bundled long-only SMA-Cross config produces the v0.2.0 numerics
    exactly. n_trades is exact integer; floats compared to 1e-9 abs / 1e-12 rel."""
    repo_root = Path(__file__).resolve().parents[2]
    spy_root = (repo_root / "data" / "raw").as_posix()
    out = tmp_path / "runs"
    out.mkdir()

    cfg = tmp_path / "sma_cross_spy.yaml"
    cfg.write_text(f"""
run_name: sma_cross_spy
strategy: sma_cross
strategy_params:
  fast: 20
  slow: 50
  size: 1.0
data:
  symbols: ["SPY"]
  timeframe: "1d"
  start: "2015-01-02"
  end: "2024-12-31"
  source: "csv"
  root: "{spy_root}"
execution:
  initial_cash: 100000
  commission_bps: 2
  slippage_bps: 5
  allow_fractional: false
portfolio:
  sizing_mode: "percent_equity"
  size: 0.95
output_root: "{out.as_posix()}"
""")

    res = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_backtest", "--config", str(cfg)],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert res.returncode == 0, res.stderr

    run_dir = next(out.iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())

    assert summary["n_trades"] == EXPECTED["n_trades"]
    for key in ("total_return", "sharpe", "max_drawdown", "final_equity"):
        assert summary[key] == pytest.approx(EXPECTED[key], abs=1e-9, rel=1e-12), (
            f"{key}: got {summary[key]!r}, expected {EXPECTED[key]!r}"
        )
