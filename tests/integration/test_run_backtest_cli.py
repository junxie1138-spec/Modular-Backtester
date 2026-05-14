from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tests.fixtures.synthetic import make_ohlcv


def _write_data(tmp_path: Path) -> Path:
    raw = tmp_path / "data"
    raw.mkdir()
    df = make_ohlcv(n=400, seed=8)
    df.to_csv(raw / "SYN.csv", index_label="date")
    return raw


def _write_config(tmp_path: Path, raw: Path, out: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
run_name: smoke_run
strategy: sma_cross
strategy_params:
  fast: 10
  slow: 30
data:
  symbols: ["SYN"]
  timeframe: "1d"
  start: "2020-01-02"
  end: "2026-12-31"
  source: "csv"
  root: "{raw.as_posix()}"
execution:
  initial_cash: 10000
  commission_bps: 0
  slippage_bps: 0
portfolio:
  sizing_mode: "percent_equity"
  size: 1.0
output_root: "{out.as_posix()}"
""")
    return cfg


def test_run_backtest_cli_produces_artifacts(tmp_path: Path):
    raw = _write_data(tmp_path)
    out = tmp_path / "runs"
    cfg = _write_config(tmp_path, raw, out)

    res = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_backtest", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    runs = list(out.iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    for f in ("config_resolved.yaml", "summary.json", "trades.csv",
              "positions.csv", "equity_curve.csv", "logs.txt"):
        assert (run_dir / f).exists(), f"missing artifact: {f}"

    summary = json.loads((run_dir / "summary.json").read_text())
    assert "total_return" in summary
    assert summary["symbol"] == "SYN"


def test_run_backtest_cli_rsi_long_short_on_spy(tmp_path: Path):
    """Run the new strategy via CLI on bundled SPY data. Verify trades.csv
    contains a short entry (positions.csv has at least one negative qty)."""
    out = tmp_path / "runs"
    cfg = tmp_path / "rsi_ls.yaml"
    repo_root = Path(__file__).resolve().parents[2]
    spy_root = (repo_root / "data" / "raw").as_posix()

    cfg.write_text(f"""
run_name: rsi_long_short_spy_smoke
strategy: rsi_long_short
strategy_params:
  period: 14
  oversold: 30
  overbought: 70
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
  commission_bps: 1
  slippage_bps: 2
  allow_fractional: false
  allow_short: true
portfolio:
  sizing_mode: "percent_equity"
  size: 0.9
output_root: "{out.as_posix()}"
""")

    res = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_backtest", "--config", str(cfg)],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert res.returncode == 0, res.stderr

    run_dir = next(out.iterdir())
    trades = pd.read_csv(run_dir / "trades.csv")
    positions = pd.read_csv(run_dir / "positions.csv")
    summary = json.loads((run_dir / "summary.json").read_text())

    assert summary["symbol"] == "SPY"
    assert summary["n_trades"] > 0, "expected at least one trade on multi-year SPY history"
    # The strategy holds both directions over a decade — at least one short.
    assert (positions["qty"] < 0).any(), "expected at least one short position bar"
    # Both BUY and SELL fills should appear (long entries and short entries).
    assert "buy" in set(trades["side"]) and "sell" in set(trades["side"])
