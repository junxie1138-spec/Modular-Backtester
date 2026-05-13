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
