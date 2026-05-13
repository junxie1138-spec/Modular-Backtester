from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.fixtures.synthetic import make_ohlcv


def test_run_wfo_cli_produces_window_and_oos_artifacts(tmp_path: Path):
    raw = tmp_path / "data"
    raw.mkdir()
    make_ohlcv(n=600, seed=42).to_csv(raw / "SYN.csv", index_label="date")

    out = tmp_path / "runs"
    cfg = tmp_path / "wfo.yaml"
    cfg.write_text(f"""
run_name: wfo_smoke
strategy: sma_cross
strategy_params:
  fast: 10
  slow: 30
data:
  symbols: ["SYN"]
  timeframe: "1d"
  start: "2020-01-02"
  end: "2030-12-31"
  source: "csv"
  root: "{raw.as_posix()}"
execution:
  initial_cash: 10000
  commission_bps: 0
  slippage_bps: 0
portfolio:
  size: 1.0
optimization:
  objective: sharpe
  param_space:
    fast: [5, 10]
    slow: [20, 50]
wfo:
  enabled: true
  train_bars: 200
  test_bars: 50
  step_bars: 50
output_root: "{out.as_posix()}"
""")
    res = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_wfo", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    run_dir = next(out.iterdir())
    for name in ("config_resolved.yaml", "summary.json", "window_results.json",
                 "oos_equity_curve.csv", "oos_trades.csv", "logs.txt"):
        assert (run_dir / name).exists(), name

    summary = json.loads((run_dir / "summary.json").read_text())
    assert "oos_summary" in summary
    assert "is_summary_avg" in summary
    assert "parameter_stability" in summary
