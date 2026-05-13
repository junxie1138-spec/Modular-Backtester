from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.fixtures.synthetic import make_ohlcv


def test_run_optimize_cli_writes_grid_results(tmp_path: Path):
    raw = tmp_path / "data"
    raw.mkdir()
    make_ohlcv(n=400, seed=21).to_csv(raw / "SYN.csv", index_label="date")

    out = tmp_path / "runs"
    cfg = tmp_path / "opt.yaml"
    cfg.write_text(f"""
run_name: opt_smoke
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
  size: 1.0
optimization:
  objective: sharpe
  param_space:
    fast: [5, 10]
    slow: [20, 50]
output_root: "{out.as_posix()}"
""")

    res = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_optimize", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    run_dir = next(out.iterdir())
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "grid_results.json").exists()
    grid = json.loads((run_dir / "grid_results.json").read_text())
    assert len(grid) == 4
    assert "best_params" in json.loads((run_dir / "summary.json").read_text())
