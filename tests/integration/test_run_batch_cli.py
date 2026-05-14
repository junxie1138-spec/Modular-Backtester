from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.fixtures.synthetic import make_ohlcv


def test_run_batch_iterates_symbols(tmp_path: Path):
    raw = tmp_path / "data"
    raw.mkdir()
    make_ohlcv(n=300, seed=10).to_csv(raw / "A.csv", index_label="date")
    make_ohlcv(n=300, seed=11).to_csv(raw / "B.csv", index_label="date")
    out = tmp_path / "runs"
    cfg = tmp_path / "batch.yaml"
    cfg.write_text(f"""
run_name: batch_smoke
strategy: sma_cross
strategy_params: {{fast: 10, slow: 30}}
data:
  symbols: ["A", "B"]
  timeframe: "1d"
  start: "2020-01-02"
  end: "2030-12-31"
  source: "csv"
  root: "{raw.as_posix()}"
execution: {{initial_cash: 10000, commission_bps: 0, slippage_bps: 0}}
portfolio: {{size: 1.0}}
output_root: "{out.as_posix()}"
""")
    res = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_batch", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    run_dir = next(out.iterdir())
    assert (run_dir / "batch_summary.json").exists()
    by_symbol = json.loads((run_dir / "batch_summary.json").read_text())
    assert set(by_symbol.keys()) == {"A", "B"}
    for sym, summary in by_symbol.items():
        assert "total_return" in summary
        assert (run_dir / f"{sym}_equity_curve.csv").exists()


def test_multi_symbol_mean_reversion_smoke(tmp_path, monkeypatch):
    """End-to-end: 3-symbol universe, 2024 window, exit 0 with artifacts."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    cfg = Path("configs/backtests/mean_rev_v04_smoke.yaml")
    assert cfg.exists(), f"missing {cfg}"
    monkeypatch.setenv("BACKTESTER_OUTPUT_ROOT", str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_batch", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    run_dirs = list(tmp_path.glob("*mean_rev_v04_smoke*"))
    assert run_dirs, f"no run dir under {tmp_path}"
    run_dir = run_dirs[0]
    assert (run_dir / "portfolio_equity_curve.csv").exists()
    assert (run_dir / "batch_summary.json").exists()
    summary = json.loads((run_dir / "batch_summary.json").read_text())
    assert "portfolio_total_return" in summary
    assert summary["n_symbols"] == 3
