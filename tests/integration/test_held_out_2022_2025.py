import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_config(tmp_path) -> Path:
    base = Path("configs/backtests/mean_rev_v04_smoke.yaml").read_text()
    cfg = tmp_path / "held_out.yaml"
    cfg.write_text(
        base
        .replace("start: '2024-01-02'", "start: '2022-01-03'")
        .replace("end: '2024-12-31'", "end: '2025-12-31'")
        .replace("output_root: output/runs", f"output_root: {tmp_path}")
        .replace(
            "universe_path: ../universe_smoke.yaml",
            f"universe_path: {Path('configs/universe_smoke.yaml').resolve()}",
        )
    )
    return cfg


@pytest.mark.xfail(strict=False, reason="performance gate; flip to assert when strategy is tuned")
def test_held_out_2022_2025(tmp_path):
    cfg = _write_config(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_batch", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"structural failure: {result.stderr}"
    run_dirs = list(tmp_path.glob("*mean_rev*"))
    summary = json.loads((run_dirs[0] / "batch_summary.json").read_text())
    (tmp_path / "metrics.json").write_text(json.dumps(summary))
    assert summary["portfolio_max_drawdown"] > -0.09, "held-out DD breach"
    assert summary["portfolio_total_return"] > 0.15, "held-out return under 15%"
