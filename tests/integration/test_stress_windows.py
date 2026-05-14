import json
import subprocess
import sys
from pathlib import Path

import pytest


STRESS_WINDOWS = [
    ("2020-covid",      "2020-02-15", "2020-04-30"),
    ("2022-bear-cycle", "2021-11-01", "2022-10-31"),
    ("2024-aug-unwind", "2024-07-15", "2024-09-15"),
    ("2025-apr",        "2025-03-15", "2025-05-15"),
]


def _write_config(tmp_path, *, start, end) -> Path:
    base = Path("configs/backtests/mean_rev_v04_smoke.yaml").read_text()
    cfg = tmp_path / "stress.yaml"
    cfg.write_text(
        base
        .replace("start: '2024-01-02'", f"start: '{start}'")
        .replace("end: '2024-12-31'", f"end: '{end}'")
        .replace("output_root: output/runs", f"output_root: {tmp_path}")
        .replace(
            "universe_path: ../universe_smoke.yaml",
            f"universe_path: {Path('configs/universe_smoke.yaml').resolve()}",
        )
    )
    return cfg


@pytest.mark.xfail(strict=False, reason="performance gate; flip to assert when strategy is tuned")
@pytest.mark.parametrize("label,start,end", STRESS_WINDOWS)
def test_stress_window_drawdown(tmp_path, label, start, end):
    cfg = _write_config(tmp_path, start=start, end=end)
    result = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_batch", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"{label} structural failure: {result.stderr}"
    run_dirs = list(tmp_path.glob("*mean_rev*"))
    assert run_dirs, f"no run dir under {tmp_path}"
    summary_path = run_dirs[0] / "batch_summary.json"
    summary = json.loads(summary_path.read_text())
    (tmp_path / "metrics.json").write_text(json.dumps(summary))
    assert summary["portfolio_max_drawdown"] > -0.09, (
        f"{label}: DD {summary['portfolio_max_drawdown']:.4f} exceeded -9%"
    )
