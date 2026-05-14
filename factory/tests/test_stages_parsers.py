import json
import time
from pathlib import Path

import pytest

from factory.stages import (
    BundleNotFound,
    find_latest_bundle,
    parse_backtest_summary,
    parse_optimize_summary,
    parse_wfo_summary,
)

FIX = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_parse_backtest_summary_extracts_headline_metrics() -> None:
    raw = _load("sample_backtest_summary.json")
    parsed = parse_backtest_summary(raw, bundle_path=Path("output/runs/x"))
    assert parsed["sharpe"] == pytest.approx(0.10408681366641073)
    assert parsed["total_return"] == pytest.approx(0.04789374275677183)
    assert parsed["max_drawdown"] == pytest.approx(-0.18412515335473634)
    assert parsed["win_rate"] == pytest.approx(0.4965034965034965)
    assert parsed["n_trades"] == 286
    assert parsed["run_bundle_path"] == "output/runs/x"


def test_parse_optimize_summary_extracts_best_params_and_score() -> None:
    raw = _load("sample_optimize_summary.json")
    parsed = parse_optimize_summary(raw, bundle_path=Path("output/runs/y"))
    assert parsed["best_params"] == {"fast": 20, "slow": 100, "size": 1.0}
    assert parsed["objective"] == "sharpe"
    # best_score == best_summary[objective]
    assert parsed["best_score"] == pytest.approx(0.6652567100252722)
    assert parsed["run_bundle_path"] == "output/runs/y"


def test_parse_wfo_summary_extracts_oos_block() -> None:
    raw = _load("sample_wfo_summary.json")
    parsed = parse_wfo_summary(raw, bundle_path=Path("output/runs/z"))
    assert parsed["oos_sharpe"] == pytest.approx(0.6897185480952924)
    assert parsed["oos_total_return"] == pytest.approx(0.31078674058870415)
    assert parsed["oos_max_drawdown"] == pytest.approx(-0.07374129512318983)
    assert parsed["oos_n_trades"] == 109
    assert parsed["n_windows"] == 6
    assert parsed["parameter_stability"]["entry_percentile"]["mode"] == 30.0


def test_parsers_raise_on_missing_keys() -> None:
    with pytest.raises(KeyError):
        parse_backtest_summary({}, bundle_path=Path("x"))
    with pytest.raises(KeyError):
        parse_optimize_summary({}, bundle_path=Path("x"))
    with pytest.raises(KeyError):
        parse_wfo_summary({"oos_summary": {}}, bundle_path=Path("x"))


def test_find_latest_bundle_picks_newest_matching_run_name(tmp_path: Path) -> None:
    output_runs = tmp_path / "output" / "runs"
    output_runs.mkdir(parents=True)
    # Create three bundles with different mtimes.
    (output_runs / "20260101_0900_gen_X").mkdir()
    time.sleep(0.01)
    (output_runs / "20260101_1000_gen_X").mkdir()
    time.sleep(0.01)
    (output_runs / "20260101_1100_gen_Y").mkdir()
    found = find_latest_bundle(output_runs_dir=output_runs, run_name="gen_X")
    assert found.name == "20260101_1000_gen_X"


def test_find_latest_bundle_raises_when_no_match(tmp_path: Path) -> None:
    output_runs = tmp_path / "output" / "runs"
    output_runs.mkdir(parents=True)
    with pytest.raises(BundleNotFound):
        find_latest_bundle(output_runs_dir=output_runs, run_name="missing")
