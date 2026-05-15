from pathlib import Path

import pytest

from factory.results import (
    Record,
    build_failed_record,
    build_record,
    read_records,
    write_record,
)


def _slots() -> dict[str, str]:
    return {
        "strategy_family": "momentum",
        "signal_primitive": "close-to-close returns",
        "holding_horizon": "3-5 days",
        "direction": "long-only",
        "constraint_twist": "<=2 tunable params",
        "inspiration_anchor": "hysteresis control",
    }


def _idea() -> dict:
    return {
        "one_line_summary": "test idea",
        "hypothesis": "h",
        "novelty_justification": "n",
        "failure_mode": "f",
        "allow_short": False,
    }


def test_build_record_complete_has_all_fields() -> None:
    r = build_record(
        strategy_id="gen_42",
        timestamp="2026-05-15T09:00:00Z",
        slots=_slots(),
        idea=_idea(),
        generation_cost_usd=0.034,
        backtest={"sharpe": 1.1, "total_return": 0.2, "max_drawdown": -0.05,
                  "win_rate": 0.6, "n_trades": 20, "run_bundle_path": "p1"},
        optimize={"best_params": {"size": 1.0}, "objective": "sharpe",
                  "best_score": 1.3, "run_bundle_path": "p2"},
        wfo={"oos_sharpe": 1.2, "oos_total_return": 0.18,
             "oos_max_drawdown": -0.04, "oos_n_trades": 25,
             "parameter_stability": {}, "n_windows": 6,
             "run_bundle_path": "p3"},
        alerted=True,
    )
    assert r["status"] == "complete"
    assert r["failed_stage"] is None
    assert r["error"] is None
    assert r["strategy_id"] == "gen_42"
    assert r["slots"]["strategy_family"] == "momentum"
    assert r["backtest"]["sharpe"] == 1.1
    assert r["wfo"]["oos_sharpe"] == 1.2
    assert r["alerted"] is True


def test_build_failed_record_has_failed_stage_and_error() -> None:
    r = build_failed_record(
        strategy_id="gen_43",
        timestamp="2026-05-15T09:01:00Z",
        slots=_slots(),
        idea=_idea(),
        generation_cost_usd=0.012,
        failed_stage="validation",
        error="missing .shift(1)",
    )
    assert r["status"] == "failed"
    assert r["failed_stage"] == "validation"
    assert r["error"] == "missing .shift(1)"
    assert r["backtest"] is None
    assert r["optimize"] is None
    assert r["wfo"] is None
    assert r["alerted"] is False


def test_build_failed_record_for_generation_failure_has_no_idea() -> None:
    r = build_failed_record(
        strategy_id=None,
        timestamp="2026-05-15T09:02:00Z",
        slots=_slots(),
        idea=None,
        generation_cost_usd=0.0,
        failed_stage="generation",
        error="claude -p timeout",
    )
    assert r["status"] == "failed"
    assert r["failed_stage"] == "generation"
    assert r["strategy_id"] is None
    assert r["idea"] is None


def test_write_record_creates_node_shard(tmp_path: Path) -> None:
    d = tmp_path / "results"
    write_record(d, {"a": 1, "strategy_id": "x"}, node_id="desk")
    assert (d / "desk.jsonl").exists()


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    d = tmp_path / "results"
    write_record(d, {"a": 1, "strategy_id": "x"}, node_id="local")
    write_record(d, {"a": 2, "strategy_id": "y"}, node_id="local")
    write_record(d, {"a": 3, "strategy_id": "z"}, node_id="local")
    assert [r["a"] for r in read_records(d)] == [1, 2, 3]


def test_read_records_unions_shards(tmp_path: Path) -> None:
    d = tmp_path / "results"
    write_record(d, {"a": 1, "timestamp": "2026-05-15T09:00:00Z"}, node_id="desk")
    write_record(d, {"a": 2, "timestamp": "2026-05-15T08:00:00Z"}, node_id="laptop")
    recs = read_records(d)
    assert {r["a"] for r in recs} == {1, 2}
    # Callers that need ordering sort by the record timestamp.
    ordered = sorted(recs, key=lambda r: r["timestamp"])
    assert [r["a"] for r in ordered] == [2, 1]


def test_read_records_handles_missing_dir(tmp_path: Path) -> None:
    assert read_records(tmp_path / "nothing") == []


def test_read_records_skips_blank_lines(tmp_path: Path) -> None:
    d = tmp_path / "results"
    d.mkdir()
    (d / "local.jsonl").write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert read_records(d) == [{"a": 1}, {"a": 2}]


def test_read_records_raises_on_malformed_line(tmp_path: Path) -> None:
    d = tmp_path / "results"
    d.mkdir()
    (d / "local.jsonl").write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_records(d)
