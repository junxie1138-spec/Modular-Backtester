import json
from pathlib import Path
from unittest import mock

import pytest

from factory.discovery import _candidate, _db, _hash, _put, evaluate, index_library, review
from factory.stages import run_backtest_stage


def test_library_reads_all_shards_and_marks_missing_wfo(tmp_path: Path) -> None:
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    (raw / "SPY.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2015-01-02,1,2,1,2,100\n2024-12-31,2,3,2,3,100\n"
        "2025-01-02,3,4,3,4,100\n", encoding="utf-8")
    strategy = tmp_path / "strategies/gen_x.py"
    strategy.parent.mkdir()
    strategy.write_text("# source\n", encoding="utf-8")
    config = tmp_path / "configs/wfo/gen_x.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("strategy: gen_x\nwfo: {enabled: true}\n", encoding="utf-8")
    shards = tmp_path / "factory/data/results"
    shards.mkdir(parents=True)
    (shards / "a.jsonl").write_text(json.dumps({"strategy_id": "gen_x", "timestamp": "1", "wfo": None}) + "\n")
    (shards / "b.jsonl").write_text(json.dumps({"strategy_id": "gen_x", "timestamp": "2", "wfo": None}) + "\n")
    bundle = tmp_path / "output/runs/20250101_gen_x"
    bundle.mkdir(parents=True)
    (bundle / "summary.json").write_text("{}")

    library = index_library(tmp_path)
    assert len(library["result_shards"]) == 2
    assert library["entries"]["gen_x"]["result"]["timestamp"] == "2"
    assert library["entries"]["gen_x"]["wfo"]["status"] == "missing"
    assert "output/runs/20250101_gen_x" in library["run_artifacts"]
    assert library["search_data"]["last_bar"] == "2024-12-31"
    assert "2025-01-02" not in (tmp_path / library["search_data"]["path"]).read_text()


def test_review_requires_shortlist(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        _put(db, {"id": "disc_1", "source_id": "gen_x", "status": "generated",
                  "evidence": {}, "attempts": {}, "error": None})
    with pytest.raises(ValueError, match="shortlisted"):
        review(tmp_path, "disc_1")
    with _db(tmp_path) as db:
        assert _candidate(db, "disc_1")["attempts"] == {}


def test_stage_directs_candidate_module_and_output_to_discovery(tmp_path: Path) -> None:
    config = tmp_path / "candidate.yaml"
    config.write_text("run_name: disc_1\nstrategy: disc_1\n", encoding="utf-8")
    runs = tmp_path / "discovery/runs/candidates"
    bundle = runs / "20260101_0900_disc_1"
    bundle.mkdir(parents=True)
    (bundle / "summary.json").write_text(json.dumps({"sharpe": 1, "total_return": .1,
        "max_drawdown": -.1, "win_rate": .5, "n_trades": 4}))
    with mock.patch("factory.stages.subprocess.run", return_value=mock.Mock(returncode=0, stderr="")) as run:
        result = run_backtest_stage(canonical_config=config, strategy_id="disc_1",
            output_runs_dir=runs, tmp_dir=tmp_path / "discovery/tmp", timeout_sec=30,
            strategy_dir=tmp_path / "discovery/strategies")
    assert result.bundle_path == bundle
    env = run.call_args.kwargs["env"]
    assert env["BACKTESTER_OUTPUT_ROOT"] == str(runs)
    assert env["DISCOVERY_STRATEGY_DIR"] == str(tmp_path / "discovery/strategies")


def test_missing_novelty_or_incrementality_blocks_shortlist(tmp_path: Path) -> None:
    code = tmp_path / "discovery/strategies/disc_1.py"
    config = tmp_path / "discovery/configs/disc_1.yaml"
    code.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    code.write_text("# candidate\n")
    config.write_text("execution: {commission_bps: 2, slippage_bps: 5}\n")
    item = {"id": "disc_1", "source_id": "gen_x", "status": "generated",
            "evidence": {"analysis": {"summary": "new idea", "justification": ""},
                         "search_data_sha256": "search-hash", "code_sha256": _hash(code),
                         "config_sha256": _hash(config)}, "attempts": {}, "error": None}
    with _db(tmp_path) as db:
        _put(db, item)
    library = {"search_data": {"sha256": "search-hash"}, "entries": {
        "gen_x": {"wfo": {"status": "available", "bundle": str(tmp_path / "source")},
                  "code_sha256": "source-hash", "result": {"idea": {"one_line_summary": "old idea"}}}}}

    def stage(_root, _sid, _cfg, name, group, _overrides, _candidate):
        parsed = {"best_params": {"p": 1}} if name == "optimize" else (
            {"n_windows": 15, "oos_sortino": 1, "oos_n_trades": 20} if name == "wfo" else {"sharpe": 1})
        return {"parsed": parsed, "bundle": str(tmp_path / group)}

    with mock.patch("factory.discovery.index_library", return_value=library), \
         mock.patch("factory.discovery._stage", side_effect=stage), \
         mock.patch("factory.discovery._incrementality", return_value={"status": "missing"}):
        result = evaluate(tmp_path, "disc_1")
    assert result["status"] == "rejected"
    assert result["evidence"]["gates"] == {"novelty": False, "wfo": True,
                                            "stress": True, "incrementality": False}
    assert result["attempts"] == {"backtest": 1, "optimize": 1, "wfo": 1, "stress": 1}


def test_review_handoff_pins_period_and_hashes(tmp_path: Path) -> None:
    code = tmp_path / "discovery/strategies/disc_1.py"
    config = tmp_path / "discovery/configs/disc_1.yaml"
    search = tmp_path / "discovery/data/search/SPY.csv"
    for path in (code, config, search):
        path.parent.mkdir(parents=True, exist_ok=True)
    code.write_text("# candidate\n")
    config.write_text("execution: {commission_bps: 2, slippage_bps: 5}\n")
    search.write_text("search data\n")
    item = {"id": "disc_1", "source_id": "gen_x", "status": "shortlisted",
            "evidence": {"code_sha256": _hash(code), "config_sha256": _hash(config),
                         "search_data_sha256": _hash(search),
                         "optimize": {"parsed": {"best_params": {"period": 10}}},
                         "gates": {"novelty": True}, "wfo": {"parsed": {}},
                         "stress": {"parsed": {}}, "incrementality": {"passed": True}},
            "attempts": {}, "error": None}
    with _db(tmp_path) as db:
        _put(db, item)
    outcome = {"parsed": {"n_trades": 8}, "raw_summary": {"sortino": 1.2},
               "bundle": str(tmp_path / "discovery/runs/review/run"), "summary_sha256": "summary-hash"}
    with mock.patch("factory.discovery._snapshot", return_value={"sha256": "review-data-hash"}), \
         mock.patch("factory.discovery._stage", return_value=outcome) as run:
        result = review(tmp_path, "disc_1")
    assert result["status"] == "approved"
    overrides = run.call_args.args[5]
    assert (overrides["data"]["start"], overrides["data"]["end"]) == ("2025-01-01", "2026-05-14")
    handoff = Path(result["evidence"]["handoff"]["path"])
    doc = json.loads(handoff.read_text())
    assert result["evidence"]["handoff"]["sha256"] == _hash(handoff)
    assert doc["review"]["parameters"] == {"period": 10}
    assert doc["review"]["costs"]["commission_bps"] == 2
    assert doc["review"]["data"]["sha256"] == "review-data-hash"
    assert doc["review"]["code_sha256"] == _hash(code)
    assert doc["review"]["config_sha256"] == _hash(config)
