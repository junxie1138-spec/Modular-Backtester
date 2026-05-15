import json
from pathlib import Path

import pytest


def _write_records(results_dir: Path, recs: list[dict]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    shard = results_dir / "local.jsonl"
    with shard.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def app_with_records(tmp_settings_file: Path, tmp_path: Path):
    from factory.settings_loader import load_settings
    from factory.dashboard.server import create_app
    s = load_settings(tmp_settings_file)
    _write_records(s.paths.results_dir, [
        {"strategy_id": "gen_1", "timestamp": "2026-05-15T09:00:00Z",
         "status": "complete", "failed_stage": None, "error": None,
         "slots": {"strategy_family": "momentum"},
         "idea": {"one_line_summary": "first", "allow_short": False},
         "generation_cost_usd": 0.03,
         "generation_tokens": {"input": 3120, "output": 3540,
                               "cache_creation": 0, "cache_read": 18000},
         "backtest": {"sharpe": 0.5, "total_return": 0.1, "max_drawdown": -0.1,
                      "win_rate": 0.5, "n_trades": 10, "run_bundle_path": "p"},
         "optimize": {"best_params": {}, "objective": "sharpe", "best_score": 0.7, "run_bundle_path": "p"},
         "wfo": {"oos_sharpe": 1.2, "oos_total_return": 0.2, "oos_max_drawdown": -0.05,
                 "oos_n_trades": 25, "parameter_stability": {}, "n_windows": 6,
                 "run_bundle_path": "p"},
         "alerted": True},
        {"strategy_id": "gen_2", "timestamp": "2026-05-15T09:05:00Z",
         "status": "failed", "failed_stage": "validation",
         "error": "missing .shift(1)",
         "slots": {"strategy_family": "breakout"},
         "idea": {"one_line_summary": "second"},
         "generation_cost_usd": 0.02,
         "backtest": None, "optimize": None, "wfo": None, "alerted": False},
        {"strategy_id": None, "timestamp": "2026-05-15T09:10:00Z",
         "status": "failed", "failed_stage": "generation",
         "error": "timeout",
         "slots": {"strategy_family": "momentum"},
         "idea": None, "generation_cost_usd": 0.0,
         "backtest": None, "optimize": None, "wfo": None, "alerted": False},
    ])
    app = create_app(settings=s)
    app.config["TESTING"] = True
    return app.test_client(), s


def test_overview_html_renders(app_with_records) -> None:
    client, _ = app_with_records
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "gen_1" in body
    assert "first" in body or "compression" in body or "gen_2" in body
    # The "shortlist signal" framing must be visible (spec §9 landmine 1).
    assert "shortlist signal" in body.lower() or "shortlist" in body.lower()


def test_api_records_returns_jsonl(app_with_records) -> None:
    client, _ = app_with_records
    resp = client.get("/api/records")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) == 3
    assert data[0]["strategy_id"] == "gen_1"


def test_api_summary_aggregates_counts(app_with_records) -> None:
    client, s = app_with_records
    resp = client.get("/api/summary")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total_cycles"] == 3
    assert data["completes"] == 1
    assert data["failures_by_stage"]["validation"] == 1
    assert data["failures_by_stage"]["generation"] == 1
    # Threshold metric = wfo.oos_sharpe, threshold = 1.0; gen_1's 1.2 clears.
    assert data["above_threshold"] == 1
    assert data["cumulative_spend_usd"] == pytest.approx(0.05)
    assert data["threshold_metric"] == "wfo.oos_sharpe"
    assert data["threshold_value"] == 1.0


def test_detail_view_renders_complete_record(app_with_records) -> None:
    client, _ = app_with_records
    resp = client.get("/strategy/gen_1")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "gen_1" in body
    assert "shortlist signal" in body.lower()
    # OOS values should appear formatted.
    assert "1.200" in body or "1.2" in body  # oos_sharpe
    # Run-bundle path is surfaced so the user can inspect artifacts.
    assert "run_bundle_path" in body or "Run bundle" in body


def test_detail_view_for_failed_cycle_shows_error(app_with_records) -> None:
    client, _ = app_with_records
    resp = client.get("/strategy/gen_2")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "missing .shift(1)" in body
    assert "failed" in body.lower()


def test_detail_view_404_on_missing_id(app_with_records) -> None:
    client, _ = app_with_records
    resp = client.get("/strategy/does_not_exist")
    assert resp.status_code == 404


def test_overview_carries_refresh_dataset(app_with_records) -> None:
    client, _ = app_with_records
    body = client.get("/").get_data(as_text=True)
    assert 'data-refresh-sec="10"' in body
    assert 'data-threshold-metric="wfo.oos_sharpe"' in body
    assert 'data-threshold-value="1.0"' in body


def test_overview_has_sortable_headers(app_with_records) -> None:
    client, _ = app_with_records
    body = client.get("/").get_data(as_text=True)
    assert 'data-sort="oos_sharpe"' in body
    assert 'data-sort="timestamp"' in body


def test_api_summary_sums_cumulative_tokens(app_with_records) -> None:
    """Cumulative tokens = input + output + cache_creation + cache_read,
    summed over all records; records with no generation_tokens add 0.
    """
    client, _ = app_with_records
    data = client.get("/api/summary").get_json()
    # gen_1: 3120 + 3540 + 0 + 18000 = 24660. The other two records: no tokens -> 0.
    assert data["cumulative_tokens"] == 24660


def test_detail_view_shows_token_breakdown(app_with_records) -> None:
    """gen_1 has generation_tokens -> the detail view shows input/output/cached.
    cached = cache_creation + cache_read = 0 + 18000 = 18000.
    """
    client, _ = app_with_records
    body = client.get("/strategy/gen_1").get_data(as_text=True)
    assert "3120" in body          # input
    assert "3540" in body          # output
    assert "18000" in body         # cached = 0 + 18000
    assert "cached" in body.lower()


def test_detail_view_shows_na_when_no_tokens(app_with_records) -> None:
    """gen_2 has no generation_tokens key -> the detail view shows n/a."""
    client, _ = app_with_records
    body = client.get("/strategy/gen_2").get_data(as_text=True)
    assert "Generation tokens" in body
    assert "n/a" in body


def test_overview_shows_cumulative_tokens(app_with_records) -> None:
    """The overview shows a Cumulative tokens counter (24660 from gen_1)."""
    client, _ = app_with_records
    body = client.get("/").get_data(as_text=True)
    assert "Cumulative tokens" in body
    assert "24660" in body
