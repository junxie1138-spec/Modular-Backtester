import json
from pathlib import Path
from unittest import mock

import pytest

from factory.settings_loader import PromotionCfg
from factory.settings_loader import load_settings
from factory.sortino_migration import (
    _initial_state, _migrate_record, _needs_rerun, _read_bundle_sortino,
    drain_one_retro_promotion, migrate_shard,
)


def test_needs_rerun_same_side_is_false() -> None:
    # Both Sharpe and Sortino below the threshold -> verdict unchanged.
    assert _needs_rerun(0.4, 0.6, 1.0) is False
    # Both above -> verdict unchanged.
    assert _needs_rerun(1.5, 1.2, 1.0) is False


def test_needs_rerun_opposite_sides_is_true() -> None:
    # Sharpe below, Sortino at/above -> the metric swap flips the verdict.
    assert _needs_rerun(0.9, 1.1, 1.0) is True
    # Sharpe above, Sortino below.
    assert _needs_rerun(1.3, 0.8, 1.0) is True


def test_initial_state_below_threshold_is_na() -> None:
    assert _initial_state(
        has_promotion_block=False, oos_sortino=0.5,
        promotion_enabled=True, trigger_threshold=1.0,
    ) == "n/a"


def test_initial_state_eligible_is_pending() -> None:
    assert _initial_state(
        has_promotion_block=False, oos_sortino=1.5,
        promotion_enabled=True, trigger_threshold=1.0,
    ) == "pending"


def test_initial_state_promotion_disabled_is_na() -> None:
    assert _initial_state(
        has_promotion_block=False, oos_sortino=1.5,
        promotion_enabled=False, trigger_threshold=1.0,
    ) == "n/a"


def test_initial_state_existing_promotion_block_is_done() -> None:
    assert _initial_state(
        has_promotion_block=True, oos_sortino=1.5,
        promotion_enabled=True, trigger_threshold=1.0,
    ) == "done"


def _make_bundle(tmp_path: Path, name: str, sortino: float | None) -> Path:
    """Create a fake WFO bundle directory with a summary.json. When `sortino`
    is None the summary.json's oos_summary omits the sortino key.
    """
    bundle = tmp_path / "runs" / name
    bundle.mkdir(parents=True, exist_ok=True)
    oos = {"sharpe": 0.5, "total_return": 0.1, "max_drawdown": -0.05, "n_trades": 12}
    if sortino is not None:
        oos["sortino"] = sortino
    (bundle / "summary.json").write_text(
        json.dumps({"oos_summary": oos, "n_windows": 6}), encoding="utf-8",
    )
    return bundle


def test_read_bundle_sortino_reads_value(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "b_read", sortino=0.77)
    assert _read_bundle_sortino(bundle) == 0.77


def test_read_bundle_sortino_missing_dir_returns_none(tmp_path: Path) -> None:
    assert _read_bundle_sortino(tmp_path / "does_not_exist") is None


def test_read_bundle_sortino_summary_without_sortino_returns_none(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "b_nosortino", sortino=None)
    assert _read_bundle_sortino(bundle) is None


def test_read_bundle_sortino_non_dict_oos_summary_returns_none(tmp_path: Path) -> None:
    bundle = tmp_path / "runs" / "b_bad"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "summary.json").write_text(
        json.dumps({"oos_summary": None, "n_windows": 6}), encoding="utf-8",
    )
    assert _read_bundle_sortino(bundle) is None


def _promo_cfg(*, enabled: bool = True, trigger_threshold: float = 1.0) -> PromotionCfg:
    return PromotionCfg(
        enabled=enabled, tickers=("AAPL", "QQQ", "DIA"), data_source="yfinance",
        min_avg_sortino=0.7, trigger_metric="wfo.oos_sortino",
        trigger_threshold=trigger_threshold,
    )


def _record(
    strategy_id: str = "gen_1", *, status: str = "complete", wfo: bool = True,
    oos_sharpe: float = 0.5, bundle_path: str = "MISSING_BUNDLE",
    oos_sortino: float | None = None, promotion: dict | None = None,
) -> dict:
    """Build a results record. `wfo=False` produces a record with `wfo: None`."""
    rec: dict = {
        "strategy_id": strategy_id, "timestamp": "2026-05-15T00:00:00Z",
        "status": status, "failed_stage": None, "error": None,
        "slots": {}, "idea": {}, "generation_cost_usd": 0.1,
        "backtest": {"sharpe": 0.2}, "optimize": {"best_params": {"x": 1}},
        "promotion": promotion, "alerted": False,
    }
    if wfo:
        w: dict = {
            "oos_sharpe": oos_sharpe, "oos_total_return": 0.1,
            "oos_max_drawdown": -0.05, "oos_n_trades": 12,
            "parameter_stability": {}, "n_windows": 6,
            "run_bundle_path": bundle_path,
        }
        if oos_sortino is not None:
            w["oos_sortino"] = oos_sortino
        rec["wfo"] = w
    else:
        rec["wfo"] = None
    return rec


def test_migrate_record_backfills_sortino_and_block(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "b_ok", sortino=0.42)
    rec = _record(bundle_path=bundle.as_posix(), oos_sharpe=0.5)
    out = _migrate_record(rec, promotion_cfg=_promo_cfg())
    assert out is not None
    assert out["wfo"]["oos_sortino"] == 0.42
    assert out["sortino_migration"]["state"] == "n/a"   # 0.42 < threshold 1.0
    assert out["sortino_migration"]["needs_rerun"] is False
    assert out["sortino_migration"]["migrated_at"].endswith("Z")
    # The original record is not mutated.
    assert "oos_sortino" not in rec["wfo"]
    assert "sortino_migration" not in rec


def test_migrate_record_pending_when_clears_threshold(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "b_high", sortino=1.5)
    rec = _record(bundle_path=bundle.as_posix(), oos_sharpe=0.5)
    out = _migrate_record(rec, promotion_cfg=_promo_cfg())
    assert out is not None
    # Sortino 1.5 >= 1.0 but Sharpe 0.5 < 1.0 -> opposite sides -> needs_rerun.
    assert out["sortino_migration"]["state"] == "pending"
    assert out["sortino_migration"]["needs_rerun"] is True


def test_migrate_record_missing_bundle_returns_none(tmp_path: Path) -> None:
    rec = _record(bundle_path=(tmp_path / "absent").as_posix())
    assert _migrate_record(rec, promotion_cfg=_promo_cfg()) is None


def test_migrate_record_bundle_without_sortino_returns_none(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "b_blank", sortino=None)
    rec = _record(bundle_path=bundle.as_posix())
    assert _migrate_record(rec, promotion_cfg=_promo_cfg()) is None


def test_migrate_record_already_has_sortino_returns_none(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "b_done", sortino=0.4)
    rec = _record(bundle_path=bundle.as_posix(), oos_sortino=0.9)
    assert _migrate_record(rec, promotion_cfg=_promo_cfg()) is None


def test_migrate_record_skips_failed_record() -> None:
    assert _migrate_record(_record(status="failed"), promotion_cfg=_promo_cfg()) is None


def test_migrate_record_skips_record_without_wfo() -> None:
    assert _migrate_record(_record(wfo=False), promotion_cfg=_promo_cfg()) is None


def _settings(tmp_path: Path, *, promotion_enabled: bool, trigger_threshold: float):
    """Write a complete settings.toml under tmp_path and load it. node_id
    defaults to 'local', so the shard is results/local.jsonl.
    """
    toml = f"""
[paths]
backtester_root = "{tmp_path.as_posix()}"
strategies_dir  = "strategies"
configs_dir     = "configs/wfo"
registry_file   = "backtester/strategies/registry.py"
output_runs_dir = "output/runs"
dedup_dir       = "factory/data/dedup"
results_dir     = "factory/data/results"
factory_log     = "factory/logs/factory.log"
tmp_dir         = "factory/data/_tmp"

[generation]
claude_cmd = "claude"
claude_flags = ["-p"]
generation_timeout_sec = 60

[stages]
stage_timeout_sec = 300

[alerts]
alert_threshold_metric = "wfo.oos_sortino"
alert_threshold = 1.0
telegram_bot_token = ""
telegram_chat_id = ""
dashboard_base_url = "http://127.0.0.1:8787"

[loop]
mode = "continuous"
inter_cycle_sleep_sec = 0
max_cycles = 1

[dashboard]
host = "127.0.0.1"
port = 8787
auto_refresh_sec = 10

[promotion]
enabled = {str(promotion_enabled).lower()}
tickers = ["AAPL", "QQQ", "DIA"]
data_source = "yfinance"
min_avg_sortino = 0.7
trigger_metric = "wfo.oos_sortino"
trigger_threshold = {trigger_threshold}
""".strip()
    p = tmp_path / "settings.toml"
    p.write_text(toml, encoding="utf-8")
    return load_settings(p)


def _write_shard(shard: Path, records: list[dict]) -> None:
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8",
    )


def test_migrate_shard_migrates_then_is_idempotent(tmp_path: Path) -> None:
    settings = _settings(tmp_path, promotion_enabled=True, trigger_threshold=1.0)
    bundle = _make_bundle(tmp_path, "bundle_gen_1", sortino=1.5)
    shard = settings.paths.results_dir / f"{settings.node_id}.jsonl"
    _write_shard(shard, [_record("gen_1", bundle_path=bundle.as_posix())])

    n1 = migrate_shard(settings)
    assert n1 == 1
    migrated = json.loads(shard.read_text(encoding="utf-8").splitlines()[0])
    assert migrated["wfo"]["oos_sortino"] == 1.5
    assert migrated["sortino_migration"]["state"] == "pending"

    # Second pass is a no-op — the record already carries oos_sortino.
    assert migrate_shard(settings) == 0


def test_migrate_shard_no_shard_returns_zero(tmp_path: Path) -> None:
    settings = _settings(tmp_path, promotion_enabled=True, trigger_threshold=1.0)
    assert migrate_shard(settings) == 0


def test_migrate_shard_leaves_unbackfillable_record(tmp_path: Path) -> None:
    settings = _settings(tmp_path, promotion_enabled=False, trigger_threshold=1.0)
    shard = settings.paths.results_dir / f"{settings.node_id}.jsonl"
    rec = _record("gen_x", bundle_path=(tmp_path / "gone").as_posix())
    _write_shard(shard, [rec])
    assert migrate_shard(settings) == 0
    # Record is byte-unchanged: still no oos_sortino, no migration block.
    after = json.loads(shard.read_text(encoding="utf-8").splitlines()[0])
    assert "oos_sortino" not in after["wfo"]
    assert "sortino_migration" not in after


def _pending_record(strategy_id: str = "gen_1") -> dict:
    """A complete, already-backfilled record queued for retro-promotion."""
    rec = _record(strategy_id, oos_sortino=1.5)
    rec["sortino_migration"] = {
        "migrated_at": "2026-05-17T00:00:00Z", "needs_rerun": True,
        "state": "pending",
    }
    return rec


def test_drain_runs_promotion_for_pending_record(tmp_path: Path) -> None:
    settings = _settings(tmp_path, promotion_enabled=True, trigger_threshold=1.0)
    cfg = settings.paths.configs_dir / "gen_1.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("run_name: gen_1\n", encoding="utf-8")
    shard = settings.paths.results_dir / f"{settings.node_id}.jsonl"
    _write_shard(shard, [_pending_record("gen_1")])

    from factory.promote import PromotionResult
    fake = PromotionResult(
        ran=True, tickers=("AAPL", "QQQ", "DIA"), per_ticker={},
        avg_sortino=1.2, min_avg_sortino_threshold=0.7, passed=True, error=None,
    )
    with mock.patch(
        "factory.sortino_migration.promote_strategy", return_value=fake,
    ) as ps:
        handled = drain_one_retro_promotion(settings)

    assert handled is True
    assert ps.call_count == 1
    out = json.loads(shard.read_text(encoding="utf-8").splitlines()[0])
    assert out["sortino_migration"]["state"] == "done"
    assert out["promotion"]["passed"] is True


def test_drain_marks_na_when_canonical_config_missing(tmp_path: Path) -> None:
    settings = _settings(tmp_path, promotion_enabled=True, trigger_threshold=1.0)
    shard = settings.paths.results_dir / f"{settings.node_id}.jsonl"
    _write_shard(shard, [_pending_record("gen_no_config")])

    with mock.patch("factory.sortino_migration.promote_strategy") as ps:
        handled = drain_one_retro_promotion(settings)

    assert handled is True
    ps.assert_not_called()
    out = json.loads(shard.read_text(encoding="utf-8").splitlines()[0])
    assert out["sortino_migration"]["state"] == "n/a"


def test_drain_noop_when_nothing_pending(tmp_path: Path) -> None:
    settings = _settings(tmp_path, promotion_enabled=True, trigger_threshold=1.0)
    shard = settings.paths.results_dir / f"{settings.node_id}.jsonl"
    rec = _record("gen_1", oos_sortino=0.5)
    rec["sortino_migration"] = {
        "migrated_at": "2026-05-17T00:00:00Z", "needs_rerun": False,
        "state": "n/a",
    }
    _write_shard(shard, [rec])
    assert drain_one_retro_promotion(settings) is False


def test_drain_noop_on_empty_shard(tmp_path: Path) -> None:
    settings = _settings(tmp_path, promotion_enabled=True, trigger_threshold=1.0)
    assert drain_one_retro_promotion(settings) is False


def test_drain_marks_na_when_best_params_missing(tmp_path: Path) -> None:
    settings = _settings(tmp_path, promotion_enabled=True, trigger_threshold=1.0)
    cfg = settings.paths.configs_dir / "gen_1.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("run_name: gen_1\n", encoding="utf-8")
    shard = settings.paths.results_dir / f"{settings.node_id}.jsonl"
    rec = _pending_record("gen_1")
    rec["optimize"] = None   # a complete record may carry optimize: None
    _write_shard(shard, [rec])

    with mock.patch("factory.sortino_migration.promote_strategy") as ps:
        handled = drain_one_retro_promotion(settings)

    assert handled is True
    ps.assert_not_called()
    out = json.loads(shard.read_text(encoding="utf-8").splitlines()[0])
    assert out["sortino_migration"]["state"] == "n/a"
