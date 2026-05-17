import json
from pathlib import Path
from unittest import mock

import pytest
import yaml

from factory.promote import (
    PromotionResult,
    _build_promotion_config,
    promote_strategy,
)
from factory.settings_loader import PromotionCfg
from factory.stages import StageError


def _canonical_config(strategy_id: str) -> str:
    cfg = {
        "run_name": strategy_id,
        "strategy": strategy_id,
        "strategy_params": {"size": 1.0, "fast": 20},
        "data": {"symbols": ["SPY"], "timeframe": "1d", "start": "2015-01-02",
                 "end": "2024-12-31", "source": "csv", "root": "data/raw"},
        "execution": {"initial_cash": 100000, "commission_bps": 2,
                      "slippage_bps": 5, "allow_fractional": False, "allow_short": False},
        "portfolio": {"sizing_mode": "percent_equity", "size": 0.95},
        "optimization": {"objective": "sharpe", "param_space": {"fast": [10, 20, 30]}},
        "wfo": {"enabled": True, "train_bars": 756, "test_bars": 252, "step_bars": 252},
    }
    return yaml.safe_dump(cfg, sort_keys=False)


def _cfg(min_avg_sortino: float = 0.7, tickers: tuple[str, ...] = ("AAPL", "QQQ", "DIA")) -> PromotionCfg:
    return PromotionCfg(
        enabled=True,
        tickers=tickers,
        data_source="yfinance",
        min_avg_sortino=min_avg_sortino,
        trigger_metric="wfo.oos_sortino",
        trigger_threshold=1.0,
    )


def test_build_promotion_config_swaps_symbol_source_and_params(tmp_path: Path) -> None:
    canonical = tmp_path / "configs" / "wfo" / "gen_x.yaml"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(_canonical_config("gen_x"), encoding="utf-8")

    cfg_path, run_name = _build_promotion_config(
        canonical_path=canonical,
        strategy_id="gen_x",
        ticker="AAPL",
        optimized_params={"size": 0.5, "fast": 30},
        data_source="yfinance",
        tmp_dir=tmp_path / "_tmp",
    )
    assert cfg_path.exists()
    assert run_name == "gen_x_promo_AAPL_wfo"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["data"]["symbols"] == ["AAPL"]
    assert cfg["data"]["source"] == "yfinance"
    assert cfg["strategy_params"] == {"size": 0.5, "fast": 30}
    assert cfg["run_name"] == "gen_x_promo_AAPL_wfo"
    # Untouched fields carry through.
    assert cfg["strategy"] == "gen_x"
    assert cfg["wfo"]["enabled"] is True


def _seed_promotion_bundle(output_runs_dir: Path, run_name: str, oos_sortino: float) -> None:
    """Pre-create a bundle dir + summary.json so the parser can find it."""
    bundle = output_runs_dir / f"20260101_0900_{run_name}"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "summary.json").write_text(json.dumps({
        "oos_summary": {
            "sharpe": oos_sortino,
            "sortino": oos_sortino,
            "total_return": 0.18,
            "max_drawdown": -0.07,
            "n_trades": 80,
        },
        "parameter_stability": {},
        "n_windows": 6,
    }), encoding="utf-8")


def test_promote_strategy_passes_when_all_tickers_clear_avg_threshold(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_x"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"
    # Avg sortino = (0.9 + 0.8 + 0.7) / 3 = 0.8 > threshold 0.7.
    _seed_promotion_bundle(output_runs, "gen_x_promo_AAPL_wfo", 0.9)
    _seed_promotion_bundle(output_runs, "gen_x_promo_QQQ_wfo", 0.8)
    _seed_promotion_bundle(output_runs, "gen_x_promo_DIA_wfo", 0.7)

    fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("factory.promote.subprocess.run", return_value=fake_proc):
        result = promote_strategy(
            strategy_id="gen_x",
            optimized_params={"size": 1.0, "fast": 20},
            canonical_config_path=canonical,
            promotion_cfg=_cfg(min_avg_sortino=0.7),
            tmp_dir=tmp_path / "_tmp",
            output_runs_dir=output_runs,
            stage_timeout_sec=60,
        )

    assert result.ran is True
    assert result.passed is True
    assert result.avg_sortino == pytest.approx(0.8)
    assert set(result.per_ticker.keys()) == {"AAPL", "QQQ", "DIA"}
    assert result.per_ticker["AAPL"]["oos_sortino"] == pytest.approx(0.9)
    assert result.error is None


def test_promote_strategy_fails_when_avg_below_threshold(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_x"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"
    # Avg = (0.5 + 0.4 + 0.3) / 3 = 0.4 < 0.7.
    _seed_promotion_bundle(output_runs, "gen_x_promo_AAPL_wfo", 0.5)
    _seed_promotion_bundle(output_runs, "gen_x_promo_QQQ_wfo", 0.4)
    _seed_promotion_bundle(output_runs, "gen_x_promo_DIA_wfo", 0.3)

    fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("factory.promote.subprocess.run", return_value=fake_proc):
        result = promote_strategy(
            strategy_id="gen_x",
            optimized_params={"size": 1.0},
            canonical_config_path=canonical,
            promotion_cfg=_cfg(min_avg_sortino=0.7),
            tmp_dir=tmp_path / "_tmp",
            output_runs_dir=output_runs,
            stage_timeout_sec=60,
        )

    assert result.ran is True
    assert result.passed is False
    assert result.avg_sortino == pytest.approx(0.4)
    assert result.error is None  # No subprocess errors; just below threshold


def test_promote_strategy_continues_on_per_ticker_failure(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_x"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"
    # AAPL succeeds, QQQ fails, DIA succeeds. passed=False because not all tickers completed.
    _seed_promotion_bundle(output_runs, "gen_x_promo_AAPL_wfo", 0.9)
    # No QQQ bundle — its subprocess will exit 0 but find_latest_bundle will raise.
    _seed_promotion_bundle(output_runs, "gen_x_promo_DIA_wfo", 0.9)

    fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("factory.promote.subprocess.run", return_value=fake_proc):
        result = promote_strategy(
            strategy_id="gen_x",
            optimized_params={"size": 1.0},
            canonical_config_path=canonical,
            promotion_cfg=_cfg(min_avg_sortino=0.5),
            tmp_dir=tmp_path / "_tmp",
            output_runs_dir=output_runs,
            stage_timeout_sec=60,
        )

    assert result.ran is True
    # Only 2 of 3 tickers completed; avg is over the 2, but passed=False because
    # one ticker is missing entirely.
    assert set(result.per_ticker.keys()) == {"AAPL", "DIA"}
    assert result.avg_sortino == pytest.approx(0.9)
    assert result.passed is False
    assert result.error is not None
    assert "QQQ" in result.error


def test_promote_strategy_fails_when_subprocess_nonzero(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_x"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"

    fake_proc = mock.Mock(returncode=1, stdout="", stderr="ImportError: yfinance install failed")
    with mock.patch("factory.promote.subprocess.run", return_value=fake_proc):
        result = promote_strategy(
            strategy_id="gen_x",
            optimized_params={"size": 1.0},
            canonical_config_path=canonical,
            promotion_cfg=_cfg(tickers=("AAPL",)),
            tmp_dir=tmp_path / "_tmp",
            output_runs_dir=output_runs,
            stage_timeout_sec=60,
        )

    assert result.ran is True
    assert result.passed is False
    assert result.per_ticker == {}
    assert result.avg_sortino is None
    assert "AAPL" in result.error and "exit=1" in result.error


def _write_build_report(path: Path, classifications: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "min_hourly_bars": 7000,
        "symbols": {s: {"classification": c} for s, c in classifications.items()},
    }), encoding="utf-8")


def test_tradable_tickers_no_report_keeps_all() -> None:
    from factory.promote import _tradable_tickers
    assert _tradable_tickers(("AAPL", "QQQ", "DIA"), None) == ["AAPL", "QQQ", "DIA"]


def test_tradable_tickers_filters_insufficient(tmp_path: Path) -> None:
    from factory.promote import _tradable_tickers
    report = tmp_path / "_build_report.json"
    _write_build_report(report, {
        "AAPL": "tradable", "QQQ": "insufficient_history", "DIA": "tradable",
    })
    assert _tradable_tickers(("AAPL", "QQQ", "DIA"), report) == ["AAPL", "DIA"]


def test_promote_strategy_skips_insufficient_history_tickers(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_x"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"
    # Only AAPL and DIA are tradable; QQQ is insufficient and must be skipped.
    _seed_promotion_bundle(output_runs, "gen_x_promo_AAPL_wfo", 0.9)
    _seed_promotion_bundle(output_runs, "gen_x_promo_DIA_wfo", 0.8)
    report = tmp_path / "_build_report.json"
    _write_build_report(report, {
        "AAPL": "tradable", "QQQ": "insufficient_history", "DIA": "tradable",
    })

    fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("factory.promote.subprocess.run", return_value=fake_proc):
        result = promote_strategy(
            strategy_id="gen_x",
            optimized_params={"size": 1.0},
            canonical_config_path=canonical,
            promotion_cfg=_cfg(min_avg_sortino=0.7),
            tmp_dir=tmp_path / "_tmp",
            output_runs_dir=output_runs,
            stage_timeout_sec=60,
            build_report_path=report,
        )

    # QQQ never ran; passing is judged over the two eligible tickers only.
    assert set(result.per_ticker.keys()) == {"AAPL", "DIA"}
    assert result.avg_sortino == pytest.approx(0.85)
    assert result.passed is True
    assert result.error is None
