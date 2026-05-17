"""Held-out promotion stage: re-run a shortlisted strategy on alternate tickers.

Triggered after WFO succeeds and clears settings.promotion.trigger_threshold.
For each held-out ticker, build a promotion-specific YAML (canonical config
cloned with data.symbols + data.source + strategy_params swapped, run_name
suffixed) and run a full WFO via subprocess. Aggregate OOS Sortino across the
panel; gate against min_avg_sortino.

Promotion failures do NOT fail the cycle (the cycle's status stays
"complete"). The promotion result is informational on the dashboard; the
alert trigger is unchanged (still keyed on the main WFO threshold).
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from factory.settings_loader import PromotionCfg
from factory.stages import (
    BundleNotFound,
    StageError,
    find_latest_bundle,
    parse_wfo_summary,
)

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PromotionResult:
    ran: bool
    tickers: tuple[str, ...]
    per_ticker: dict[str, dict[str, Any]]
    avg_sortino: Optional[float]
    min_avg_sortino_threshold: float
    passed: bool
    error: Optional[str] = None


def _build_promotion_config(
    *,
    canonical_path: Path,
    strategy_id: str,
    ticker: str,
    optimized_params: Mapping[str, Any],
    data_source: str,
    tmp_dir: Path,
) -> tuple[Path, str]:
    """Clone the canonical YAML for one held-out ticker.

    Returns (config_path, run_name).
    """
    raw = yaml.safe_load(canonical_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise StageError(f"canonical config is not a mapping: {canonical_path}")
    raw["data"] = dict(raw.get("data", {}))
    raw["data"]["symbols"] = [ticker]
    raw["data"]["source"] = data_source
    raw["strategy_params"] = dict(optimized_params)
    run_name = f"{strategy_id}_promo_{ticker}_wfo"
    raw["run_name"] = run_name
    out_dir = tmp_dir / strategy_id / "promote"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.yaml"
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out_path, run_name


def _run_promotion_wfo(
    *,
    cfg_path: Path,
    run_name: str,
    output_runs_dir: Path,
    timeout_sec: int,
    backtester_root: Optional[Path],
) -> dict[str, Any]:
    """Run a single held-out WFO subprocess and parse its summary.json."""
    cmd = [sys.executable, "-m", "backtester.runners.run_wfo", "--config", str(cfg_path)]
    log.info("promotion wfo: run_name=%s cmd=%s", run_name, cmd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            cwd=str(backtester_root) if backtester_root else None,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as exc:
        raise StageError(f"promotion wfo {run_name} timed out after {timeout_sec}s") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-1500:]
        raise StageError(
            f"promotion wfo {run_name} exit={proc.returncode}; stderr tail:\n{tail}"
        )
    try:
        bundle = find_latest_bundle(output_runs_dir=output_runs_dir, run_name=run_name)
    except BundleNotFound as exc:
        raise StageError(f"promotion wfo {run_name}: {exc}") from exc
    summary_path = bundle / "summary.json"
    if not summary_path.exists():
        raise StageError(f"promotion wfo {run_name}: summary.json missing in {bundle}")
    try:
        raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageError(f"promotion wfo {run_name}: summary.json not valid JSON: {exc}") from exc
    try:
        return parse_wfo_summary(raw_summary, bundle_path=bundle)
    except KeyError as exc:
        raise StageError(f"promotion wfo {run_name}: summary missing key: {exc}") from exc


def _tradable_tickers(
    tickers: Sequence[str], build_report_path: Optional[Path],
) -> list[str]:
    """Filter promotion tickers to those classified `tradable` in the hourly
    build report.

    When no report path is given, or the file is absent (the factory is still
    on daily data), every ticker passes through unchanged. When the report is
    present, a ticker is kept only if its classification is exactly
    `tradable` — a ticker that is missing or `insufficient_history` is dropped
    so promotion never runs a WFO on thin hourly data.
    """
    if build_report_path is None or not Path(build_report_path).exists():
        return list(tickers)
    try:
        report = json.loads(Path(build_report_path).read_text(encoding="utf-8"))
        symbols = report.get("symbols", {})
    except (json.JSONDecodeError, OSError):
        return list(tickers)
    return [
        t for t in tickers
        if symbols.get(t, {}).get("classification") == "tradable"
    ]


def promote_strategy(
    *,
    strategy_id: str,
    optimized_params: Mapping[str, Any],
    canonical_config_path: Path,
    promotion_cfg: PromotionCfg,
    tmp_dir: Path,
    output_runs_dir: Path,
    stage_timeout_sec: int,
    backtester_root: Optional[Path] = None,
    build_report_path: Optional[Path] = None,
) -> PromotionResult:
    """Run the strategy on each held-out ticker with the SPY-optimized params.

    Continues even if individual tickers fail (captures partial per_ticker
    data). passed=True only if ALL tickers succeed AND avg oos_sortino clears
    promotion_cfg.min_avg_sortino.
    """
    eligible = _tradable_tickers(promotion_cfg.tickers, build_report_path)
    skipped = [t for t in promotion_cfg.tickers if t not in eligible]
    if skipped:
        log.info(
            "promotion %s skipping insufficient-history tickers: %s",
            strategy_id, ", ".join(skipped),
        )
    per_ticker: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for ticker in eligible:
        try:
            cfg_path, run_name = _build_promotion_config(
                canonical_path=canonical_config_path,
                strategy_id=strategy_id,
                ticker=ticker,
                optimized_params=optimized_params,
                data_source=promotion_cfg.data_source,
                tmp_dir=tmp_dir,
            )
            parsed = _run_promotion_wfo(
                cfg_path=cfg_path,
                run_name=run_name,
                output_runs_dir=output_runs_dir,
                timeout_sec=stage_timeout_sec,
                backtester_root=backtester_root,
            )
            per_ticker[ticker] = parsed
            log.info(
                "promotion %s ticker=%s oos_sortino=%.3f",
                strategy_id, ticker, parsed.get("oos_sortino", 0.0),
            )
        except StageError as exc:
            errors.append(f"{ticker}: {exc}")
            log.warning("promotion %s ticker=%s failed: %s", strategy_id, ticker, exc)

    if per_ticker:
        sortinos = [float(p["oos_sortino"]) for p in per_ticker.values()]
        avg = sum(sortinos) / len(sortinos)
    else:
        avg = None

    all_tickers_completed = len(per_ticker) == len(eligible)
    passed = (
        all_tickers_completed
        and avg is not None
        and avg >= promotion_cfg.min_avg_sortino
    )
    error = "; ".join(errors) if errors else None
    return PromotionResult(
        ran=True,
        tickers=tuple(promotion_cfg.tickers),
        per_ticker=per_ticker,
        avg_sortino=avg,
        min_avg_sortino_threshold=promotion_cfg.min_avg_sortino,
        passed=passed,
        error=error,
    )
