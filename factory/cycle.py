from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from factory.dedup import append_summary, read_tail
from factory.filesystem import (
    RegistryAlreadyHasStrategy,
    append_registry_entry,
    pick_unused_strategy_id,
    write_strategy_artifacts,
)
from dataclasses import asdict
from factory.generate import GenerationError, GenerationResult, call_claude
from factory.notify import NotifyConfig, extract_metric, maybe_send_alert
from factory.promote import PromotionResult, promote_strategy
from factory.prompt import build_prompt
from factory.results import build_failed_record, build_record, write_record
from factory.settings_loader import Settings
from factory.slots import pull_slots
from factory.stages import (
    StageError,
    StageResult,
    run_backtest_stage,
    run_optimize_stage,
    run_wfo_stage,
)
from factory.validate import (
    FunctionalValidationError,
    StaticValidationError,
    validate_functional,
    validate_static,
)

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class CycleOutcome:
    status: str                     # "complete" | "failed"
    failed_stage: Optional[str]
    strategy_id: Optional[str]
    record: dict[str, Any]


def _now_unix_int() -> int:
    return int(time.time())


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _notify_cfg(s: Settings) -> NotifyConfig:
    return NotifyConfig(
        alert_threshold_metric=s.alerts.alert_threshold_metric,
        alert_threshold=s.alerts.alert_threshold,
        telegram_bot_token=s.alerts.telegram_bot_token,
        telegram_chat_id=s.alerts.telegram_chat_id,
        dashboard_base_url=s.alerts.dashboard_base_url,
    )


def run_cycle(settings: Settings, *, rng: random.Random) -> CycleOutcome:
    """Execute one full cycle (§3 steps 1-17) and return the outcome.

    Never raises on expected failure modes; everything that goes wrong becomes
    a failed record (§3.1). The dedup-log append is at the FIRST possible
    moment after a parseable one_line_summary exists (§3.2).
    """
    s = settings
    paths = s.paths
    slots = pull_slots(rng)
    ts = _iso_now()
    base_strategy_id = f"gen_{s.node_id}_{_now_unix_int()}"
    # Step 1-3: slots + dedup tail + prompt.
    dedup_tail = read_tail(paths.dedup_log, n=30)
    strategy_id = pick_unused_strategy_id(base_strategy_id, strategies_dir=paths.strategies_dir)
    prompt = build_prompt(strategy_id=strategy_id, slots=slots, dedup_tail=dedup_tail)
    log.info("cycle start id=%s slots=%s", strategy_id, slots)

    # Step 4-5: generate + parse.
    try:
        gen: GenerationResult = call_claude(
            prompt=prompt,
            claude_cmd=s.generation.claude_cmd,
            claude_flags=s.generation.claude_flags,
            timeout_sec=s.generation.generation_timeout_sec,
        )
    except GenerationError as exc:
        rec = build_failed_record(
            strategy_id=None, timestamp=ts, slots=slots, idea=None,
            generation_cost_usd=0.0, failed_stage="generation", error=str(exc),
        )
        write_record(paths.results_dir, rec, node_id=s.node_id)
        log.warning("cycle id=%s generation failed: %s", strategy_id, exc)
        return CycleOutcome(status="failed", failed_stage="generation",
                            strategy_id=None, record=rec)

    parsed = gen.parsed
    cost = gen.cost_usd
    idea = {
        "one_line_summary": parsed["one_line_summary"],
        "hypothesis": parsed["hypothesis"],
        "novelty_justification": parsed["novelty_justification"],
        "failure_mode": parsed["failure_mode"],
        "allow_short": bool(parsed["allow_short"]),
    }

    # Step 6: dedup-log append (BEFORE validation, BEFORE stages — §3.2).
    append_summary(paths.dedup_log, parsed["one_line_summary"])

    # Step 7: validate (Tier 1 + Tier 2).
    try:
        validate_static(
            strategy_id=strategy_id,
            strategy_src=parsed["strategy_file"],
            config_src=parsed["config_file"],
            allow_short=bool(parsed["allow_short"]),
        )
        validate_functional(
            strategy_id=strategy_id,
            strategy_src=parsed["strategy_file"],
            allow_short=bool(parsed["allow_short"]),
            tmp_dir=paths.tmp_dir / "validate",
        )
    except (StaticValidationError, FunctionalValidationError) as exc:
        rec = build_failed_record(
            strategy_id=strategy_id, timestamp=ts, slots=slots, idea=idea,
            generation_cost_usd=cost, failed_stage="validation", error=str(exc),
        )
        write_record(paths.results_dir, rec, node_id=s.node_id)
        log.warning("cycle id=%s validation failed: %s", strategy_id, exc)
        return CycleOutcome(status="failed", failed_stage="validation",
                            strategy_id=strategy_id, record=rec)

    # Step 8-10: write files + register.
    write_strategy_artifacts(
        strategy_id=strategy_id,
        strategy_src=parsed["strategy_file"],
        config_src=parsed["config_file"],
        strategies_dir=paths.strategies_dir,
        configs_dir=paths.configs_dir,
    )
    try:
        append_registry_entry(strategy_id=strategy_id, registry_file=paths.registry_file)
    except RegistryAlreadyHasStrategy:
        log.info("registry already has %s; continuing", strategy_id)

    canonical_cfg = paths.configs_dir / f"{strategy_id}.yaml"

    # Step 11-13: run the three stages sequentially.
    bt: Optional[StageResult] = None
    opt: Optional[StageResult] = None
    wfo: Optional[StageResult] = None
    screened_out = False
    screen_reason: Optional[str] = None
    for stage_name, runner in (
        ("backtest", run_backtest_stage),
        ("optimize", run_optimize_stage),
        ("wfo", run_wfo_stage),
    ):
        if stage_name == "wfo" and s.screening.enabled:
            assert opt is not None
            best_score = opt.parsed.get("best_score")
            if best_score is not None and best_score < s.screening.min_optimize_score:
                screened_out = True
                screen_reason = (
                    f"optimize best_score {best_score:.3f} < floor "
                    f"{s.screening.min_optimize_score:.3f}"
                )
                log.info("cycle id=%s screened out before WFO: %s",
                         strategy_id, screen_reason)
                break
        try:
            result = runner(
                canonical_config=canonical_cfg,
                strategy_id=strategy_id,
                output_runs_dir=paths.output_runs_dir,
                tmp_dir=paths.tmp_dir,
                timeout_sec=s.stages.stage_timeout_sec,
                backtester_root=paths.backtester_root,
            )
        except StageError as exc:
            rec = build_failed_record(
                strategy_id=strategy_id, timestamp=ts, slots=slots, idea=idea,
                generation_cost_usd=cost, failed_stage=stage_name, error=str(exc),
                backtest=bt.parsed if bt else None,
                optimize=opt.parsed if opt else None,
            )
            write_record(paths.results_dir, rec, node_id=s.node_id)
            log.warning("cycle id=%s stage=%s failed: %s", strategy_id, stage_name, exc)
            return CycleOutcome(status="failed", failed_stage=stage_name,
                                strategy_id=strategy_id, record=rec)
        if stage_name == "backtest":
            bt = result
        elif stage_name == "optimize":
            opt = result
        else:
            wfo = result

    assert bt is not None and opt is not None

    # Step 13.5: held-out promotion (v0.3). Runs only if promotion is enabled
    # AND the WFO trigger metric clears the configured threshold. Failure
    # inside promote_strategy is captured into the promotion block; it never
    # raises into the cycle. The cycle's status stays "complete" regardless
    # of promotion outcome — the alert trigger is unchanged (still keyed on
    # the main WFO threshold).
    promotion_dict: Optional[dict[str, Any]] = None
    if s.promotion.enabled and wfo is not None:
        provisional = {
            "backtest": bt.parsed, "optimize": opt.parsed, "wfo": wfo.parsed,
        }
        trigger_value = extract_metric(provisional, s.promotion.trigger_metric)
        if trigger_value is not None and trigger_value >= s.promotion.trigger_threshold:
            log.info(
                "cycle id=%s WFO cleared promotion trigger (%s=%.3f >= %.3f); running held-out on %s",
                strategy_id, s.promotion.trigger_metric, trigger_value,
                s.promotion.trigger_threshold, list(s.promotion.tickers),
            )
            promo: PromotionResult = promote_strategy(
                strategy_id=strategy_id,
                optimized_params=opt.parsed["best_params"],
                canonical_config_path=canonical_cfg,
                promotion_cfg=s.promotion,
                tmp_dir=paths.tmp_dir,
                output_runs_dir=paths.output_runs_dir,
                stage_timeout_sec=s.stages.stage_timeout_sec,
                backtester_root=paths.backtester_root,
            )
            promotion_dict = asdict(promo)
            log.info(
                "cycle id=%s promotion passed=%s avg_sharpe=%s",
                strategy_id, promo.passed,
                f"{promo.avg_sharpe:.3f}" if promo.avg_sharpe is not None else "n/a",
            )

    # Step 14-15: build complete record.
    rec = build_record(
        strategy_id=strategy_id, timestamp=ts, slots=slots, idea=idea,
        generation_cost_usd=cost,
        backtest=bt.parsed, optimize=opt.parsed,
        wfo=wfo.parsed if wfo is not None else None,
        promotion=promotion_dict,
        screened_out=screened_out,
        screen_reason=screen_reason,
        alerted=False,  # patched below after maybe_send_alert
    )

    # Step 16: alert (conditional). maybe_send_alert never raises.
    # NOTE: alert trigger is unchanged (still wfo.oos_sharpe by default).
    # Promotion is informational on the dashboard, not a gate on alerts.
    notify_result = maybe_send_alert(rec, _notify_cfg(s))
    rec["alerted"] = bool(notify_result.sent)

    write_record(paths.results_dir, rec, node_id=s.node_id)
    log.info("cycle id=%s complete oos_sharpe=%s screened=%s alerted=%s",
             strategy_id,
             wfo.parsed.get("oos_sharpe") if wfo is not None else "n/a",
             screened_out, rec["alerted"])
    return CycleOutcome(status="complete", failed_stage=None,
                        strategy_id=strategy_id, record=rec)
