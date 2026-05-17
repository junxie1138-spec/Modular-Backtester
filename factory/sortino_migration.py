"""Migrate this machine's archived strategy records onto the OOS-Sortino system.

The factory's alert and promotion metrics moved from Sharpe to OOS Sortino,
but records produced before that change carry no `oos_sortino`. This module
backfills it — by reading each record's existing WFO bundle, never by
recomputing — flags records whose promote/no-promote verdict would flip under
the new metric, and queues retroactive promotion for records that now clear
the threshold.

Every write touches only this machine's own shard, `results/<node_id>.jsonl`,
so the distributed factory's sole-writer-per-shard invariant — and therefore
conflict-free git sync — is preserved.

Two public entry points, both called from factory/loop.py:
  - migrate_shard(settings)            — one idempotent pass, at startup.
  - drain_one_retro_promotion(settings) — at most one retro-promotion, per cycle.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from factory.promote import promote_strategy
from factory.results import Record
from factory.settings_loader import PromotionCfg, Settings

log = logging.getLogger(__name__)


def _needs_rerun(
    oos_sharpe: float, oos_sortino: float, trigger_threshold: float,
) -> bool:
    """True when Sharpe and Sortino fall on opposite sides of the promotion
    trigger threshold — i.e. the Sharpe->Sortino swap flips this strategy's
    promote/no-promote standing, so a re-optimisation on sortino might change
    the verdict.
    """
    return (oos_sharpe >= trigger_threshold) != (oos_sortino >= trigger_threshold)


def _initial_state(
    *,
    has_promotion_block: bool,
    oos_sortino: float,
    promotion_enabled: bool,
    trigger_threshold: float,
) -> str:
    """The `sortino_migration.state` a record receives at first migration.

    done    — the record already has a promotion block; it is past the
              retro-promotion stage, nothing to queue.
    pending — no promotion block, promotion is enabled, and oos_sortino
              clears the trigger threshold: eligible for retro-promotion.
    n/a     — not eligible: promotion disabled, or below the threshold.
    """
    if has_promotion_block:
        return "done"
    if promotion_enabled and oos_sortino >= trigger_threshold:
        return "pending"
    return "n/a"
