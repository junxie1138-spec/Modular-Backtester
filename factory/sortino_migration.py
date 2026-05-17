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


def _read_bundle_sortino(bundle_path: str | Path) -> float | None:
    """Read `oos_summary.sortino` from a WFO bundle's summary.json.

    Returns None when the bundle directory, its summary.json, or the
    oos_summary.sortino value is missing or unreadable — the caller then
    leaves the record untouched (no recompute).
    """
    summary = Path(bundle_path) / "summary.json"
    if not summary.is_file():
        return None
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    oos = data.get("oos_summary")
    if not isinstance(oos, dict):
        return None
    sortino = oos.get("sortino")
    if sortino is None:
        return None
    try:
        return float(sortino)
    except (TypeError, ValueError):
        return None


def _migrate_record(record: Record, *, promotion_cfg: PromotionCfg) -> Record | None:
    """Return a migrated copy of `record`, or None to leave it untouched.

    None is returned for records that need no migration (not `complete`, no
    `wfo` block, or already carrying `wfo.oos_sortino`) and for records whose
    WFO-bundle sortino cannot be recovered.
    """
    if record.get("status") != "complete":
        return None
    wfo = record.get("wfo")
    if not isinstance(wfo, dict):
        return None
    if "oos_sortino" in wfo:
        return None  # already migrated, or natively sortino — idempotency key

    bundle_path = wfo.get("run_bundle_path")
    sortino = _read_bundle_sortino(bundle_path) if bundle_path else None
    if sortino is None:
        log.warning(
            "sortino migration: cannot recover oos_sortino for %s "
            "(bundle missing or incomplete: %s); leaving record untouched",
            record.get("strategy_id"), bundle_path,
        )
        return None

    oos_sharpe = wfo.get("oos_sharpe")
    needs_rerun = (
        _needs_rerun(float(oos_sharpe), sortino, promotion_cfg.trigger_threshold)
        if isinstance(oos_sharpe, (int, float)) and not isinstance(oos_sharpe, bool)
        else False
    )
    state = _initial_state(
        has_promotion_block=record.get("promotion") is not None,
        oos_sortino=sortino,
        promotion_enabled=promotion_cfg.enabled,
        trigger_threshold=promotion_cfg.trigger_threshold,
    )
    migrated: Record = dict(record)
    migrated["wfo"] = {**wfo, "oos_sortino": sortino}
    migrated["sortino_migration"] = {
        "migrated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "needs_rerun": needs_rerun,
        "state": state,
    }
    return migrated


def _read_shard(shard: Path) -> list[Record]:
    """Read one NDJSON shard into a list of records. [] if the shard is absent."""
    if not shard.is_file():
        return []
    out: list[Record] = []
    for line in shard.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _write_shard(shard: Path, records: list[Record]) -> None:
    """Rewrite a shard from a record list, matching results.write_record's
    line format (compact separators, ensure_ascii=False, trailing newline)."""
    shard.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(r, ensure_ascii=False, separators=(",", ":"))
        for r in records
    ]
    shard.write_text("\n".join(lines) + "\n", encoding="utf-8")


def migrate_shard(settings: Settings) -> int:
    """Migrate this machine's own results shard in place. Idempotent.

    Returns the number of records migrated on this pass (0 on a no-op pass).
    Runs regardless of sync mode — a standalone machine has a shard too.
    Only ever reads and rewrites `results/<node_id>.jsonl`, so the
    sole-writer-per-shard sync invariant is preserved.
    """
    shard = settings.paths.results_dir / f"{settings.node_id}.jsonl"
    records = _read_shard(shard)
    if not records:
        return 0
    migrated_count = 0
    for i, record in enumerate(records):
        migrated = _migrate_record(record, promotion_cfg=settings.promotion)
        if migrated is not None:
            records[i] = migrated
            migrated_count += 1
    if migrated_count:
        _write_shard(shard, records)
        log.info(
            "sortino migration: migrated %d record(s) in %s",
            migrated_count, shard.name,
        )
    return migrated_count
