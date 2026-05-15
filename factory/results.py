from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

log = logging.getLogger(__name__)

Record = dict[str, Any]

FAILED_STAGES = frozenset({"generation", "validation", "backtest", "optimize", "wfo"})


def build_record(
    *,
    strategy_id: str,
    timestamp: str,
    slots: Mapping[str, str],
    idea: Mapping[str, Any],
    generation_cost_usd: float,
    backtest: Optional[Mapping[str, Any]],
    optimize: Optional[Mapping[str, Any]],
    wfo: Optional[Mapping[str, Any]],
    alerted: bool,
    promotion: Optional[Mapping[str, Any]] = None,
) -> Record:
    """Build a `status: complete` results record (§6, reconciled).

    `promotion` is an optional v0.3 held-out-validation block. None means the
    promotion stage did not run (either disabled in settings or the WFO
    trigger threshold was not cleared).
    """
    return {
        "strategy_id": strategy_id,
        "timestamp": timestamp,
        "status": "complete",
        "failed_stage": None,
        "error": None,
        "slots": dict(slots),
        "idea": dict(idea),
        "generation_cost_usd": float(generation_cost_usd),
        "backtest": dict(backtest) if backtest is not None else None,
        "optimize": dict(optimize) if optimize is not None else None,
        "wfo": dict(wfo) if wfo is not None else None,
        "promotion": dict(promotion) if promotion is not None else None,
        "alerted": bool(alerted),
    }


def build_failed_record(
    *,
    strategy_id: Optional[str],
    timestamp: str,
    slots: Mapping[str, str],
    idea: Optional[Mapping[str, Any]],
    generation_cost_usd: float,
    failed_stage: str,
    error: str,
    backtest: Optional[Mapping[str, Any]] = None,
    optimize: Optional[Mapping[str, Any]] = None,
    wfo: Optional[Mapping[str, Any]] = None,
    promotion: Optional[Mapping[str, Any]] = None,
) -> Record:
    """Build a `status: failed` results record (§3.1)."""
    if failed_stage not in FAILED_STAGES:
        raise ValueError(f"failed_stage must be one of {sorted(FAILED_STAGES)}, got {failed_stage!r}")
    return {
        "strategy_id": strategy_id,
        "timestamp": timestamp,
        "status": "failed",
        "failed_stage": failed_stage,
        "error": error,
        "slots": dict(slots),
        "idea": dict(idea) if idea is not None else None,
        "generation_cost_usd": float(generation_cost_usd),
        "backtest": dict(backtest) if backtest is not None else None,
        "optimize": dict(optimize) if optimize is not None else None,
        "wfo": dict(wfo) if wfo is not None else None,
        "promotion": dict(promotion) if promotion is not None else None,
        "alerted": False,
    }


def write_record(store_path: Path, record: Record) -> None:
    """Append one JSON object as a single line to the JSONL store."""
    store_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with store_path.open("a", encoding="utf-8") as f:
        f.write(line)


def read_records(store_path: Path) -> list[Record]:
    """Read all records from the JSONL store. Skips blank lines.

    Raises ValueError on any non-blank line that isn't valid JSON (corruption).
    """
    if not store_path.exists():
        return []
    out: list[Record] = []
    for i, raw in enumerate(store_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"results store corruption at line {i}: {exc}") from exc
    return out
