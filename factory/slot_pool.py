from __future__ import annotations

import hashlib
import itertools
import json
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

from factory.slots import SLOT_NAMES, SLOTS, _INCOMPATIBLE


POOL_SCHEMA_VERSION = 1
DEFAULT_STALE_CLAIM_SEC = 24 * 60 * 60


@dataclass(slots=True, frozen=True)
class SlotClaim:
    """A claimed slot combination from the precomputed SQLite pool."""

    combination_id: str
    slots: dict[str, str]
    claimed_by: str
    claimed_at: int


class SlotPoolExhausted(RuntimeError):
    """Raised when no unused or stale slot combinations remain."""


def _canonical_slots(slots: Mapping[str, str]) -> dict[str, str]:
    return {name: str(slots[name]) for name in SLOT_NAMES}


def combination_id(slots: Mapping[str, str]) -> str:
    """Stable ID for a slot tuple across machines and pool rebuilds."""
    canonical = _canonical_slots(slots)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def slot_table_signature(
    slots_table: Mapping[str, Iterable[str]] = SLOTS,
    incompatible: Iterable[tuple[str, str]] = _INCOMPATIBLE,
) -> str:
    """Hash of current slot definitions that affect generated combinations."""
    payload = {
        "slot_names": list(SLOT_NAMES),
        "slots": {
            name: list(dict.fromkeys(slots_table[name]))
            for name in SLOT_NAMES
        },
        "incompatible": sorted(list(pair) for pair in incompatible),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def iter_valid_combinations(
    slots_table: Mapping[str, Iterable[str]] = SLOTS,
    incompatible: Iterable[tuple[str, str]] = _INCOMPATIBLE,
) -> list[dict[str, str]]:
    """Return every valid slot combination from SLOTS.

    Duplicate values inside a slot, such as the weighted long-only direction,
    are intentionally de-duplicated for the database. Weighting belongs to
    random draws; a finite exhaustion pool should represent unique tuples.
    """
    values_by_slot = [
        tuple(dict.fromkeys(slots_table[name]))
        for name in SLOT_NAMES
    ]
    incompatible_set = frozenset(incompatible)
    out: list[dict[str, str]] = []
    for values in itertools.product(*values_by_slot):
        slots = dict(zip(SLOT_NAMES, values))
        if (slots["constraint_twist"], slots["exit_rule"]) in incompatible_set:
            continue
        out.append(slots)
    return out


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS slot_pool_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS slot_combinations (
            combination_id TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('unused', 'claimed', 'consumed')),
            strategy_family TEXT NOT NULL,
            signal_primitive TEXT NOT NULL,
            holding_horizon TEXT NOT NULL,
            direction TEXT NOT NULL,
            exit_rule TEXT NOT NULL,
            constraint_twist TEXT NOT NULL,
            inspiration_anchor TEXT NOT NULL,
            claimed_by TEXT,
            claimed_at INTEGER,
            consumed_at INTEGER,
            strategy_id TEXT,
            outcome_status TEXT,
            failed_stage TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_slot_pool_status ON slot_combinations(status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_slot_pool_status_id "
        "ON slot_combinations(status, combination_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_slot_pool_claimed_at "
        "ON slot_combinations(status, claimed_at)"
    )
    conn.execute(
        """
        INSERT INTO slot_pool_meta(key, value)
        VALUES ('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(POOL_SCHEMA_VERSION),),
    )


def initialize_slot_pool(
    db_path: Path,
    *,
    slots_table: Mapping[str, Iterable[str]] = SLOTS,
    incompatible: Iterable[tuple[str, str]] = _INCOMPATIBLE,
) -> int:
    """Create/update the pre-populated DB and return row count.

    Existing rows keep their status. New combinations introduced by future slot
    edits are inserted as unused; removed combinations are left in place as
    history so consumed/claimed audit data is not destroyed.
    """
    now = int(time.time())
    current_signature = slot_table_signature(slots_table, incompatible)
    with _connect(db_path) as conn:
        _create_schema(conn)
        existing_signature = conn.execute(
            "SELECT value FROM slot_pool_meta WHERE key = 'slot_table_signature'"
        ).fetchone()
        if (
            existing_signature is not None
            and existing_signature["value"] == current_signature
        ):
            row = conn.execute("SELECT COUNT(*) AS n FROM slot_combinations").fetchone()
            return int(row["n"])

        combinations = iter_valid_combinations(slots_table, incompatible)
        conn.executemany(
            """
            INSERT INTO slot_combinations (
                combination_id, status, strategy_family, signal_primitive,
                holding_horizon, direction, exit_rule, constraint_twist,
                inspiration_anchor
            )
            VALUES (?, 'unused', ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(combination_id) DO NOTHING
            """,
            [
                (
                    combination_id(slots),
                    slots["strategy_family"],
                    slots["signal_primitive"],
                    slots["holding_horizon"],
                    slots["direction"],
                    slots["exit_rule"],
                    slots["constraint_twist"],
                    slots["inspiration_anchor"],
                )
                for slots in combinations
            ],
        )
        conn.execute(
            """
            INSERT INTO slot_pool_meta(key, value)
            VALUES ('slot_table_signature', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (current_signature,),
        )
        conn.execute(
            """
            INSERT INTO slot_pool_meta(key, value)
            VALUES ('last_initialized_at', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(now),),
        )
        row = conn.execute("SELECT COUNT(*) AS n FROM slot_combinations").fetchone()
        return int(row["n"])


def _row_to_claim(row: sqlite3.Row) -> SlotClaim:
    slots = {name: str(row[name]) for name in SLOT_NAMES}
    return SlotClaim(
        combination_id=str(row["combination_id"]),
        slots=slots,
        claimed_by=str(row["claimed_by"]),
        claimed_at=int(row["claimed_at"]),
    )


def claim_slot_combination(
    db_path: Path,
    *,
    rng: random.Random,
    node_id: str,
    stale_after_sec: int = DEFAULT_STALE_CLAIM_SEC,
    slots_table: Mapping[str, Iterable[str]] = SLOTS,
    incompatible: Iterable[tuple[str, str]] = _INCOMPATIBLE,
) -> SlotClaim:
    """Atomically claim one unused or stale precomputed slot tuple."""
    initialize_slot_pool(db_path, slots_table=slots_table, incompatible=incompatible)
    now = int(time.time())
    stale_before = now - stale_after_sec

    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        stale_rows = conn.execute(
            """
            SELECT combination_id
            FROM slot_combinations
            WHERE status = 'claimed'
              AND claimed_at IS NOT NULL
              AND claimed_at < ?
            """,
            (stale_before,),
        ).fetchall()
        if stale_rows:
            stale_ids = [str(row["combination_id"]) for row in stale_rows]
            conn.executemany(
                """
                UPDATE slot_combinations
                SET status = 'unused',
                    claimed_by = NULL,
                    claimed_at = NULL,
                    consumed_at = NULL,
                    strategy_id = NULL,
                    outcome_status = NULL,
                    failed_stage = NULL
                WHERE combination_id = ?
                  AND status = 'claimed'
                """,
                [(cid,) for cid in stale_ids],
            )

        count_row = conn.execute(
            "SELECT COUNT(*) AS n FROM slot_combinations WHERE status = 'unused'"
        ).fetchone()
        unused_count = int(count_row["n"])
        if unused_count <= 0:
            raise SlotPoolExhausted("no unused slot combinations remain")

        offset = rng.randrange(unused_count)
        selected = conn.execute(
            """
            SELECT combination_id
            FROM slot_combinations
            WHERE status = 'unused'
            ORDER BY combination_id
            LIMIT 1 OFFSET ?
            """,
            (offset,),
        ).fetchone()
        if selected is None:
            raise SlotPoolExhausted("no unused slot combinations remain")

        selected_id = str(selected["combination_id"])
        conn.execute(
            """
            UPDATE slot_combinations
            SET status = 'claimed',
                claimed_by = ?,
                claimed_at = ?,
                consumed_at = NULL,
                strategy_id = NULL,
                outcome_status = NULL,
                failed_stage = NULL
            WHERE combination_id = ?
              AND status = 'unused'
            """,
            (node_id, now, selected_id),
        )
        row = conn.execute(
            """
            SELECT *
            FROM slot_combinations
            WHERE combination_id = ?
            """,
            (selected_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"claimed slot combination disappeared: {selected_id}")
        return _row_to_claim(row)


def consume_slot_combination(
    db_path: Path,
    *,
    combination_id: str,
    claimed_by: str,
    claimed_at: int,
    strategy_id: Optional[str],
    outcome_status: str,
    failed_stage: Optional[str],
) -> None:
    """Mark a claimed combination consumed after a cycle attempt finishes."""
    now = int(time.time())
    with _connect(db_path) as conn:
        _create_schema(conn)
        cur = conn.execute(
            """
            UPDATE slot_combinations
            SET status = 'consumed',
                consumed_at = ?,
                strategy_id = ?,
                outcome_status = ?,
                failed_stage = ?
            WHERE combination_id = ?
              AND status = 'claimed'
              AND claimed_by = ?
              AND claimed_at = ?
            """,
            (
                now,
                strategy_id,
                outcome_status,
                failed_stage,
                combination_id,
                claimed_by,
                claimed_at,
            ),
        )
        if cur.rowcount != 1:
            raise RuntimeError(
                "slot combination could not be consumed from the active claim: "
                f"{combination_id}"
            )


def slot_pool_counts(
    db_path: Path,
    *,
    slots_table: Mapping[str, Iterable[str]] = SLOTS,
    incompatible: Iterable[tuple[str, str]] = _INCOMPATIBLE,
) -> dict[str, int]:
    """Return counts by status for diagnostics/tests."""
    initialize_slot_pool(db_path, slots_table=slots_table, incompatible=incompatible)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM slot_combinations GROUP BY status"
        ).fetchall()
    counts = {"unused": 0, "claimed": 0, "consumed": 0}
    counts.update({str(row["status"]): int(row["n"]) for row in rows})
    return counts
