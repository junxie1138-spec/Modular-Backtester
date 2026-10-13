import random
import sqlite3
import time
from pathlib import Path

import pytest

from factory.slot_pool import (
    SlotPoolExhausted,
    claim_slot_combination,
    combination_id,
    consume_slot_combination,
    initialize_slot_pool,
    iter_valid_combinations,
    release_slot_combination,
    slot_pool_counts,
)
from factory.slot_pool import CORE_SLOT_NAMES
from factory.slots import SLOTS, _INCOMPATIBLE


SMALL_SLOTS = {
    "strategy_family": ("momentum", "breakout"),
    "signal_primitive": ("close-to-close returns",),
    "holding_horizon": ("1-2 days",),
    "direction": ("long-only", "long-only", "long/short"),
    "exit_rule": ("fixed-bar exit", "signal-reversal exit"),
    "constraint_twist": ("simple", "symmetric entry/exit rule"),
    "inspiration_anchor": ("hysteresis control",),
}
SMALL_INCOMPATIBLE = frozenset({
    ("symmetric entry/exit rule", "fixed-bar exit"),
})


def test_iter_valid_combinations_excludes_incompatible_pairs() -> None:
    combos = iter_valid_combinations()
    assert combos
    for slots in combos:
        assert set(slots) == set(CORE_SLOT_NAMES)
        assert (slots["constraint_twist"], slots["exit_rule"]) not in _INCOMPATIBLE


def test_iter_valid_combinations_deduplicates_weighted_direction() -> None:
    unique_count = 1
    for name in CORE_SLOT_NAMES:
        unique_count *= len(tuple(dict.fromkeys(SLOTS[name])))
    expected = unique_count - (
        len(SLOTS["strategy_family"])
        * len(SLOTS["signal_primitive"])
        * len(SLOTS["holding_horizon"])
        * len(tuple(dict.fromkeys(SLOTS["direction"])))
        * len(_INCOMPATIBLE)
    )
    assert len(iter_valid_combinations()) == expected
    assert expected == 71_680


def test_combination_id_is_deterministic() -> None:
    slots = iter_valid_combinations(SMALL_SLOTS, SMALL_INCOMPATIBLE)[0]
    assert combination_id(slots) == combination_id(dict(reversed(list(slots.items()))))
    assert combination_id(slots) == combination_id({**slots, "inspiration_anchor": "another anchor"})


def test_initialize_populates_sqlite_pool(tmp_path: Path) -> None:
    db = tmp_path / "slot_pool.sqlite"
    n = initialize_slot_pool(
        db,
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    assert n == len(iter_valid_combinations(SMALL_SLOTS, SMALL_INCOMPATIBLE))
    assert slot_pool_counts(
        db,
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    ) == {"unused": n, "claimed": 0, "consumed": 0}
    assert n == len(iter_valid_combinations(
        {**SMALL_SLOTS, "inspiration_anchor": ("first", "second")}, SMALL_INCOMPATIBLE,
    ))


def test_claim_and_consume_slot_combination(tmp_path: Path) -> None:
    db = tmp_path / "slot_pool.sqlite"
    claim = claim_slot_combination(
        db,
        rng=random.Random(0),
        node_id="local",
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    counts = slot_pool_counts(
        db,
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    assert counts["claimed"] == 1
    assert claim.slots["inspiration_anchor"] == "hysteresis control"
    assert counts["unused"] == (
        len(iter_valid_combinations(SMALL_SLOTS, SMALL_INCOMPATIBLE)) - 1
    )

    consume_slot_combination(
        db,
        combination_id=claim.combination_id,
        claimed_by=claim.claimed_by,
        claimed_at=claim.claimed_at,
        strategy_id="gen_local_1",
        outcome_status="failed",
        failed_stage="generation",
    )
    counts = slot_pool_counts(
        db,
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    assert counts["claimed"] == 0
    assert counts["consumed"] == 1

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            """
            SELECT status, strategy_id, outcome_status, failed_stage
            FROM slot_combinations
            WHERE combination_id = ?
            """,
            (claim.combination_id,),
        ).fetchone()
    assert row == ("consumed", "gen_local_1", "failed", "generation")


def test_release_returns_unfinished_combination_to_pool(tmp_path: Path) -> None:
    db = tmp_path / "slot_pool.sqlite"
    claim = claim_slot_combination(
        db, rng=random.Random(0), node_id="local",
        slots_table=SMALL_SLOTS, incompatible=SMALL_INCOMPATIBLE,
    )
    release_slot_combination(
        db, combination_id=claim.combination_id,
        claimed_by=claim.claimed_by, claimed_at=claim.claimed_at,
    )
    assert slot_pool_counts(db, slots_table=SMALL_SLOTS, incompatible=SMALL_INCOMPATIBLE)["unused"] == len(
        iter_valid_combinations(SMALL_SLOTS, SMALL_INCOMPATIBLE)
    )


def test_legacy_pool_keeps_consumed_core_and_drops_unused_anchors(tmp_path: Path) -> None:
    db = tmp_path / "slot_pool.sqlite"
    initialize_slot_pool(db, slots_table=SMALL_SLOTS, incompatible=SMALL_INCOMPATIBLE)
    core = iter_valid_combinations(SMALL_SLOTS, SMALL_INCOMPATIBLE)[0]
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM slot_combinations")
        for status, anchor in (("consumed", "old anchor"), ("unused", "other anchor")):
            legacy = {**core, "inspiration_anchor": anchor}
            conn.execute(
                "INSERT INTO slot_combinations (combination_id, status, strategy_family, "
                "signal_primitive, holding_horizon, direction, exit_rule, constraint_twist, "
                "inspiration_anchor) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(status) + anchor, status, *(core[name] for name in CORE_SLOT_NAMES), anchor),
            )
        conn.execute("UPDATE slot_pool_meta SET value = 'old' WHERE key = 'slot_table_signature'")
    assert initialize_slot_pool(db, slots_table=SMALL_SLOTS, incompatible=SMALL_INCOMPATIBLE) == len(
        iter_valid_combinations(SMALL_SLOTS, SMALL_INCOMPATIBLE)
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT status FROM slot_combinations WHERE combination_id = ?", (combination_id(core),)).fetchone() == ("consumed",)
        assert conn.execute("SELECT COUNT(*) FROM slot_combinations WHERE inspiration_anchor != '' AND status = 'unused'").fetchone() == (0,)


def test_legacy_generation_failure_does_not_consume_core(tmp_path: Path) -> None:
    db = tmp_path / "slot_pool.sqlite"
    initialize_slot_pool(db, slots_table=SMALL_SLOTS, incompatible=SMALL_INCOMPATIBLE)
    core = iter_valid_combinations(SMALL_SLOTS, SMALL_INCOMPATIBLE)[0]
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM slot_combinations WHERE combination_id = ?", (combination_id(core),))
        conn.execute(
            "INSERT INTO slot_combinations (combination_id, status, strategy_family, "
            "signal_primitive, holding_horizon, direction, exit_rule, constraint_twist, "
            "inspiration_anchor, failed_stage) VALUES (?, 'consumed', ?, ?, ?, ?, ?, ?, ?, 'generation')",
            ('old-failed-id', *(core[name] for name in CORE_SLOT_NAMES), 'old anchor'),
        )
        conn.execute("UPDATE slot_pool_meta SET value = 'old' WHERE key = 'slot_table_signature'")
    initialize_slot_pool(db, slots_table=SMALL_SLOTS, incompatible=SMALL_INCOMPATIBLE)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT status FROM slot_combinations WHERE combination_id = ?", (combination_id(core),)).fetchone() == ("unused",)


def test_initialize_preserves_consumed_rows(tmp_path: Path) -> None:
    db = tmp_path / "slot_pool.sqlite"
    claim = claim_slot_combination(
        db,
        rng=random.Random(0),
        node_id="local",
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    consume_slot_combination(
        db,
        combination_id=claim.combination_id,
        claimed_by=claim.claimed_by,
        claimed_at=claim.claimed_at,
        strategy_id="gen_local_1",
        outcome_status="complete",
        failed_stage=None,
    )
    initialize_slot_pool(
        db,
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    counts = slot_pool_counts(
        db,
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    assert counts["consumed"] == 1
    assert counts["claimed"] == 0


def test_stale_claim_is_reclaimed(tmp_path: Path) -> None:
    db = tmp_path / "slot_pool.sqlite"
    first = claim_slot_combination(
        db,
        rng=random.Random(0),
        node_id="local",
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    old = int(time.time()) - 10
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE slot_combinations
            SET claimed_at = ?
            WHERE combination_id = ?
            """,
            (old, first.combination_id),
        )
    second = claim_slot_combination(
        db,
        rng=random.Random(0),
        node_id="local",
        stale_after_sec=1,
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    assert second.combination_id == first.combination_id
    assert second.claimed_at != first.claimed_at or second.claimed_by == first.claimed_by


def test_consume_requires_active_claim(tmp_path: Path) -> None:
    db = tmp_path / "slot_pool.sqlite"
    initialize_slot_pool(
        db,
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    row = iter_valid_combinations(SMALL_SLOTS, SMALL_INCOMPATIBLE)[0]
    with pytest.raises(RuntimeError, match="active claim"):
        consume_slot_combination(
            db,
            combination_id=combination_id(row),
            claimed_by="local",
            claimed_at=int(time.time()),
            strategy_id="gen_local_1",
            outcome_status="failed",
            failed_stage="generation",
        )


def test_stale_original_claim_cannot_consume_reclaimed_slot(tmp_path: Path) -> None:
    db = tmp_path / "slot_pool.sqlite"
    first = claim_slot_combination(
        db,
        rng=random.Random(0),
        node_id="old-node",
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    old = int(time.time()) - 10
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE slot_combinations
            SET claimed_at = ?
            WHERE combination_id = ?
            """,
            (old, first.combination_id),
        )
    stale_first = type(first)(
        combination_id=first.combination_id,
        slots=first.slots,
        claimed_by=first.claimed_by,
        claimed_at=old,
    )
    second = claim_slot_combination(
        db,
        rng=random.Random(0),
        node_id="new-node",
        stale_after_sec=1,
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    assert second.combination_id == first.combination_id
    assert second.claimed_by == "new-node"

    with pytest.raises(RuntimeError, match="active claim"):
        consume_slot_combination(
            db,
            combination_id=stale_first.combination_id,
            claimed_by=stale_first.claimed_by,
            claimed_at=stale_first.claimed_at,
            strategy_id="gen_old_1",
            outcome_status="failed",
            failed_stage="generation",
        )

    consume_slot_combination(
        db,
        combination_id=second.combination_id,
        claimed_by=second.claimed_by,
        claimed_at=second.claimed_at,
        strategy_id="gen_new_1",
        outcome_status="complete",
        failed_stage=None,
    )
    assert slot_pool_counts(
        db,
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )["consumed"] == 1


def test_exhausted_pool_raises(tmp_path: Path) -> None:
    db = tmp_path / "slot_pool.sqlite"
    initialize_slot_pool(
        db,
        slots_table=SMALL_SLOTS,
        incompatible=SMALL_INCOMPATIBLE,
    )
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE slot_combinations SET status = 'consumed'")
    with pytest.raises(SlotPoolExhausted):
        claim_slot_combination(
            db,
            rng=random.Random(0),
            node_id="local",
            slots_table=SMALL_SLOTS,
            incompatible=SMALL_INCOMPATIBLE,
        )
