import random
from collections import Counter

from factory.slots import SLOT_NAMES, SLOTS, _INCOMPATIBLE, pull_slots


def test_seven_slots_with_expected_names() -> None:
    assert SLOT_NAMES == (
        "strategy_family",
        "signal_primitive",
        "holding_horizon",
        "direction",
        "exit_rule",
        "constraint_twist",
        "inspiration_anchor",
    )
    for name in SLOT_NAMES:
        assert len(SLOTS[name]) >= 3, name


def test_pull_returns_one_per_slot() -> None:
    rng = random.Random(42)
    pulled = pull_slots(rng)
    assert set(pulled.keys()) == set(SLOT_NAMES)
    for name, value in pulled.items():
        assert value in SLOTS[name], (name, value)


def test_pull_is_diverse_across_many_calls() -> None:
    rng = random.Random(0)
    families = Counter()
    for _ in range(200):
        families[pull_slots(rng)["strategy_family"]] += 1
    # All distinct families should appear within 200 pulls (~12 families).
    assert len(families) == len(SLOTS["strategy_family"])


def test_direction_is_weighted_toward_long_only() -> None:
    # spec: long-only x2, long/short x1
    rng = random.Random(7)
    counts = Counter(pull_slots(rng)["direction"] for _ in range(3000))
    assert counts["long-only"] > counts["long/short"]
    # Expect ratio ~2:1 — allow generous tolerance for randomness.
    ratio = counts["long-only"] / counts["long/short"]
    assert 1.4 < ratio < 2.6, ratio


def test_exit_rule_slot_has_six_values() -> None:
    assert len(SLOTS["exit_rule"]) == 6


def test_exit_entries_migrated_out_of_constraint_twist() -> None:
    twists = SLOTS["constraint_twist"]
    assert "fixed-bar exit (no signal-based exit)" not in twists
    assert "no stop-loss allowed" not in twists
    assert twists == (
        "<=2 tunable params",
        "regime filter on 200-day MA",
        "signal-scaled position sizing",
        "symmetric entry/exit rule",
        "two-primitive AND (both must agree)",
        "percentile threshold instead of fixed level",
        "warmup <=10 bars",
        "two-bar confirmation before entry",
    )


def test_guard_never_yields_incompatible_pair() -> None:
    rng = random.Random(2026)
    for _ in range(3000):
        slots = pull_slots(rng)
        pair = (slots["constraint_twist"], slots["exit_rule"])
        assert pair not in _INCOMPATIBLE, pair


def test_guard_preserves_exit_rule_support() -> None:
    # The guard re-picks only constraint_twist, so every exit_rule value
    # must still be reachable. This asserts preserved support, not frequency.
    rng = random.Random(99)
    seen = {pull_slots(rng)["exit_rule"] for _ in range(3000)}
    assert seen == set(SLOTS["exit_rule"])
