from __future__ import annotations

import random
from typing import Mapping

# exit_rule slot values, named so SLOTS and the guard logic stay in sync.
_EXIT_TRAILING_HWM = (
    "rolling-high trailing stop (track the highest close since entry; exit "
    "when close falls k*ATR below that in-trade high-water mark; the stop "
    "only ratchets up)"
)
_EXIT_FIXED_BAR = "fixed-bar exit (exit exactly N bars after entry, no signal-based exit)"
_EXIT_SIGNAL_REVERSAL = "signal-reversal exit (exit only when the entry condition flips)"
_EXIT_PROFIT_TARGET_TIME = (
    "profit-target + time-stop (exit at +X% gain or after N bars, whichever "
    "comes first)"
)
_EXIT_VOL_STOP = (
    "fixed volatility-stop (exit when close falls below entry price minus "
    "k*ATR - fixed, not trailing)"
)
_EXIT_BREAKEVEN_TRAIL = (
    "breakeven-then-trail (after price reaches +X%, move the stop to entry "
    "price, then trail by k*ATR; the stop only ever moves up, never down)"
)

# constraint_twist values referenced by the incompatibility guard.
_TWIST_SYMMETRIC = "symmetric entry/exit rule"
_TWIST_SHORT_WARMUP = "warmup <=10 bars"

SLOT_NAMES: tuple[str, ...] = (
    "strategy_family",
    "signal_primitive",
    "holding_horizon",
    "direction",
    "exit_rule",
    "constraint_twist",
    "inspiration_anchor",
)

SLOTS: Mapping[str, tuple[str, ...]] = {
    "strategy_family": (
        "momentum", "mean-reversion", "breakout", "volatility-targeting",
        "seasonality", "regime-switching", "range-compression",
        "gap-behavior", "drawdown-recovery", "autocorrelation",
        "relative-position", "trend-strength",
    ),
    "signal_primitive": (
        "close-to-close returns", "high-low range dynamics",
        "volume-confirmed moves", "volatility (std/ATR)",
        "gap (open vs prior close)", "rolling rank/percentile",
        "consecutive-streak count", "distance-from-MA (z-score)",
        "rate-of-change acceleration", "drawdown depth",
    ),
    "holding_horizon": (
        "1-2 days", "3-5 days", "1-2 weeks", "3-4 weeks",
    ),
    "direction": (
        "long-only", "long-only", "long/short",
    ),
    "exit_rule": (
        _EXIT_TRAILING_HWM,
        _EXIT_FIXED_BAR,
        _EXIT_SIGNAL_REVERSAL,
        _EXIT_PROFIT_TARGET_TIME,
        _EXIT_VOL_STOP,
        _EXIT_BREAKEVEN_TRAIL,
    ),
    "constraint_twist": (
        "<=2 tunable params", "regime filter on 200-day MA",
        "signal-scaled position sizing", _TWIST_SYMMETRIC,
        "two-primitive AND (both must agree)",
        "percentile threshold instead of fixed level",
        _TWIST_SHORT_WARMUP,
        "two-bar confirmation before entry",
    ),
    "inspiration_anchor": (
        "hysteresis control", "predator-prey cycles",
        "queue overflow / capacity limits", "signal-to-noise filtering",
        "spring tension / elastic restoring force",
        "epidemic curves (susceptible-infected)",
        "traffic shockwaves", "elastic vs plastic deformation",
        "refractory period after a spike", "tide tables / standing waves",
    ),
}


# (constraint_twist, exit_rule) pairs that must never co-occur.
# - "symmetric entry/exit rule" implies the exit is the logical inverse of the
#   entry, which only holds for the signal-reversal exit.
# - "warmup <=10 bars" is too short for the ATR-based exits to be stable.
_INCOMPATIBLE: frozenset[tuple[str, str]] = frozenset(
    {
        (_TWIST_SYMMETRIC, e)
        for e in SLOTS["exit_rule"]
        if e != _EXIT_SIGNAL_REVERSAL
    }
    | {
        (_TWIST_SHORT_WARMUP, e)
        for e in (_EXIT_TRAILING_HWM, _EXIT_VOL_STOP, _EXIT_BREAKEVEN_TRAIL)
    }
)


def pull_slots(rng: random.Random) -> dict[str, str]:
    """Return one randomly-chosen value per slot.

    After drawing all slots, re-pick `constraint_twist` from its compatible
    subset if the drawn (constraint_twist, exit_rule) pair is in _INCOMPATIBLE.
    `exit_rule` is never re-drawn, so its draw distribution is unaffected.
    The compatible subset is never empty: for any single exit_rule value at
    most 2 of the 8 constraint_twist values are forbidden.
    """
    slots = {name: rng.choice(SLOTS[name]) for name in SLOT_NAMES}
    exit_rule = slots["exit_rule"]
    if (slots["constraint_twist"], exit_rule) in _INCOMPATIBLE:
        compatible = [
            t for t in SLOTS["constraint_twist"]
            if (t, exit_rule) not in _INCOMPATIBLE
        ]
        slots["constraint_twist"] = rng.choice(compatible)
    return slots
