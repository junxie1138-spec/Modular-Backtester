from __future__ import annotations

import random
from typing import Mapping

SLOT_NAMES: tuple[str, ...] = (
    "strategy_family",
    "signal_primitive",
    "holding_horizon",
    "direction",
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
    "constraint_twist": (
        "<=2 tunable params", "regime filter on 200-day MA",
        "signal-scaled position sizing", "symmetric entry/exit rule",
        "fixed-bar exit (no signal-based exit)",
        "two-primitive AND (both must agree)",
        "percentile threshold instead of fixed level",
        "warmup <=10 bars", "no stop-loss allowed",
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


def pull_slots(rng: random.Random) -> dict[str, str]:
    """Return one randomly-chosen value per slot."""
    return {name: rng.choice(SLOTS[name]) for name in SLOT_NAMES}
