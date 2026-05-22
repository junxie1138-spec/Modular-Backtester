from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

# --- Fixed (non-tunable) mechanism constants -------------------------------
# The twist caps tunable params at 2. Everything below is hardcoded so the
# only optimised knobs are the two exit parameters in GapSeasonParams.
_ATR_WINDOW = 14          # bars for the ATR used by the trailing stop
_MAX_HOLD = 20            # ~3-4 trading weeks: hard horizon cap on the hold
_GAP_THRESHOLD = 0.0      # entry requires a strictly positive first-of-month gap


@dataclass(slots=True)
class GapSeasonParams:
    """Exactly two tunable parameters, both governing the exit mechanic."""
    breakeven_pct: float = 0.03      # +X% unrealised gain that arms breakeven
    trail_atr_mult: float = 2.5      # k: ATR multiple for the trailing stop


class GeneratedStrategy(BaseStrategy[GapSeasonParams]):
    """Seasonal gap omen: on the first trading day of each calendar month,
    go long if the overnight gap (open vs prior close) is positive, then hold
    through the month with a breakeven-then-ATR-trail exit."""

    strategy_id = "gen_a1_1778911165"

    @classmethod
    def params_type(cls) -> type[GapSeasonParams]:
        return GapSeasonParams

    @staticmethod
    def warmup_bars(params: GapSeasonParams) -> int:
        # ATR needs _ATR_WINDOW bars; the gap needs one prior close.
        return _ATR_WINDOW + 1

    def indicators(self, data: pd.DataFrame, params: GapSeasonParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        openp = data["open"].astype(float)

        prev_close = close.shift(1)

        # True range -> simple-average ATR (NaN during warmup, handled later).
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(_ATR_WINDOW).mean()

        # Overnight gap into this bar; first bar has no prior close -> 0.
        gap = (openp / prev_close - 1.0).fillna(0.0)

        # First trading day of each calendar month.
        period = data.index.to_period("M")
        first_of_month = pd.Series(~period.duplicated(), index=data.index)

        ind = pd.DataFrame(index=data.index)
        ind["atr"] = atr
        ind["gap"] = gap
        ind["first_of_month"] = first_of_month.astype(bool)
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSeasonParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        gap = indicators["gap"].to_numpy(dtype=float)
        first = indicators["first_of_month"].to_numpy(dtype=bool)

        k = float(params.trail_atr_mult)
        be = float(params.breakeven_pct)

        sig = np.zeros(n, dtype=int)

        position = 0
        entry_price = 0.0
        stop = 0.0
        armed = False
        bars_held = 0

        for i in range(n):
            a = atr[i]

            if position == 0:
                # Entry: first trading day of the month with a positive gap,
                # and a usable ATR for the protective stop.
                if (
                    first[i]
                    and gap[i] > _GAP_THRESHOLD
                    and np.isfinite(a)
                    and a > 0.0
                ):
                    position = 1
                    entry_price = close[i]
                    stop = entry_price - k * a   # initial protective stop
                    armed = False
                    bars_held = 0
                    sig[i] = 1
                else:
                    sig[i] = 0
            else:
                bars_held += 1

                # Breakeven arming: once +X% is touched, lift stop to entry.
                if not armed and high[i] >= entry_price * (1.0 + be):
                    armed = True
                    if entry_price > stop:
                        stop = entry_price

                # Trailing: once armed, ratchet the stop up by k*ATR.
                if armed and np.isfinite(a):
                    candidate = close[i] - k * a
                    if candidate > stop:
                        stop = candidate

                # Exit: stop pierced this bar, or the holding horizon is hit.
                if low[i] <= stop or bars_held >= _MAX_HOLD:
                    position = 0
                    sig[i] = 0
                else:
                    sig[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
