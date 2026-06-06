from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TurnOfMonthRangeParams:
    """Only two tunable params (hard twist): both govern the exit."""
    profit_target: float = 0.04
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[TurnOfMonthRangeParams]):
    """Long-only turn-of-month seasonality, gated by high-low range contraction.

    Entry (decided on bar N's close, filled on N+1):
      * the bar sits inside the turn-of-month window: one of the first 2 or
        last 3 trading days of its calendar month;
      * the market is in a contracted-range 'susceptible' state: the 5-bar
        mean high-low range is below 90% of the 20-bar mean high-low range;
      * a 50-bar trend filter holds (close above its 50-bar mean).

    Exit: profit-target OR time-stop, whichever comes first - flatten when the
    return from entry reaches +profit_target, or after max_hold bars held.
    """

    strategy_id = "gen_a2_1779153208"

    # Hardcoded structural constants (kept out of the param space by the twist).
    _RANGE_FAST = 5
    _RANGE_SLOW = 20
    _TREND_WIN = 50
    _COMPRESSION_RATIO = 0.9
    _TOM_HEAD = 2   # first N trading days of the month
    _TOM_TAIL = 3   # last N trading days of the month

    @classmethod
    def params_type(cls) -> type[TurnOfMonthRangeParams]:
        return TurnOfMonthRangeParams

    def warmup_bars(self, params: TurnOfMonthRangeParams) -> int:
        # Longest lookback is the 50-bar trend mean; pad for safety.
        return 60

    def indicators(self, data: pd.DataFrame, params: TurnOfMonthRangeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        # High-low range dynamics: contraction = fast mean below slow mean.
        rng = (high - low).abs()
        r_fast = rng.rolling(self._RANGE_FAST, min_periods=self._RANGE_FAST).mean()
        r_slow = rng.rolling(self._RANGE_SLOW, min_periods=self._RANGE_SLOW).mean()
        ma_trend = close.rolling(self._TREND_WIN, min_periods=self._TREND_WIN).mean()

        out["r_fast"] = r_fast
        out["r_slow"] = r_slow
        out["ma_trend"] = ma_trend

        # Turn-of-month window from the datetime index alone.
        periods = data.index.to_period("M")
        grp = pd.Series(periods, index=data.index)
        pos_from_start = grp.groupby(grp).cumcount()
        counts = grp.groupby(grp).transform("size")
        pos_from_end = counts - 1 - pos_from_start
        tom = (pos_from_start < self._TOM_HEAD) | (pos_from_end < self._TOM_TAIL)

        # Range-contraction gate. NaN comparisons evaluate to False (safe).
        compression = (r_fast < self._COMPRESSION_RATIO * r_slow).fillna(False)
        trend = (close > ma_trend).fillna(False)

        entry_raw = tom.to_numpy(dtype=bool) & compression.to_numpy(dtype=bool) & trend.to_numpy(dtype=bool)
        out["entry_raw"] = pd.Series(entry_raw, index=data.index).astype(bool)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TurnOfMonthRangeParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        entry_raw = indicators["entry_raw"].to_numpy(dtype=bool)

        pt = float(params.profit_target)
        mh = int(params.max_hold)
        if mh < 1:
            mh = 1

        sig = np.zeros(n, dtype=np.int64)
        position = 0
        entry_price = 0.0
        bars_held = 0

        # Path-dependent profit-target + time-stop exit.
        for i in range(n):
            if position == 1:
                bars_held += 1
                ret = close[i] / entry_price - 1.0 if entry_price > 0.0 else 0.0
                if ret >= pt or bars_held >= mh:
                    position = 0
                    sig[i] = 0
                else:
                    sig[i] = 1
            else:
                if entry_raw[i]:
                    position = 1
                    entry_price = close[i]
                    bars_held = 0
                    sig[i] = 1
                else:
                    sig[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0

        # MANDATORY one-bar shift: decide on bar N, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
