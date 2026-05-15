from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

# Fixed (non-tunable) structural constants — the hard twist caps tunables at 2.
_ATR_PERIOD = 14
_DRAWDOWN_LOOKBACK = 50


@dataclass(slots=True)
class Params:
    # Number of consecutive down-closes (while in a drawdown) required to enter.
    streak_len: int = 4
    # Trailing-stop width: exit when close falls atr_mult * ATR below the
    # in-trade high-water mark.
    atr_mult: float = 3.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778883616"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        # roll_high needs 50 bars; ATR needs 14 (+1 for the prev-close shift).
        return max(_DRAWDOWN_LOOKBACK, _ATR_PERIOD + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Consecutive down-close streak count, vectorised.
        # diff()'s first value is NaN -> down is False there (NaN < 0 -> False).
        diff = close.diff()
        down = diff < 0
        # Each non-down bar starts a new group; cumulative sum within a group
        # of all-down bars yields the running consecutive count.
        grp = (~down).cumsum()
        down_streak = down.groupby(grp).cumsum()

        # Average True Range (NaN during warmup, handled in generate_signals).
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(_ATR_PERIOD, min_periods=_ATR_PERIOD).mean()

        # Rolling high of close -> drawdown gate (close below this == drawdown).
        roll_high = close.rolling(
            _DRAWDOWN_LOOKBACK, min_periods=_DRAWDOWN_LOOKBACK
        ).max()

        out = pd.DataFrame(index=data.index)
        out["down_streak"] = down_streak.astype(float)
        out["atr"] = atr
        out["roll_high"] = roll_high
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        streak = indicators["down_streak"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        roll_high = indicators["roll_high"].to_numpy(dtype=float)
        n = len(close)

        sig = np.zeros(n, dtype=int)
        in_pos = False
        high_water = 0.0

        for i in range(n):
            if not in_pos:
                # Entry: a long enough down-streak while price sits below its
                # 50-bar rolling high (i.e. in a drawdown). NaN-safe gates.
                ready = (
                    np.isfinite(atr[i])
                    and np.isfinite(roll_high[i])
                    and np.isfinite(streak[i])
                )
                if (
                    ready
                    and streak[i] >= float(params.streak_len)
                    and close[i] < roll_high[i]
                ):
                    in_pos = True
                    high_water = close[i]
                    sig[i] = 1
                else:
                    sig[i] = 0
            else:
                # Ratchet the high-water mark up only.
                if close[i] > high_water:
                    high_water = close[i]
                stop_level = high_water - params.atr_mult * atr[i]
                if np.isfinite(stop_level) and close[i] < stop_level:
                    in_pos = False
                    sig[i] = 0
                else:
                    sig[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
