from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumStreakParams:
    entry_streak: int = 3
    exit_streak: int = 2
    vol_lookback: int = 20
    vol_mult: float = 1.0
    size: float = 1.0


def _consecutive_streak(mask: pd.Series) -> pd.Series:
    """Per-bar count of the current True run length in `mask`; 0 on False bars.

    Example:
        mask  = [F, T, T, T, F, T]
        out   = [0, 1, 2, 3, 0, 1]
    """
    mask = mask.fillna(False).astype(bool)
    grp = (mask != mask.shift(1)).cumsum()
    counts = mask.groupby(grp).cumcount().add(1)
    return counts.where(mask, 0).astype(int)


class MomentumStreakStrategy(BaseStrategy[MomentumStreakParams]):
    """
    Purpose:
        Symmetric momentum: enter LONG after `entry_streak` consecutive up-days
        (close > prev_close) confirmed by above-average volume on the entry
        bar; enter SHORT after `entry_streak` consecutive down-days similarly
        confirmed. Exit a long after `exit_streak` consecutive down-days;
        exit a short after `exit_streak` consecutive up-days. Doji days
        (close == prev_close) reset both streak counters to 0.

    Inputs:
        OHLCV dataframe with datetime index and `close`, `volume` columns.

    Outputs:
        SignalFrame with `signal` in {-1, 0, 1} and `size` columns.

    Side effects:
        None.

    Requires:
        `execution.allow_short: true` in the run config for the short side to
        fire. Without it, the portfolio simulator raises ShortNotAllowedError
        on the first short signal.
    """

    strategy_id = "momentum_streak"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return MomentumStreakParams

    def warmup_bars(self, params: MomentumStreakParams) -> int:
        return max(params.entry_streak, params.exit_streak, params.vol_lookback) + 1

    def indicators(self, data: pd.DataFrame, params: MomentumStreakParams) -> pd.DataFrame:
        close = data["close"]
        prev_close = close.shift(1)
        up = (close > prev_close).fillna(False)
        down = (close < prev_close).fillna(False)

        green_streak = _consecutive_streak(up)
        red_streak = _consecutive_streak(down)

        vol_sma = data["volume"].rolling(params.vol_lookback).mean()
        vol_confirm = (data["volume"] > params.vol_mult * vol_sma).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["up"] = up.astype(bool)
        out["down"] = down.astype(bool)
        out["green_streak"] = green_streak
        out["red_streak"] = red_streak
        out["vol_sma"] = vol_sma
        out["vol_confirm"] = vol_confirm.astype(bool)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MomentumStreakParams,
    ) -> SignalFrame:
        green_streak = indicators["green_streak"]
        red_streak = indicators["red_streak"]
        vol_confirm = indicators["vol_confirm"]

        long_entry = (green_streak >= params.entry_streak) & vol_confirm
        short_entry = (red_streak >= params.entry_streak) & vol_confirm
        long_exit = red_streak >= params.exit_streak
        short_exit = green_streak >= params.exit_streak

        state = np.zeros(len(data), dtype=int)
        for i in range(1, len(data)):
            prev = state[i - 1]
            le = bool(long_entry.iloc[i])
            se = bool(short_entry.iloc[i])
            lx = bool(long_exit.iloc[i])
            sx = bool(short_exit.iloc[i])
            if prev == 0:
                state[i] = 1 if le else (-1 if se else 0)
            elif prev == 1:
                if se:
                    state[i] = -1
                elif lx:
                    state[i] = 0
                else:
                    state[i] = 1
            else:  # prev == -1
                if le:
                    state[i] = 1
                elif sx:
                    state[i] = 0
                else:
                    state[i] = -1

        signal = pd.Series(state, index=data.index).shift(1).fillna(0).astype(int)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
