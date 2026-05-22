from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    streak_min: int = 3
    decel_min: int = 2
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    sma_window: int = 200
    max_hold: int = 2
    base_size: float = 1.0
    decel_size_bonus: float = 0.25


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_a1_1778886542"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    @staticmethod
    def warmup_bars(params: ShockwaveParams) -> int:
        return int(max(params.sma_window, params.atr_period)) + 2

    @staticmethod
    def _streak(flag: pd.Series) -> pd.Series:
        # Length of the current run of True values, 0 on False bars.
        flag = flag.astype(bool)
        grp = (flag != flag.shift()).cumsum()
        return flag.astype(int).groupby(grp).cumsum()

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        ind = pd.DataFrame(index=data.index)

        # True range / ATR for the fixed volatility stop.
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        ind["atr"] = atr

        # 200-day MA regime filter (the hard twist).
        sma = close.rolling(params.sma_window, min_periods=params.sma_window).mean()
        ind["regime_long"] = (close > sma).fillna(False)

        # Consecutive down-close streak.
        down = (close < prev_close).fillna(False)
        down_streak = self._streak(down)
        ind["down_streak"] = down_streak

        # Decelerating-loss streak: down bars whose loss is smaller than the
        # prior bar's loss -- the selling shockwave is losing energy.
        loss = prev_close - close
        shrinking = (down & (loss < loss.shift(1))).fillna(False)
        decel_streak = self._streak(shrinking)
        ind["decel_streak"] = decel_streak

        # Entry: long down-streak AND a decelerating-loss profile AND bull regime.
        entry = (
            (down_streak >= params.streak_min)
            & (decel_streak >= params.decel_min)
            & ind["regime_long"]
        )
        ind["entry"] = entry.fillna(False)

        # Size scales modestly with how exhausted the sell-off is.
        extra = (decel_streak - params.decel_min).clip(lower=0)
        size = params.base_size + params.decel_size_bonus * extra
        ind["size"] = size.clip(lower=0.1, upper=2.0).fillna(params.base_size)

        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].to_numpy(dtype=bool)

        n = len(close)
        pos = np.zeros(n, dtype=int)

        in_pos = False
        stop_level = 0.0
        held = 0

        for i in range(n):
            if in_pos:
                held += 1
                exit_now = False
                if close[i] <= stop_level:
                    exit_now = True
                elif held >= params.max_hold:
                    exit_now = True
                if exit_now:
                    pos[i] = 0
                    in_pos = False
                else:
                    pos[i] = 1
            else:
                if entry[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                    in_pos = True
                    stop_level = close[i] - params.atr_stop_mult * atr[i]
                    held = 0
                    pos[i] = 1
                else:
                    pos[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size = indicators["size"].shift(1).fillna(params.base_size)
        df["size"] = size.clip(lower=0.1).astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
