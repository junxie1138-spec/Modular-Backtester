from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback_high: int = 20
    min_streak: int = 3
    ma_window: int = 200
    atr_window: int = 14
    vol_spike_pct: float = 0.95
    vol_lookback: int = 100
    refractory_bars: int = 5
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779884688"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.ma_window,
            params.vol_lookback,
            params.lookback_high,
            params.atr_window,
        )) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        rolling_high = close.rolling(params.lookback_high, min_periods=1).max()
        drawdown = close / rolling_high - 1.0

        deepening_bool = (drawdown < drawdown.shift(1)).fillna(False)
        deepening_int = deepening_bool.astype(int)
        streak_groups = (~deepening_bool).cumsum()
        streak = deepening_int.groupby(streak_groups).cumsum()

        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()

        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        atr_pct = atr.rolling(params.vol_lookback, min_periods=params.vol_lookback).rank(pct=True)
        spike = (atr_pct >= params.vol_spike_pct).fillna(False).astype(int)

        out = pd.DataFrame({
            "drawdown": drawdown,
            "deepening": deepening_int,
            "streak": streak.astype(float),
            "ma": ma,
            "atr": atr,
            "atr_pct": atr_pct.fillna(0.0),
            "spike": spike,
        }, index=data.index)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy()
        ma_arr = indicators["ma"].to_numpy()
        streak_arr = indicators["streak"].to_numpy()
        deepening_arr = indicators["deepening"].to_numpy().astype(bool)
        spike_arr = indicators["spike"].to_numpy().astype(bool)

        signal = np.zeros(n, dtype=np.int64)
        in_pos = False
        bars_since_spike = n + 1

        min_streak = int(params.min_streak)
        refractory = int(params.refractory_bars)

        for t in range(1, n):
            if spike_arr[t]:
                bars_since_spike = 0
            else:
                bars_since_spike += 1

            ma_val = ma_arr[t]
            regime_ok = np.isfinite(ma_val) and close[t] > ma_val

            if in_pos:
                if deepening_arr[t]:
                    in_pos = False
                    signal[t] = 0
                else:
                    signal[t] = 1
            else:
                prev_streak = streak_arr[t - 1]
                prev_streak_ok = np.isfinite(prev_streak) and prev_streak >= min_streak
                turn_now = not deepening_arr[t]
                in_refractory = bars_since_spike < refractory
                if regime_ok and prev_streak_ok and turn_now and not in_refractory:
                    in_pos = True
                    signal[t] = 1
                else:
                    signal[t] = 0

        df = pd.DataFrame({"signal": signal}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
