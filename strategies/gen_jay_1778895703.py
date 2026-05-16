from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalZScoreParams:
    ma_window: int = 60
    z_lookback: int = 60
    z_entry: float = -0.8
    seasonal_min_obs: int = 8
    seasonal_threshold: float = 0.0002
    atr_window: int = 14
    trail_k: float = 2.5


class GeneratedStrategy(BaseStrategy[SeasonalZScoreParams]):
    strategy_id = "gen_jay_1778895703"

    @classmethod
    def params_type(cls):
        return SeasonalZScoreParams

    @staticmethod
    def warmup_bars(params: SeasonalZScoreParams) -> int:
        # need seasonal_min_obs+1 occurrences of each DOM; each DOM appears ~every 22 bars
        seasonal_warmup = (params.seasonal_min_obs + 1) * 22
        return max(params.ma_window, params.z_lookback, params.atr_window, seasonal_warmup) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalZScoreParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Primitive 1: z-score distance from rolling MA
        ma = close.rolling(params.ma_window).mean()
        std = close.rolling(params.z_lookback).std()
        z_score = (close - ma) / std.replace(0.0, np.nan)

        # Primitive 2: rolling average return per day-of-month (seasonal score)
        # Loop is over 31 DOM values (not n bars), so vectorised within each group
        daily_return = close.pct_change()
        dom = pd.Series(data.index.day, index=data.index)
        seasonal_score = pd.Series(np.nan, index=data.index, dtype=float)
        for d in range(1, 32):
            mask = dom == d
            if mask.sum() < params.seasonal_min_obs + 1:
                continue
            grp = daily_return[mask]
            # shift(1): exclude current bar's own return from its own forecast
            rolled = (
                grp.shift(1)
                .rolling(params.seasonal_min_obs, min_periods=params.seasonal_min_obs)
                .mean()
            )
            seasonal_score[mask] = rolled.values

        # ATR
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        ind = pd.DataFrame(index=data.index)
        ind["z_score"] = z_score
        ind["seasonal_score"] = seasonal_score
        ind["atr"] = atr
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalZScoreParams,
    ) -> SignalFrame:
        close = data["close"].values
        z = indicators["z_score"].values
        seasonal = indicators["seasonal_score"].values
        atr_vals = indicators["atr"].values
        n = len(close)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_trade = False
        hwm = 0.0

        for i in range(n):
            if in_trade:
                if close[i] > hwm:
                    hwm = close[i]
                a = atr_vals[i]
                if not np.isnan(a) and a > 0.0:
                    stop = hwm - params.trail_k * a
                    if close[i] <= stop:
                        in_trade = False
                        signal[i] = 0
                        continue
                signal[i] = 1
            else:
                z_ok = not np.isnan(z[i]) and z[i] < params.z_entry
                seas_ok = not np.isnan(seasonal[i]) and seasonal[i] > params.seasonal_threshold
                if z_ok and seas_ok:
                    in_trade = True
                    hwm = close[i]
                    signal[i] = 1

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
