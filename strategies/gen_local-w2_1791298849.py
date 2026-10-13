from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_window: int = 20
    percentile: float = 0.85
    atr_mult: float = 2.0
    percentile_window: int = 126
    vol_window: int = 20
    atr_window: int = 14
    target_vol: float = 0.12
    breakeven_pct: float = 0.02
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791298849"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return max(params.ma_window + params.percentile_window,
                   params.vol_window + params.percentile_window,
                   params.atr_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"]
        mean = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        sd = close.rolling(params.ma_window, min_periods=params.ma_window).std()
        z = (close - mean) / sd.replace(0, np.nan)
        returns = close.pct_change()
        vol = returns.rolling(params.vol_window, min_periods=params.vol_window).std() * np.sqrt(252)
        previous_close = close.shift(1)
        true_range = pd.concat((data["high"] - data["low"],
                                (data["high"] - previous_close).abs(),
                                (data["low"] - previous_close).abs()), axis=1).max(axis=1)
        result = pd.DataFrame(index=data.index)
        result["z"] = z.fillna(0.0)
        result["threshold"] = z.abs().rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.percentile).fillna(0.0)
        result["vol_ann"] = vol.fillna(0.0)
        result["vol_median"] = vol.rolling(params.percentile_window, min_periods=params.percentile_window).median().fillna(0.0)
        result["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean().fillna(0.0)
        return result

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = (params.target_vol / indicators["vol_ann"].replace(0, np.nan)).clip(0.05, 0.95).fillna(0.95).astype(float)
        position = 0
        entry = None
        stop = None
        held = 0
        warmup = GeneratedStrategy.warmup_bars(params)

        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            exited = False
            if position:
                if entry is None:
                    entry = float(data["open"].iloc[i])
                held += 1
                gain = position * (close / entry - 1)
                if stop is not None and position * (close - stop) <= 0:
                    exited = True
                else:
                    if gain >= params.breakeven_pct and stop is None:
                        stop = entry
                    if stop is not None:
                        candidate = close - position * params.atr_mult * float(indicators["atr"].iloc[i])
                        stop = max(stop, candidate) if position == 1 else min(stop, candidate)
                if held >= params.max_hold:
                    exited = True
                if exited:
                    position, entry, stop, held = 0, None, None, 0

            if position == 0 and not exited and i >= warmup:
                z = float(indicators["z"].iloc[i])
                threshold = float(indicators["threshold"].iloc[i])
                previous_z = float(indicators["z"].iloc[i - 1])
                previous_threshold = float(indicators["threshold"].iloc[i - 1])
                if threshold > 0 and abs(z) > threshold and abs(previous_z) <= previous_threshold:
                    direction = 1 if z > 0 else -1
                    if indicators["vol_ann"].iloc[i] <= indicators["vol_median"].iloc[i]:
                        direction = -direction
                    position = direction
                    entry, stop, held = None, None, 0
            df.iat[i, df.columns.get_loc("signal")] = position

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
