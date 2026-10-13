from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeNoiseParams:
    vol_window: int = 6
    noise_ratio: float = 1.5
    pullback_atr: float = 0.75
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RangeNoiseParams]):
    strategy_id = "gen_local_1790993650"

    @classmethod
    def params_type(cls) -> type[RangeNoiseParams]:
        return RangeNoiseParams

    @staticmethod
    def warmup_bars(params: RangeNoiseParams) -> int:
        window = max(3, min(9, int(params.vol_window)))
        return window + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeNoiseParams) -> pd.DataFrame:
        window = max(3, min(9, int(params.vol_window)))
        result = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        high = data["high"].where(data["high"] > 0)
        low = data["low"].where(data["low"] > 0)
        previous = close.shift(1)
        result["log_return"] = np.log(close / previous)
        result["log_range"] = np.log(high / low).where(high >= low)
        result["range_variance"] = result["log_range"].pow(2) / (4.0 * np.log(2.0))
        result["range_volatility"] = np.sqrt(result["range_variance"].rolling(window, min_periods=window).mean())
        result["close_volatility"] = result["log_return"].rolling(window, min_periods=window).std(ddof=0)
        result["noise_ratio"] = result["range_volatility"] / result["close_volatility"].where(result["close_volatility"] > 1e-12)
        components = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1)
        result["true_range"] = components.max(axis=1).where(components.notna().all(axis=1))
        result["atr"] = result["true_range"].rolling(window, min_periods=window).mean()
        result["displacement"] = close.diff(3)
        result["pullback_atr"] = -result["displacement"] / result["atr"].where(result["atr"] > 0)
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeNoiseParams) -> SignalFrame:
        eligible = (
            (indicators["noise_ratio"] >= params.noise_ratio)
            & (indicators["pullback_atr"] >= params.pullback_atr)
            & (indicators["log_return"] < 0)
        ).fillna(False).to_numpy(dtype=bool)
        hold = max(3, min(5, int(params.hold_bars)))
        warmup = GeneratedStrategy.warmup_bars(params)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
                continue
            if i >= warmup - 1 and eligible[i]:
                signal[i] = 1
                exit_bar = i + hold
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
