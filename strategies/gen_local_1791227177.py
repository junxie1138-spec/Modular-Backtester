from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAutocorrParams:
    corr_lookback: int = 30
    min_autocorr: float = -0.25
    min_contraction: float = 0.10
    trail_atr: float = 2.0
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[RangeAutocorrParams]):
    strategy_id = "gen_local_1791227177"

    @classmethod
    def params_type(cls) -> type[RangeAutocorrParams]:
        return RangeAutocorrParams

    def warmup_bars(self, params: RangeAutocorrParams) -> int:
        return max(params.corr_lookback + 2, 14)

    def indicators(self, data: pd.DataFrame, params: RangeAutocorrParams) -> pd.DataFrame:
        day_range = ((data["high"] - data["low"]) / data["close"].where(data["close"] > 0)).replace([np.inf, -np.inf], np.nan)
        range_change = day_range.diff()
        autocorr = range_change.rolling(params.corr_lookback, min_periods=params.corr_lookback).corr(range_change.shift(1))
        contraction = ((day_range.shift(1) - day_range) / day_range.shift(1).replace(0, np.nan)).clip(0, 1)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        return pd.DataFrame({
            "range_autocorr": autocorr.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "contraction": contraction.fillna(0.0),
            "atr": true_range.rolling(14, min_periods=14).mean().fillna(0.0),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeAutocorrParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        holding = False
        entry_bar = 0
        peak = -np.inf
        stop = -np.inf
        position_size = 0.25
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            if not np.isfinite(close) or not np.isfinite(atr):
                continue

            if holding:
                peak = max(peak, close)
                stop = max(stop, peak - params.trail_atr * atr)
                if close <= stop or i - entry_bar + 1 >= params.max_hold:
                    holding = False
                    continue
                signals[i] = 1
                sizes[i] = position_size
            elif i >= warmup - 1 and atr > 0:
                corr = float(indicators["range_autocorr"].iloc[i])
                contraction = float(indicators["contraction"].iloc[i])
                if corr < params.min_autocorr and contraction >= params.min_contraction and close > float(data["open"].iloc[i]):
                    strength = min(1.0, (params.min_autocorr - corr) / 0.5) * min(1.0, contraction / 0.5)
                    position_size = 0.25 + 0.70 * strength
                    holding = True
                    entry_bar = i + 1
                    peak = -np.inf
                    stop = -np.inf
                    signals[i] = 1
                    sizes[i] = position_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
