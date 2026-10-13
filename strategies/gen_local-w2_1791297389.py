from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeCompressionParams:
    range_window: int = 12
    atr_window: int = 14
    baseline_window: int = 60
    compression_ratio: float = 0.65
    frac_order: float = 0.4
    frac_terms: int = 12
    pressure_window: int = 30
    pressure_z: float = 0.3
    stop_atr: float = 2.5
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[RangeCompressionParams]):
    strategy_id = "gen_local-w2_1791297389"

    @classmethod
    def params_type(cls):
        return RangeCompressionParams

    def warmup_bars(self, params: RangeCompressionParams) -> int:
        return max(
            params.frac_terms + params.pressure_window,
            max(params.range_window, params.atr_window) + params.baseline_window + 7,
        )

    def indicators(self, data: pd.DataFrame, params: RangeCompressionParams) -> pd.DataFrame:
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()

        weights = [1.0]
        for k in range(1, params.frac_terms):
            weights.append(-weights[-1] * (params.frac_order - k + 1) / k)
        fractional_close = sum(weight * close.shift(k) for k, weight in enumerate(weights))
        fractional_close = fractional_close.div(close.replace(0, np.nan))
        mean = fractional_close.rolling(params.pressure_window, min_periods=params.pressure_window).mean()
        std = fractional_close.rolling(params.pressure_window, min_periods=params.pressure_window).std(ddof=0)
        pressure = (fractional_close - mean).div(std.replace(0, np.nan))

        range_width = data["high"].rolling(params.range_window, min_periods=params.range_window).max() - data["low"].rolling(params.range_window, min_periods=params.range_window).min()
        range_ratio = range_width.div(atr.replace(0, np.nan))
        baseline = range_ratio.rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        compression = range_ratio.lt(baseline * params.compression_ratio)
        prior_high = data["high"].shift(1).rolling(params.range_window, min_periods=params.range_window).max()

        return pd.DataFrame({
            "atr": atr.fillna(0.0),
            "fractional_close": fractional_close.fillna(0.0),
            "pressure": pressure.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "range_ratio": range_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "compression": compression.fillna(False),
            "breakout": close.gt(prior_high).fillna(False),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeCompressionParams,
    ) -> SignalFrame:
        breakout_pressure = indicators["breakout"] & indicators["pressure"].gt(params.pressure_z)
        recent_compression = indicators["compression"].shift(1).rolling(6, min_periods=6).max().fillna(0).astype(bool)
        eligible = (breakout_pressure & breakout_pressure.shift(1).fillna(False) & recent_compression & indicators["atr"].gt(0)).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        desired = 0
        entry_bar = -1
        stop = 0.0

        for i in range(len(data)):
            if desired == 1 and entry_bar < 0:
                entry_bar = i
                stop = opens[i] - params.stop_atr * atrs[i - 1]
            elif desired == 0:
                entry_bar = -1

            if entry_bar >= 0:
                if closes[i] < stop or i - entry_bar + 1 >= params.max_hold:
                    desired = 0
            elif i >= self.warmup_bars(params) and eligible[i]:
                desired = 1
            raw[i] = desired

        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
