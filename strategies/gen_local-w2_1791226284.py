from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownCompressionParams:
    peak_bars: int = 40
    atr_bars: int = 14
    min_drawdown: float = 0.03
    min_depth_step: float = 0.001
    max_range_atr: float = 1.4
    stop_atr: float = 1.5
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[DrawdownCompressionParams]):
    strategy_id = "gen_local-w2_1791226284"

    @classmethod
    def params_type(cls) -> type[DrawdownCompressionParams]:
        return DrawdownCompressionParams

    @classmethod
    def warmup_bars(cls, params: DrawdownCompressionParams) -> int:
        return max(params.peak_bars + 2, params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: DrawdownCompressionParams) -> pd.DataFrame:
        peak = data["close"].rolling(params.peak_bars, min_periods=params.peak_bars).max()
        depth = ((peak - data["close"]) / peak.replace(0, np.nan)).fillna(0.0)
        previous_close = data["close"].shift(1)
        true_range = pd.concat((
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ), axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean().fillna(0.0)
        width = data["high"].rolling(3, min_periods=3).max() - data["low"].rolling(3, min_periods=3).min()
        return pd.DataFrame({
            "depth": depth,
            "depth_change": depth.diff().fillna(0.0),
            "atr": atr,
            "range_atr": (width / atr.replace(0, np.nan)).fillna(1e9),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownCompressionParams,
    ) -> SignalFrame:
        depth_change = indicators["depth_change"]
        compressed = indicators["range_atr"].lt(params.max_range_atr)
        common = (
            indicators["depth"].shift(2).ge(params.min_drawdown)
            & compressed
            & compressed.shift(1, fill_value=False)
        )
        long_setup = common & depth_change.lt(-params.min_depth_step) & depth_change.shift(1).lt(-params.min_depth_step)
        short_setup = common & depth_change.gt(params.min_depth_step) & depth_change.shift(1).gt(params.min_depth_step)
        long_entry = long_setup & ~long_setup.shift(1, fill_value=False)
        short_entry = short_setup & ~short_setup.shift(1, fill_value=False)

        intent = np.zeros(len(data), dtype=int)
        previous_position = 0
        entry_price = 0.0
        stop_distance = 0.0
        held_bars = 0
        for i in range(len(data)):
            position = int(intent[i - 1]) if i else 0
            if position and not previous_position:
                entry_price = float(data["open"].iat[i])
                stop_distance = params.stop_atr * float(indicators["atr"].iat[i - 1])
                held_bars = 0
            if position:
                held_bars += 1
                close = float(data["close"].iat[i])
                stopped = (position == 1 and close < entry_price - stop_distance) or (
                    position == -1 and close > entry_price + stop_distance
                )
                intent[i] = 0 if stopped or held_bars >= params.max_hold else position
            elif i >= self.warmup_bars(params):
                intent[i] = 1 if long_entry.iat[i] else -1 if short_entry.iat[i] else 0
            previous_position = position

        df = pd.DataFrame({"signal": intent, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
