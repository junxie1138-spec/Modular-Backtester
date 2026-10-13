from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeSlopeParams:
    compression_ratio: float = 0.75
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[RangeSlopeParams]):
    strategy_id = "gen_local-w2_1791295521"

    @classmethod
    def params_type(cls) -> type[RangeSlopeParams]:
        return RangeSlopeParams

    @classmethod
    def warmup_bars(cls, params: RangeSlopeParams) -> int:
        return 70

    def indicators(self, data: pd.DataFrame, params: RangeSlopeParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - prior_close).abs(),
             (data["low"] - prior_close).abs()],
            axis=1,
        ).max(axis=1)
        result["atr"] = true_range.rolling(14, min_periods=14).mean()
        range_width = (data["high"].rolling(10).max() - data["low"].rolling(10).min()) / data["close"].replace(0, np.nan)
        result["compressed"] = (range_width < params.compression_ratio * range_width.rolling(60, min_periods=60).median()).fillna(False)
        result["prior_high"] = data["high"].rolling(10, min_periods=10).max().shift(1)
        latent = ((data["high"] + data["low"] + data["close"]) / 3).ewm(span=8, adjust=False).mean()
        slope = latent.diff()
        expected_slope = slope.ewm(span=21, adjust=False).mean().shift(1)
        result["slope"] = slope
        result["innovation"] = slope - expected_slope
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeSlopeParams,
    ) -> SignalFrame:
        entry = (
            indicators["compressed"].shift(1).fillna(False)
            & (data["close"] > indicators["prior_high"])
            & (indicators["slope"] > 0)
            & (indicators["innovation"] > 0)
            & (indicators["innovation"].shift(1) <= 0)
            & indicators["atr"].notna()
        ).fillna(False).to_numpy()
        close = data["close"].to_numpy()
        opening = data["open"].to_numpy()
        atr = indicators["atr"].to_numpy()
        raw = np.zeros(len(data), dtype=int)
        held = False
        entry_bar = -1
        entry_price = 0.0
        fixed_stop = 0.0
        for i in range(len(data)):
            if i and raw[i - 1] == 1 and not held:
                held = True
                entry_bar = i
                entry_price = opening[i]
                fixed_stop = entry_price - params.stop_atr * atr[i - 1]
            elif i and raw[i - 1] == 0 and held:
                held = False
            if held:
                raw[i] = int(not (close[i] < fixed_stop or i - entry_bar >= 20))
            elif entry[i]:
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
