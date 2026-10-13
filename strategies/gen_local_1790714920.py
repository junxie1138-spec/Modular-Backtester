from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    volatility_window: int = 10
    baseline_window: int = 40
    range_window: int = 20
    compression_ratio: float = 0.75
    range_threshold: float = 0.8
    profit_target: float = 0.025
    max_hold: int = 8
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1790714920"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return max(params.baseline_window + params.volatility_window, params.range_window) + 1

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        returns = data["close"].pct_change()
        short_vol = returns.rolling(params.volatility_window, min_periods=params.volatility_window).std()
        baseline_vol = returns.rolling(params.baseline_window, min_periods=params.baseline_window).std()
        out["compression"] = short_vol / baseline_vol.replace(0, np.nan)
        rolling_low = data["low"].rolling(params.range_window, min_periods=params.range_window).min()
        rolling_high = data["high"].rolling(params.range_window, min_periods=params.range_window).max()
        span = (rolling_high - rolling_low).replace(0, np.nan)
        out["range_location"] = (data["close"] - rolling_low) / span
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        signal = pd.Series(0, index=data.index, dtype=int)
        valid = indicators["compression"].notna() & indicators["range_location"].notna()
        compressed = indicators["compression"] <= params.compression_ratio
        high = indicators["range_location"] >= params.range_threshold
        low = indicators["range_location"] <= 1.0 - params.range_threshold
        signal.loc[valid & compressed & high] = 1
        signal.loc[valid & compressed & low] = -1

        # Convert setup signals into target- and time-limited position states.
        raw = signal.to_numpy()
        close = data["close"].to_numpy(dtype=float)
        managed = np.zeros(len(data), dtype=int)
        direction = 0
        entry = 0.0
        held = 0
        for i in range(len(data)):
            if direction:
                held += 1
                reached_target = (close[i] / entry - 1.0) * direction >= params.profit_target
                if reached_target or held >= params.max_hold or raw[i] == -direction:
                    direction = 0
                    held = 0
            if direction == 0 and raw[i] != 0:
                direction = int(raw[i])
                entry = close[i]
                held = 0
            managed[i] = direction

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(managed, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
