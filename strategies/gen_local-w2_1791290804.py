from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.5
    entry_level: float = 0.65
    raw_position_ceiling: float = 0.55
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791290804"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not 0.0 < params.entry_level < 1.0:
            raise ValueError("entry_level must be between zero and one")
        if not 0.0 <= params.raw_position_ceiling <= 1.0:
            raise ValueError("raw_position_ceiling must be between zero and one")
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")

        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        d = params.fractional_order
        w1 = -d
        w2 = w1 * (1.0 - d) / 2.0
        w3 = w2 * (2.0 - d) / 3.0
        fractional_close = close + w1 * close.shift(1) + w2 * close.shift(2) + w3 * close.shift(3)
        fractional_low = fractional_close.rolling(6, min_periods=6).min()
        fractional_high = fractional_close.rolling(6, min_periods=6).max()
        fractional_span = (fractional_high - fractional_low).replace(0.0, np.nan)
        fractional_position = (fractional_close - fractional_low) / fractional_span
        raw_low = close.rolling(6, min_periods=6).min()
        raw_high = close.rolling(6, min_periods=6).max()
        raw_span = (raw_high - raw_low).replace(0.0, np.nan)
        raw_position = (close - raw_low) / raw_span
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "fractional_low": fractional_low,
            "fractional_high": fractional_high,
            "fractional_span": fractional_span,
            "fractional_position": fractional_position,
            "previous_fractional_position": fractional_position.shift(1),
            "raw_low": raw_low,
            "raw_high": raw_high,
            "raw_span": raw_span,
            "raw_position": raw_position,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        entries = (
            indicators["fractional_position"].gt(params.entry_level)
            & indicators["previous_fractional_position"].le(params.entry_level)
            & indicators["raw_position"].le(params.raw_position_ceiling)
        ).fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        for bar in range(len(data)):
            if exit_bar >= 0:
                if bar == exit_bar:
                    exit_bar = -1
                    continue
                signals[bar] = 1
            elif entries[bar]:
                signals[bar] = 1
                exit_bar = bar + params.holding_bars
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
