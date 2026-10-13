from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPressureParams:
    gap_window: int = 4
    strength_threshold: float = 1.1
    max_gap_share: float = 0.6


class GeneratedStrategy(BaseStrategy[GapPressureParams]):
    strategy_id = "gen_local-w2_1791260426"

    @classmethod
    def params_type(cls) -> type[GapPressureParams]:
        return GapPressureParams

    @staticmethod
    def warmup_bars(params: GapPressureParams) -> int:
        return int(params.gap_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapPressureParams) -> pd.DataFrame:
        if params.gap_window < 2:
            raise ValueError("gap_window must be at least 2")
        if not np.isfinite(params.strength_threshold) or params.strength_threshold <= 0:
            raise ValueError("strength_threshold must be positive and finite")
        if not np.isfinite(params.max_gap_share) or not 0 < params.max_gap_share <= 1:
            raise ValueError("max_gap_share must be in (0, 1]")
        result = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1).where(lambda s: s > 0)
        result["gap"] = (data["open"] / prior_close - 1.0).replace([np.inf, -np.inf], np.nan)
        window = int(params.gap_window)
        result["gap_sum"] = result["gap"].rolling(window, min_periods=window).sum()
        result["gap_energy"] = result["gap"].pow(2).rolling(window, min_periods=window).sum()
        result["absolute_gap_sum"] = result["gap"].abs().rolling(window, min_periods=window).sum()
        result["largest_absolute_gap"] = result["gap"].abs().rolling(window, min_periods=window).max()
        result["strength"] = result["gap_sum"] / np.sqrt(result["gap_energy"].where(result["gap_energy"] > 0))
        result["largest_gap_share"] = result["largest_absolute_gap"] / result["absolute_gap_sum"].where(result["absolute_gap_sum"] > 0)
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapPressureParams,
    ) -> SignalFrame:
        distributed = indicators["largest_gap_share"].le(params.max_gap_share)
        long_entry = distributed & indicators["strength"].ge(params.strength_threshold)
        short_entry = distributed & indicators["strength"].le(-params.strength_threshold)
        entry = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(entry):
            direction = int(direction)
            if position == 0:
                position = direction
            elif direction == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "strength": [np.nan, 1.2, 0.0, -1.2, -1.2, 0.0, 1.2, 1.2],
        "largest_gap_share": [np.nan, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, GapPressureParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    reflected = indicators.copy()
    reflected["strength"] *= -1
    mirrored = GeneratedStrategy.generate_signals(data, reflected, None, GapPressureParams())
    assert mirrored.data["signal"].tolist() == (-result.data["signal"]).tolist()
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
