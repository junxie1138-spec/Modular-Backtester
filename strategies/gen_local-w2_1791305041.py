from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    streak_bars: int = 2
    contraction_ratio: float = 0.95
    pressure_threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local-w2_1791305041"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    @staticmethod
    def warmup_bars(params: CompressionParams) -> int:
        if not 1 <= params.streak_bars <= 8:
            raise ValueError("streak_bars must be between 1 and 8")
        if not 0.0 < params.contraction_ratio <= 1.0:
            raise ValueError("contraction_ratio must be in (0, 1]")
        if not 0.0 < params.pressure_threshold <= 1.0:
            raise ValueError("pressure_threshold must be in (0, 1]")
        return int(params.streak_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        width = (data["high"] - data["low"]).where(lambda x: x > 0)
        close = data["close"].where(data["close"] > 0)
        normalized_range = (width / close).replace([np.inf, -np.inf], np.nan)
        previous_range = normalized_range.shift(1)
        contracting = (
            normalized_range.notna()
            & previous_range.notna()
            & normalized_range.lt(previous_range * params.contraction_ratio)
        )
        streak = contracting.astype(int).groupby((~contracting).cumsum()).cumsum()
        streak = streak.clip(upper=params.streak_bars).astype(int)
        pressure = ((data["close"] - data["open"]) / width)
        pressure = pressure.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        compressed = streak.ge(params.streak_bars)
        setup = np.where(
            compressed & pressure.ge(params.pressure_threshold),
            1,
            np.where(compressed & pressure.le(-params.pressure_threshold), -1, 0),
        )
        return pd.DataFrame({
            "normalized_range": normalized_range.fillna(0.0),
            "previous_range": previous_range.fillna(0.0),
            "contracting": contracting.astype(int),
            "compression_streak": streak,
            "body_pressure": pressure,
            "setup": setup,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, desired in enumerate(setups):
            if position == 0:
                position = int(desired)
            elif desired == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    width = np.array([10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    direction = np.array([1, 1, 1, 1, -1, -1, 1])
    data = pd.DataFrame({
        "open": 100.0,
        "high": 100.0 + width / 2.0,
        "low": 100.0 - width / 2.0,
        "close": 100.0 + 0.4 * width * direction,
        "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=len(width)))
    params = CompressionParams(contraction_ratio=1.0)
    indicators = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 0, 1, 1, 0, -1]
    assert indicators.notna().all().all()
    assert GeneratedStrategy.warmup_bars(params) <= 10
    for length in range(1, len(data) + 1):
        prefix = data.iloc[:length]
        prefix_indicators = GeneratedStrategy.indicators(prefix, params)
        prefix_result = GeneratedStrategy.generate_signals(prefix, prefix_indicators, None, params)
        pd.testing.assert_frame_equal(prefix_indicators, indicators.iloc[:length])
        pd.testing.assert_frame_equal(prefix_result.data, result.data.iloc[:length])


if __name__ == "__main__":
    _self_check()
