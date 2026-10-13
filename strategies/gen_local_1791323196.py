from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RelativePositionParams:
    fractional_order: float = 0.6
    memory_bars: int = 12
    position_window: int = 3
    discordance_threshold: float = 0.3


class GeneratedStrategy(BaseStrategy[RelativePositionParams]):
    strategy_id = "gen_local_1791323196"

    @classmethod
    def params_type(cls):
        return RelativePositionParams

    def warmup_bars(self, params: RelativePositionParams) -> int:
        return int(params.memory_bars + params.position_window - 1)

    def indicators(self, data: pd.DataFrame, params: RelativePositionParams) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError("fractional_order must be between zero and one")
        if params.memory_bars < 2 or params.position_window < 2:
            raise ValueError("lookbacks must be at least two")
        if not (0.0 < params.discordance_threshold < 1.0):
            raise ValueError("discordance_threshold must be between zero and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.memory_bars):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        out["fractional_low"] = fractional.rolling(params.position_window).min()
        out["fractional_high"] = fractional.rolling(params.position_window).max()
        out["fractional_span"] = out["fractional_high"] - out["fractional_low"]
        out["fractional_position"] = (
            (fractional - out["fractional_low"])
            / out["fractional_span"].where(out["fractional_span"] > 0.0)
        ).where(out["fractional_span"] != 0.0, 0.5)
        out["session_span"] = data["high"] - data["low"]
        out["session_position"] = (
            (close - data["low"])
            / out["session_span"].where(out["session_span"] > 0.0)
        ).where(out["session_span"] != 0.0, 0.5)
        out["discordance"] = (
            out["session_position"] - out["fractional_position"]
        ).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RelativePositionParams,
    ) -> SignalFrame:
        score = indicators["discordance"]
        desired = np.select(
            [score >= params.discordance_threshold,
             score <= -params.discordance_threshold],
            [1, -1],
            default=0,
        )
        targets = np.zeros(len(data), dtype=int)
        position = 0
        for bar, direction in enumerate(desired):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            targets[bar] = position
        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RelativePositionParams()
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    close = 100.0 + np.sin(np.arange(40))
    data = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0,
         "close": close, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    derived = strategy.indicators(data, params)
    assert derived["discordance"].iloc[:strategy.warmup_bars(params) - 1].isna().all()
    pd.testing.assert_frame_equal(
        derived.iloc[:25], strategy.indicators(data.iloc[:25], params)
    )
    scores = pd.DataFrame(
        {"discordance": [np.nan, 0.4, 0.0, -0.4, -0.4, 0.0, 0.4, 0.4]},
        index=index[:8],
    )
    result = strategy.generate_signals(data.iloc[:8], scores, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    mirrored = strategy.generate_signals(data.iloc[:8], -scores, ctx, params).data
    assert mirrored["signal"].equals(-result["signal"])
    assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
