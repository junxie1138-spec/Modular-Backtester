from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SweepParams:
    channel_bars: int = 3
    sweep_bars: int = 2


class GeneratedStrategy(BaseStrategy[SweepParams]):
    strategy_id = "gen_local-w2_1791271856"

    @classmethod
    def params_type(cls):
        return SweepParams

    def warmup_bars(self, params: SweepParams) -> int:
        return int(max(params.channel_bars, params.sweep_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: SweepParams) -> pd.DataFrame:
        if params.channel_bars < 1 or params.sweep_bars < 1:
            raise ValueError("Lookbacks must be positive.")
        out = pd.DataFrame(index=data.index)
        out["previous_low"] = data["low"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["upper"] = out["previous_high"].rolling(
            params.channel_bars, min_periods=params.channel_bars
        ).max()
        out["lower"] = out["previous_low"].rolling(
            params.channel_bars, min_periods=params.channel_bars
        ).min()
        out["rejected_down"] = (
            data["low"].lt(out["previous_low"])
            & data["close"].gt(out["previous_low"])
        ).fillna(False)
        out["rejected_up"] = (
            data["high"].gt(out["previous_high"])
            & data["close"].lt(out["previous_high"])
        ).fillna(False)
        for side in ("down", "up"):
            mask = out["rejected_" + side]
            out[side + "_streak"] = mask.astype(int).groupby(
                (~mask).cumsum()
            ).cumsum()
        out["long_entry"] = (
            out["down_streak"].ge(params.sweep_bars)
            & data["close"].gt(out["upper"])
        ).fillna(False)
        out["short_entry"] = (
            out["up_streak"].ge(params.sweep_bars)
            & data["close"].lt(out["lower"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SweepParams,
    ) -> SignalFrame:
        candidate = (
            indicators["long_entry"].fillna(False).astype(int)
            - indicators["short_entry"].fillna(False).astype(int)
        )
        targets = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(candidate.to_numpy()):
            if position != 0 and direction == -position:
                position = 0
            elif position == 0:
                position = int(direction)
            targets[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = targets
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    data = pd.DataFrame(
        {
            "open": [10, 10, 9, 11.5, 11, 7, 5],
            "high": [11, 11, 12, 13, 14, 15, 16],
            "low": [9, 8, 7, 8, 6, 5, 4],
            "close": [10, 9.5, 11.5, 11, 6.5, 5.5, 4.5],
            "volume": [100] * 7,
        },
        index=pd.date_range("2020-01-01", periods=7),
    )
    strategy = GeneratedStrategy()
    params = SweepParams(channel_bars=2, sweep_bars=2)
    ctx = StrategyContext(
        symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params)
    )
    result = strategy.generate_signals(
        data, strategy.indicators(data, params), ctx, params
    ).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, -1]
    assert result["size"].gt(0).all()
