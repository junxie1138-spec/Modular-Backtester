from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    shock_multiple: float = 1.8
    echo_floor: float = 0.85
    hold_bars: int = 17


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local_1791297099"

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    @classmethod
    def warmup_bars(cls, params: EchoParams) -> int:
        return 8

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        baseline = true_range.rolling(5, min_periods=5).mean().shift(2)
        shock_range = true_range.shift(1)
        scale = baseline.replace(0, np.nan)
        shock_strength = (shock_range / scale).fillna(0.0)
        echo_strength = (true_range / scale).fillna(0.0)
        echo = (true_range < shock_range) & (echo_strength >= params.echo_floor)
        shock = shock_strength >= params.shock_multiple
        upward = (data["close"].shift(1) > data["open"].shift(1)) & (data["close"] > data["high"].shift(1))
        downward = (data["close"].shift(1) < data["open"].shift(1)) & (data["close"] < data["low"].shift(1))
        candidate = np.select([shock & echo & upward, shock & echo & downward], [1, -1], default=0)
        return pd.DataFrame({"candidate": candidate}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EchoParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i, candidate in enumerate(indicators["candidate"].to_numpy(dtype=int)):
            if position and i >= exit_at:
                position = 0
            if not position and i > exit_at and candidate:
                position = int(candidate)
                exit_at = i + params.hold_bars
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
