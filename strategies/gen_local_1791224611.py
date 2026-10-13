from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 5
    accel_lag: int = 3
    baseline_bars: int = 63
    run_bars: int = 5
    threshold: float = 1.25
    atr_bars: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791224611"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return max(
            params.roc_bars + params.accel_lag + params.baseline_bars + 1,
            params.run_bars + 1,
            params.atr_bars,
        )

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        roc = data["close"].pct_change(params.roc_bars)
        acceleration = roc.diff(params.accel_lag)
        baseline = acceleration.rolling(params.baseline_bars).mean().shift(1)
        scale = acceleration.rolling(params.baseline_bars).std().shift(1)
        z = ((acceleration - baseline) / scale.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )

        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        result = pd.DataFrame(index=data.index)
        result["acceleration_z"] = z.fillna(0.0)
        result["sign_balance"] = (
            np.sign(data["close"].diff()).rolling(params.run_bars).sum().fillna(0.0)
        )
        result["atr"] = true_range.rolling(params.atr_bars).mean().fillna(0.0)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        z = indicators["acceleration_z"]
        balance = indicators["sign_balance"]
        required = params.run_bars - 2
        long_entry = (
            (z > params.threshold)
            & (z.shift(1).fillna(0.0) <= params.threshold)
            & (balance >= required)
        ).to_numpy()
        short_entry = (
            (z < -params.threshold)
            & (z.shift(1).fillna(0.0) >= -params.threshold)
            & (balance <= -required)
        ).to_numpy()

        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        target = 0
        held = 0
        stop = 0.0
        age = 0

        for i in range(len(data)):
            position = target
            if position != held:
                held = position
                age = 0
                if position:
                    stop = opens[i] - position * params.stop_atr * atr[i - 1]

            if position:
                age += 1
                if position * (closes[i] - stop) <= 0 or age >= params.hold_bars:
                    target = 0
            elif long_entry[i]:
                target = 1
            elif short_entry[i]:
                target = -1
            targets[i] = target

        df = pd.DataFrame(index=data.index)
        df["signal"] = targets
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
