from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_window: int = 30
    atr_window: int = 14
    entry_z: float = 1.5
    stop_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791298827"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.ma_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        mean = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        deviation = close.rolling(params.ma_window, min_periods=params.ma_window).std()
        z = ((close - mean) / deviation.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["z"] = z.fillna(0.0)

        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = true_range.rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().fillna(0.0)

        change = z.diff()
        ready = out["atr"] > 0
        out["long_entry"] = (
            (z.shift(2) <= -params.entry_z)
            & (change.shift(1) > 0)
            & (change > 0)
            & (z < 0)
            & ready
        ).fillna(False)
        out["short_entry"] = (
            (z.shift(2) >= params.entry_z)
            & (change.shift(1) < 0)
            & (change < 0)
            & (z > 0)
            & ready
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        z = indicators["z"].to_numpy()
        long_entry = indicators["long_entry"].to_numpy()
        short_entry = indicators["short_entry"].to_numpy()
        position = 0
        entry_price = 0.0
        stop = 0.0
        held = 0

        for i in range(len(data)):
            filled_position = decisions[i - 1] if i else 0
            if filled_position != position:
                position = filled_position
                held = 0
                if position:
                    entry_price = opens[i]
                    distance = params.stop_atr * atr[i - 1]
                    stop = entry_price - position * distance

            if position:
                held += 1
                stopped = closes[i] <= stop if position == 1 else closes[i] >= stop
                reverted = z[i] >= 0 if position == 1 else z[i] <= 0
                decisions[i] = 0 if stopped or reverted or held >= params.max_hold else position
            elif long_entry[i]:
                decisions[i] = 1
            elif short_entry[i]:
                decisions[i] = -1

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
