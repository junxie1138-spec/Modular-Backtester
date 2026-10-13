from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeAnalogParams:
    ma_length: int = 20
    entry_z: float = 1.25


class GeneratedStrategy(BaseStrategy[ShapeAnalogParams]):
    strategy_id = "gen_local-w2_1791301271"

    @classmethod
    def params_type(cls):
        return ShapeAnalogParams

    @classmethod
    def warmup_bars(cls, params: ShapeAnalogParams) -> int:
        return max(params.ma_length + 92, 14)

    def indicators(self, data: pd.DataFrame, params: ShapeAnalogParams) -> pd.DataFrame:
        close = data["close"]
        mean = close.rolling(params.ma_length).mean()
        deviation = close.rolling(params.ma_length).std(ddof=0)
        z = (close - mean) / deviation.replace(0, np.nan)

        distances = []
        outcomes = []
        for lag in range(6, 91):
            distance = sum((z.shift(step) - z.shift(lag + step)) ** 2 for step in range(3))
            distances.append(distance.fillna(np.inf).to_numpy())
            outcomes.append((close.shift(lag - 4) / close.shift(lag) - 1).fillna(0).to_numpy())

        distance_array = np.column_stack(distances)
        outcome_array = np.column_stack(outcomes)
        nearest = np.argpartition(distance_array, 6, axis=1)[:, :7]
        matched_returns = np.take_along_axis(outcome_array, nearest, axis=1)
        analog_return = matched_returns.mean(axis=1)
        analog_vote = np.sign(matched_returns).mean(axis=1)

        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(14).mean().fillna(0)

        return pd.DataFrame(
            {
                "z": z.fillna(0),
                "analog_return": analog_return,
                "analog_vote": analog_vote,
                "atr": atr,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShapeAnalogParams,
    ) -> SignalFrame:
        opening = data["open"].to_numpy()
        high = data["high"].to_numpy()
        low = data["low"].to_numpy()
        z = indicators["z"].to_numpy()
        forecast = indicators["analog_return"].to_numpy()
        vote = indicators["analog_vote"].to_numpy()
        atr = indicators["atr"].to_numpy()
        raw = np.zeros(len(data), dtype=int)

        position = 0
        entry = 0.0
        stop = np.nan
        armed = False
        held = 0
        for i in range(len(data)):
            if position:
                if held == 0:
                    entry = opening[i]
                held += 1
                if position == 1:
                    if armed and low[i] <= stop:
                        position = 0
                    else:
                        if high[i] >= entry * 1.01:
                            armed = True
                        if armed:
                            stop = max(entry if np.isnan(stop) else stop, entry, high[i] - 1.5 * atr[i])
                else:
                    if armed and high[i] >= stop:
                        position = 0
                    else:
                        if low[i] <= entry * 0.99:
                            armed = True
                        if armed:
                            stop = min(entry if np.isnan(stop) else stop, entry, low[i] + 1.5 * atr[i])
                if held >= 5:
                    position = 0
            elif (
                i >= self.warmup_bars(params)
                and abs(z[i]) >= params.entry_z
                and abs(forecast[i]) >= 0.002
                and abs(vote[i]) >= 0.4
                and forecast[i] * vote[i] > 0
            ):
                position = 1 if forecast[i] > 0 else -1
                held = 0
                armed = False
                stop = np.nan
            raw[i] = position

        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
