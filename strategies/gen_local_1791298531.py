from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    entry_z: float = -0.8
    min_analog_return: float = 0.01


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791298531"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return 300

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        close = data["close"]
        mean = close.rolling(20).mean()
        z = ((close - mean) / close.rolling(20).std().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        true_range = pd.concat([data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()
        future_return = close.shift(-15).div(close).sub(1)
        offsets = (0, 3, 6, 9)
        shape = [z.shift(offset) for offset in offsets]
        distances = []
        outcomes = []
        for lag in range(20, 272):
            distance = sum((part - z.shift(lag + offset)).pow(2) for part, offset in zip(shape, offsets))
            outcome = future_return.shift(lag)
            valid = z.shift(lag).lt(-0.25) & outcome.notna()
            distances.append(distance.where(valid, np.inf).fillna(np.inf).to_numpy())
            outcomes.append(outcome.fillna(0).to_numpy())
        distances = np.stack(distances)
        outcomes = np.stack(outcomes)
        nearest = np.argpartition(distances, 4, axis=0)[:5]
        selected_distances = np.take_along_axis(distances, nearest, axis=0)
        selected_returns = np.take_along_axis(outcomes, nearest, axis=0)
        count = np.isfinite(selected_distances).sum(axis=0)
        analog_return = np.divide(np.where(np.isfinite(selected_distances), selected_returns, 0).sum(axis=0), count, out=np.zeros(len(data)), where=count > 0)
        out = pd.DataFrame(index=data.index)
        out["z"] = z.fillna(0)
        out["atr"] = atr.fillna(0)
        out["analog_return"] = analog_return
        out["analog_count"] = count
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy()
        z = indicators["z"].to_numpy()
        atr = indicators["atr"].to_numpy()
        analog_return = indicators["analog_return"].to_numpy()
        analog_count = indicators["analog_count"].to_numpy()
        holding = False
        active = False
        entry = stop = 0.0
        entry_bar = 0
        for i, price in enumerate(close):
            if holding:
                if not active and price >= entry * 1.03:
                    active = True
                    stop = max(stop, entry)
                if active:
                    stop = max(stop, price - 2.5 * atr[i])
                if price <= stop or i - entry_bar >= 20:
                    holding = False
            elif i >= self.warmup_bars(params) and z[i] <= params.entry_z and analog_count[i] == 5 and analog_return[i] >= params.min_analog_return and atr[i] > 0:
                holding = True
                active = False
                entry = price
                stop = entry - 2.5 * atr[i]
                entry_bar = i
            raw[i] = int(holding)
        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
