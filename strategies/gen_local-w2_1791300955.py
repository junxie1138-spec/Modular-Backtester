from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    analog_span: int = 126
    breakeven_pct: float = 0.02


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791300955"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return params.analog_span + 22

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        close = data["close"]
        move = close.pct_change(fill_method=None)
        relative_volume = data["volume"].div(data["volume"].rolling(20).mean().replace(0, np.nan))
        shock = (move * relative_volume).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        confirmed = (relative_volume >= 1.25) & (move.abs() >= 0.005)

        distances = []
        outcomes = []
        for lag in range(8, params.analog_span + 1):
            distance = sum((shock.shift(j) - shock.shift(lag + j)).abs() for j in range(3))
            distance = distance.where(confirmed.shift(lag).fillna(False), np.inf)
            outcome = close.shift(lag - 7).div(close.shift(lag)).sub(1)
            distances.append(distance.to_numpy(dtype=float))
            outcomes.append(outcome.to_numpy(dtype=float))

        distance_array = np.column_stack(distances)
        outcome_array = np.column_stack(outcomes)
        distance_array[~np.isfinite(distance_array) | ~np.isfinite(outcome_array)] = np.inf
        nearest = np.argpartition(distance_array, 4, axis=1)[:, :5]
        chosen_distance = np.take_along_axis(distance_array, nearest, axis=1)
        chosen_outcome = np.take_along_axis(outcome_array, nearest, axis=1)
        valid = np.isfinite(chosen_distance)
        chosen_outcome = np.where(valid, chosen_outcome, 0.0)
        edge = chosen_outcome.mean(axis=1)
        agreement = np.sum(np.sign(chosen_outcome) == np.sign(edge)[:, None], axis=1)
        edge[(valid.sum(axis=1) < 5) | (agreement < 4)] = 0.0
        edge[:self.warmup_bars(params)] = 0.0

        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().fillna(0.0)
        return pd.DataFrame({
            "edge": edge,
            "confirmed": confirmed.fillna(False),
            "atr": atr,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        position = 0
        entry = 0.0
        entry_bar = -1
        stop = 0.0
        active_trail = False
        desired = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        edge = indicators["edge"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].to_numpy(dtype=bool)

        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry = opens[i]
                    stop = entry - position * 2.0 * atr[i - 1]
                    active_trail = False

                hit = lows[i] <= stop if position == 1 else highs[i] >= stop
                if hit or i - entry_bar >= 9:
                    position = 0
                    desired[i] = 0
                    continue

                if position == 1:
                    active_trail = active_trail or highs[i] >= entry * (1.0 + params.breakeven_pct)
                    if active_trail:
                        stop = max(stop, entry, highs[i] - 2.0 * atr[i])
                else:
                    active_trail = active_trail or lows[i] <= entry * (1.0 - params.breakeven_pct)
                    if active_trail:
                        stop = min(stop, entry, lows[i] + 2.0 * atr[i])
            elif confirmed[i] and abs(edge[i]) >= 0.008 and atr[i] > 0:
                position = 1 if edge[i] > 0 else -1
                entry_bar = i + 1

            desired[i] = position

        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
