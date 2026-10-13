from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MirrorParams:
    pattern: int = 12
    percentile_window: int = 126
    match_quantile: float = 0.25
    hold_bars: int = 18
    breakeven_pct: float = 0.025
    atr_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[MirrorParams]):
    strategy_id = "gen_local-w2_1791224430"

    @classmethod
    def params_type(cls):
        return MirrorParams

    @classmethod
    def warmup_bars(cls, params: MirrorParams) -> int:
        return 252 + params.pattern + params.percentile_window + 1

    def indicators(self, data: pd.DataFrame, params: MirrorParams) -> pd.DataFrame:
        close = data["close"]
        returns = close.pct_change()
        scale = returns.pow(2).rolling(params.pattern).mean().pow(0.5).replace(0, np.nan)
        distances = {}
        outcomes = {}
        for lag in (63, 126, 189, 252):
            distance = pd.Series(0.0, index=data.index)
            for j in range(params.pattern):
                current = returns.shift(j) / scale
                mirrored = returns.shift(lag + params.pattern - 1 - j) / scale.shift(lag)
                distance = distance + (current - mirrored).abs()
            distances[lag] = distance / params.pattern
            outcomes[lag] = close.shift(lag - params.hold_bars) / close.shift(lag) - 1

        distance_frame = pd.DataFrame(distances).replace([np.inf, -np.inf], np.nan)
        outcome_frame = pd.DataFrame(outcomes).replace([np.inf, -np.inf], np.nan)
        weights = 1 / (distance_frame + 1e-4)
        weights = weights.where(outcome_frame.notna(), 0).fillna(0)
        forecast = (weights * outcome_frame.fillna(0)).sum(axis=1) / weights.sum(axis=1).replace(0, np.nan)
        best_distance = distance_frame.min(axis=1)
        threshold = best_distance.shift(1).rolling(params.percentile_window).quantile(params.match_quantile)

        previous_close = close.shift(1)
        true_range = pd.concat(
            [(data["high"] - data["low"]).abs(),
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(14).mean()
        return pd.DataFrame({
            "distance": best_distance.fillna(np.inf),
            "threshold": threshold.fillna(-1.0),
            "forecast": forecast.fillna(0.0),
            "atr": atr.fillna(0.0),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MirrorParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        actual = 0
        entry = 0.0
        stop = 0.0
        active = False
        age = 0
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        distance = indicators["distance"].to_numpy()
        threshold = indicators["threshold"].to_numpy()
        forecast = indicators["forecast"].to_numpy()
        atr = indicators["atr"].to_numpy()

        for i in range(len(data)):
            filled = raw[i - 1] if i else 0
            if filled != actual:
                actual = filled
                entry = opens[i]
                stop = -np.inf if actual == 1 else np.inf
                active = False
                age = 0
            elif actual:
                age += 1

            if actual:
                breached = active and (closes[i] <= stop if actual == 1 else closes[i] >= stop)
                if breached or age >= params.hold_bars:
                    raw[i] = 0
                    continue
                gain = actual * (closes[i] / entry - 1)
                if gain >= params.breakeven_pct:
                    active = True
                if active and atr[i] > 0:
                    trail = closes[i] - actual * params.atr_mult * atr[i]
                    stop = max(stop, entry, trail) if actual == 1 else min(stop, entry, trail)
                raw[i] = actual
            elif i >= self.warmup_bars(params) and distance[i] <= threshold[i]:
                raw[i] = int(np.sign(forecast[i]))

        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
