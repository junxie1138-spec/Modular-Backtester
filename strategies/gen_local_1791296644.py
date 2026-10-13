from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    regime_window: int = 20
    volume_window: int = 20
    atr_window: int = 14
    expansion_cut: float = 1.15
    contraction_cut: float = 0.85
    edge_threshold: float = 0.15
    pressure_threshold: float = 0.20
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791296644"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.regime_window + 1, params.volume_window + 1, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        high, low, close = data["high"], data["low"], data["close"]
        bar_range = high - low
        previous_close = close.shift(1)
        true_range = pd.concat(
            [bar_range, (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        prior_range = bar_range.rolling(params.regime_window, min_periods=params.regime_window).mean().shift(1)
        prior_volume = data["volume"].rolling(params.volume_window, min_periods=params.volume_window).mean().shift(1)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr.fillna(0.0)
        out["range_ratio"] = bar_range.div(prior_range.replace(0, np.nan)).fillna(0.0)
        out["edge"] = ((high + low - high.shift(1) - low.shift(1)) / 2).div(atr.replace(0, np.nan)).fillna(0.0)
        out["pressure"] = ((2 * close - high - low).div(bar_range.replace(0, np.nan)) * data["volume"].div(prior_volume.replace(0, np.nan))).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        opening = data["open"].to_numpy()
        atr = indicators["atr"].to_numpy()
        ratio = indicators["range_ratio"].to_numpy()
        edge = indicators["edge"].to_numpy()
        pressure = indicators["pressure"].to_numpy()
        decisions = np.zeros(len(data), dtype=int)
        previous_active = 0
        entry_bar = -1
        entry_price = 0.0
        stop_distance = 0.0

        for i in range(len(data)):
            active = int(decisions[i - 1]) if i else 0
            if active and not previous_active:
                entry_bar = i
                entry_price = float(opening[i])
                stop_distance = params.stop_atr * float(atr[i - 1])

            decision = active
            if active:
                stopped = (active == 1 and close[i] <= entry_price - stop_distance) or (active == -1 and close[i] >= entry_price + stop_distance)
                if stopped or i - entry_bar + 1 >= params.hold_bars:
                    decision = 0
            elif i >= self.warmup_bars(params) and atr[i] > 0:
                expanding = ratio[i] >= params.expansion_cut
                contracting = 0 < ratio[i] <= params.contraction_cut
                if pressure[i] > params.pressure_threshold and ((expanding and edge[i] > params.edge_threshold) or (contracting and edge[i] < -params.edge_threshold)):
                    decision = 1
                elif pressure[i] < -params.pressure_threshold and ((expanding and edge[i] < -params.edge_threshold) or (contracting and edge[i] > params.edge_threshold)):
                    decision = -1

            decisions[i] = decision
            previous_active = active

        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
