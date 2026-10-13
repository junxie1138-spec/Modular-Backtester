from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSlopeParams:
    trend_span: int = 8
    atr_window: int = 14
    rank_window: int = 126
    innovation_percentile: float = 0.85
    arm_gain: float = 0.015
    trail_atr: float = 2.0
    holding_days: int = 4


class GeneratedStrategy(BaseStrategy[GapSlopeParams]):
    strategy_id = "gen_local-w2_1791296644"

    @classmethod
    def params_type(cls):
        return GapSlopeParams

    @classmethod
    def warmup_bars(cls, params: GapSlopeParams) -> int:
        return params.trend_span + params.atr_window + params.rank_window + 2

    def indicators(self, data: pd.DataFrame, params: GapSlopeParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        slope = data["close"].diff().ewm(
            span=params.trend_span, min_periods=params.trend_span
        ).mean()
        innovation = (data["open"] - previous_close - slope.shift(1)).div(
            atr.shift(1).replace(0, np.nan)
        )
        threshold = innovation.shift(1).rolling(
            params.rank_window, min_periods=params.rank_window
        ).quantile(params.innovation_percentile)
        ready = threshold.notna() & atr.notna() & slope.shift(1).notna()
        return pd.DataFrame(
            {
                "atr": atr.fillna(0.0),
                "prior_slope": slope.shift(1).fillna(0.0),
                "innovation": innovation.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "threshold": threshold.fillna(0.0),
                "ready": ready,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSlopeParams,
    ) -> SignalFrame:
        eligible = (
            indicators["ready"]
            & (indicators["prior_slope"] < 0)
            & (indicators["innovation"] > indicators["threshold"])
            & (data["open"] > data["close"].shift(1))
            & (data["close"] >= data["open"])
        ).fillna(False).to_numpy()
        opens = data["open"].to_numpy()
        lows = data["low"].to_numpy()
        closes = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        raw = np.zeros(len(data), dtype=int)
        held = False
        pending_entry = False
        armed = False
        entry = 0.0
        stop = 0.0
        days = 0

        for i in range(len(data)):
            if pending_entry:
                held = True
                pending_entry = False
                armed = False
                entry = float(opens[i])
                stop = entry - params.trail_atr * float(atr[i - 1])
                days = 0

            if held:
                days += 1
                if lows[i] <= stop or days >= params.holding_days:
                    held = False
                    raw[i] = 0
                else:
                    if closes[i] >= entry * (1.0 + params.arm_gain):
                        armed = True
                    if armed:
                        stop = max(stop, entry, closes[i] - params.trail_atr * atr[i])
                    raw[i] = 1
            elif eligible[i] and atr[i] > 0:
                raw[i] = 1
                pending_entry = True

        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
