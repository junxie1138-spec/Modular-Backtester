from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapConfirmationParams:
    gap_lookback: int = 20
    gap_mult: float = 1.5
    breakeven_pct: float = 0.05
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[GapConfirmationParams]):
    strategy_id = "gen_local_1790712711"

    @classmethod
    def params_type(cls):
        return GapConfirmationParams

    @classmethod
    def warmup_bars(cls, params: GapConfirmationParams) -> int:
        return params.gap_lookback + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GapConfirmationParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        gap = data["open"] / prev_close - 1.0
        abs_gap_median = gap.abs().rolling(params.gap_lookback, min_periods=params.gap_lookback).median()
        returns = data["close"].pct_change()
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        return pd.DataFrame(
            {"gap": gap, "abs_gap_median": abs_gap_median, "return": returns, "atr": atr},
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapConfirmationParams,
    ) -> SignalFrame:
        gap = indicators["gap"]
        baseline = indicators["abs_gap_median"]
        ret = indicators["return"]
        atr = indicators["atr"]
        setup = (
            (gap.shift(2) > params.gap_mult * baseline.shift(2))
            & (ret.shift(1) > 0)
            & (ret > 0)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atr_values = atr.to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry = 0.0
        high_water = 0.0
        stop = -np.inf

        for i in range(len(data)):
            if active:
                if np.isfinite(atr_values[i]):
                    stop = max(stop, high_water - params.trail_atr * atr_values[i])
                if lows[i] <= stop:
                    active = False
                    continue
                high_water = max(high_water, closes[i])
                if closes[i] >= entry * (1.0 + params.breakeven_pct):
                    stop = max(stop, entry)
                    if np.isfinite(atr_values[i]):
                        stop = max(stop, high_water - params.trail_atr * atr_values[i])
                signals[i] = 1
            elif setup[i] and np.isfinite(closes[i]) and np.isfinite(atr_values[i]):
                active = True
                entry = closes[i]
                high_water = closes[i]
                stop = -np.inf
                signals[i] = 1

        df = pd.DataFrame({"signal": signals, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
