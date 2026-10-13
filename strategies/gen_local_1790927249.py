from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSupplyParams:
    lookback: int = 60
    atr_period: int = 14
    gap_quantile: float = 0.80
    wick_quantile: float = 0.70
    trail_atr: float = 0.75
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[GapSupplyParams]):
    strategy_id = "gen_local_1790927249"

    @classmethod
    def params_type(cls) -> type[GapSupplyParams]:
        return GapSupplyParams

    def warmup_bars(self, params: GapSupplyParams) -> int:
        return int(params.atr_period + params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: GapSupplyParams) -> pd.DataFrame:
        if params.lookback < 2 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive; lookback must be at least two.")
        if not (0 < params.gap_quantile < 1 and 0 < params.wick_quantile < 1):
            raise ValueError("Quantiles must lie strictly between zero and one.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0 or params.hold_days < 1:
            raise ValueError("Trailing distance and holding period must be positive.")
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior_close).abs(),
            (data["low"] - prior_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        scale = out["atr"].where(out["atr"] > 0)
        out["upper_wick"] = (data["high"] - data[["open", "close"]].max(axis=1)).clip(lower=0)
        out["wick_strength"] = out["upper_wick"] / scale
        out["prior_wick_strength"] = out["wick_strength"].shift(1)
        out["wick_threshold"] = out["wick_strength"].shift(2).rolling(params.lookback, min_periods=params.lookback).quantile(params.wick_quantile)
        out["gap_strength"] = (data["open"] - prior_close) / scale.shift(1)
        out["gap_threshold"] = out["gap_strength"].shift(1).rolling(params.lookback, min_periods=params.lookback).quantile(params.gap_quantile)
        out["prior_high"] = data["high"].shift(1)
        out["entry"] = (
            (out["prior_wick_strength"] > out["wick_threshold"])
            & (out["gap_strength"] > out["gap_threshold"])
            & (out["gap_strength"] > 0)
            & (data["open"] > out["prior_high"])
            & (data["close"] > data["open"])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapSupplyParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                if not valid:
                    active = False
                else:
                    # This is a held session: yesterday's target filled at today's open.
                    age += 1
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if close[i] <= stop or age >= params.hold_days:
                        active = False
            elif entry[i] and valid:
                # Start tracking closes only after the next-open entry.
                active = True
                age = 0
                high_water = stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [100.0, 100.0, 98.0, 98.0, 98.0]}, index=index)
    ind = pd.DataFrame({"atr": [1.0, 1.0, 10.0, 10.0, 10.0], "entry": [True, False, False, False, False]}, index=index)
    # Expanding ATR must not loosen an established stop.
    result = strategy.generate_signals(data, ind, None, GapSupplyParams(hold_days=3)).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, ind, None, GapSupplyParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
