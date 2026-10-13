from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDebtParams:
    rank_window: int = 60
    stack_bars: int = 3
    tail_fraction: float = 0.15
    repair_fraction: float = 0.40
    atr_period: int = 14
    trail_atr: float = 0.60
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[GapDebtParams]):
    strategy_id = "gen_local_1790981619"

    @classmethod
    def params_type(cls):
        return GapDebtParams

    def warmup_bars(self, params):
        return int(max(params.rank_window + params.stack_bars, params.atr_period + 1))

    def indicators(self, data, params):
        if min(params.rank_window, params.stack_bars, params.atr_period) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0.0 < params.tail_fraction < 0.5 and 0.0 < params.repair_fraction <= 1.0):
            raise ValueError("Invalid percentile or repair fraction.")
        if params.trail_atr <= 0.0 or params.max_hold not in (1, 2):
            raise ValueError("Positive trail distance and one- or two-day holding required.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = data["open"] / out["previous_close"].replace(0.0, np.nan) - 1.0
        out["gap_stack"] = out["gap"].rolling(params.stack_bars, min_periods=params.stack_bars).sum()
        out["stack_rank"] = out["gap_stack"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["gap_points"] = data["open"] - out["previous_close"]
        out["repair"] = (data["open"] - data["close"]) / out["gap_points"].replace(0.0, np.nan)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        valid = indicators[["stack_rank", "gap_stack", "gap", "repair", "atr"]].notna().all(axis=1)
        valid &= indicators["atr"] > 0.0
        absorption = indicators["repair"] >= params.repair_fraction
        long_setup = valid & absorption & (indicators["stack_rank"] <= params.tail_fraction)
        long_setup &= (indicators["gap_stack"] < 0.0) & (indicators["gap"] < 0.0)
        short_setup = valid & absorption & (indicators["stack_rank"] >= 1.0 - params.tail_fraction)
        short_setup &= (indicators["gap_stack"] > 0.0) & (indicators["gap"] > 0.0)
        setup = np.select([long_setup, short_setup], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        held = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            if position:
                held += 1
                if not (np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0.0):
                    position = 0
                else:
                    # The first close belongs to the entry session after the signal shift.
                    high_water = closes[i] if np.isnan(high_water) else max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    # The required high-water stop also covers shorts on a favorable decline.
                    if closes[i] < stop or held >= params.max_hold or setup[i] == -position:
                        position = 0
                raw[i] = position
                continue
            if setup[i]:
                position = int(setup[i])
                held = 0
                high_water = np.nan
                stop = -np.inf
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame(
        {"open": [100., 100., 101., 103., 100., 99., 100.],
         "high": [101., 102., 104., 104., 101., 101., 101.],
         "low": [99., 99., 100., 99., 98., 98., 99.],
         "close": [100., 101., 103., 100., 99., 100., 100.],
         "volume": [1000.] * 7}, index=index
    )
    ind = pd.DataFrame(
        {"stack_rank": [0.1, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5],
         "gap_stack": [-0.02, 0., 0., 0., 0.02, 0., 0.],
         "gap": [-0.01, 0., 0., 0., 0.01, 0., 0.],
         "repair": [0.6, 0., 0., 0., 0.6, 0., 0.],
         "atr": [1.] * 7}, index=index
    )
    strategy = GeneratedStrategy()
    params = GapDebtParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1]
    assert result["size"].gt(0.0).all()
    ind.loc[index[0], "repair"] = 0.0
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].iloc[1] == 0
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert strategy.generate_signals(data, derived, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
