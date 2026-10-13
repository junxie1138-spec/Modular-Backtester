from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fast_window: int = 5
    slow_window: int = 20
    atr_window: int = 14
    downside_share: float = 0.25
    target_vol: float = 0.10
    trail_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791221621"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.fast_window, params.slow_window, params.atr_window) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (2 <= params.fast_window < params.slow_window and
                params.atr_window >= 1 and 3 <= params.hold_days <= 5 and
                0 < params.downside_share < 0.5 and
                params.target_vol > 0 and params.trail_atr > 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        energy = out["return"].pow(2)
        downside = out["return"].clip(upper=0).pow(2)
        out["fast_energy"] = energy.rolling(params.fast_window).mean()
        out["slow_energy"] = energy.rolling(params.slow_window).mean()
        out["fast_downside"] = downside.rolling(params.fast_window).mean()
        out["slow_downside"] = downside.rolling(params.slow_window).mean()
        out["fast_share"] = out["fast_downside"] / out["fast_energy"].replace(0, np.nan)
        out["slow_share"] = out["slow_downside"] / out["slow_energy"].replace(0, np.nan)
        out["annual_vol"] = out["return"].rolling(params.slow_window).std(ddof=0) * np.sqrt(252.0)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["qualifies"] = (
            (out["return"] > 0) &
            (out["fast_share"] < params.downside_share) &
            (out["slow_share"] > 0.40) &
            (out["fast_energy"] < out["slow_energy"]) &
            (out["annual_vol"] > 0) &
            (out["atr"] > 0)
        ).fillna(False)
        out["confirmed"] = out["qualifies"] & out["qualifies"].shift(1, fill_value=False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = np.nan
        stop = -np.inf
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                held += 1
                if valid:
                    high_water = close if held == 1 else max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                if not valid or close <= stop or held >= params.hold_days:
                    active = False
            elif valid and confirmed[i]:
                active = True
                held = 0
                high_water = np.nan
                stop = -np.inf
            signals[i] = int(active)
        df["signal"] = signals
        df["size"] = (
            params.target_vol / indicators["annual_vol"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).clip(0.05, 0.95).fillna(0.05)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [100, 101, 102, 104, 102, 100, 101, 102]}, index=index)
    ind = pd.DataFrame({
        "atr": [1, 1, 1, 1, 10, 10, 10, 10],
        "annual_vol": 0.20,
        "confirmed": [False, True, False, False, False, False, False, False]
    }, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    ind["atr"] = 100.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=40))
    flat_ind = strategy.indicators(flat, params)
    assert not flat_ind["confirmed"].any()
    result = strategy.generate_signals(flat, flat_ind, None, params).data
    assert result["signal"].eq(0).all() and result["size"].gt(0).all()
