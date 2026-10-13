from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionBreakoutParams:
    return_window: int = 12
    selling_window: int = 3
    volume_cover: float = 1.05
    atr_window: int = 14
    trail_atr: float = 0.6
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AbsorptionBreakoutParams]):
    strategy_id = "gen_local_1790920248"

    @classmethod
    def params_type(cls) -> type[AbsorptionBreakoutParams]:
        return AbsorptionBreakoutParams

    def warmup_bars(self, params: AbsorptionBreakoutParams) -> int:
        return int(max(params.return_window + 2, params.selling_window + 2,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame,
                   params: AbsorptionBreakoutParams) -> pd.DataFrame:
        if min(params.return_window, params.selling_window,
               params.atr_window, params.max_hold_bars) < 1:
            raise ValueError("Lookbacks and holding limit must be positive.")
        if not (np.isfinite(params.volume_cover) and params.volume_cover > 0
                and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("Volume cover and ATR multiplier must be positive.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["return"] = data["close"].div(
            out["previous_close"].where(out["previous_close"] > 0)
        ).sub(1).replace([np.inf, -np.inf], np.nan)
        out["return_ceiling"] = out["return"].shift(1).rolling(
            params.return_window, min_periods=params.return_window
        ).max()
        out["selling_volume"] = data["volume"].where(
            out["return"] < 0, 0.0
        ).where(out["return"].notna() & data["volume"].ge(0))
        out["selling_budget"] = out["selling_volume"].shift(1).rolling(
            params.selling_window, min_periods=params.selling_window
        ).sum()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["return_breakout"] = (
            (out["return"] > 0) & (out["return"] > out["return_ceiling"])
        )
        out["volume_absorption"] = (
            (out["selling_budget"] > 0)
            & (data["volume"] >= params.volume_cover * out["selling_budget"])
        )
        out["entry"] = (
            out["return_breakout"] & out["volume_absorption"]
            & np.isfinite(out["return"])
            & np.isfinite(out["return_ceiling"])
            & np.isfinite(out["selling_budget"])
            & np.isfinite(data["volume"])
            & np.isfinite(out["atr"]) & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: AbsorptionBreakoutParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if valid:
                    # The first held bar is the bar after the entry decision.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                if not valid or close < stop or age >= params.max_hold_bars:
                    active = False
                    age = 0
                    high_water = stop = -np.inf
                # Do not reopen on the same close that triggered an exit.
            elif valid and entries[i]:
                active = True
                age = 0
                high_water = stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Check delayed entry, a ratcheting stop despite rising ATR, and two-day expiry.
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100, 102, 104, 103, 99, 100]}, index=index)
    ind = pd.DataFrame({"atr": [1, 1, 1, 10, 1, 1],
                        "entry": [True, False, False, False, False, False]},
                       index=index)
    params = AbsorptionBreakoutParams(trail_atr=0.5, max_hold_bars=5)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    params.max_hold_bars = 2
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert result["size"].gt(0).all()
