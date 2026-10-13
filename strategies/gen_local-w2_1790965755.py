from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnBreadthParams:
    breadth_window: int = 5
    vol_window: int = 20
    atr_window: int = 14
    min_strength: float = 0.15
    target_daily_vol: float = 0.008
    breakeven_pct: float = 0.006
    trail_atr: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ReturnBreadthParams]):
    strategy_id = "gen_local-w2_1790965755"

    @classmethod
    def params_type(cls):
        return ReturnBreadthParams

    def warmup_bars(self, params):
        return int(max(params.breadth_window, params.vol_window, params.atr_window) + 1)

    def indicators(self, data, params):
        if (
            min(params.breadth_window, params.vol_window, params.atr_window) < 2
            or params.hold_bars not in (1, 2)
            or not 0.0 < params.min_strength < 1.0
            or not all(np.isfinite(x) and x > 0 for x in (
                params.target_daily_vol, params.breakeven_pct, params.trail_atr
            ))
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["return"] = data["close"].div(previous.where(previous > 0)).sub(1).replace(
            [np.inf, -np.inf], np.nan
        )
        window = out["return"].rolling(params.breadth_window, min_periods=params.breadth_window)
        out["net_return"] = window.sum()
        out["largest_gain"] = window.max().clip(lower=0)
        out["return_energy"] = out["return"].pow(2).rolling(
            params.breadth_window, min_periods=params.breadth_window
        ).sum().pow(0.5)
        out["strength"] = (
            (out["net_return"] - out["largest_gain"])
            / out["return_energy"].replace(0, np.nan)
        ).clip(0, 1).fillna(0.0)
        out["volatility"] = out["return"].rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["target_size"] = (
            params.target_daily_vol / out["volatility"].clip(lower=0.001) * out["strength"]
        ).clip(0.02, 0.95).fillna(0.02)
        out["eligible"] = (
            (out["strength"] >= params.min_strength)
            & (out["return"] > 0)
            & out["volatility"].notna()
            & out["atr"].gt(0)
        )
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": 0, "size": 0.02}, index=data.index)
        pending = False
        active = False
        armed = False
        entry = stop = peak = 0.0
        held_size = 0.02
        age = 0
        for i in range(len(data)):
            bar = data.iloc[i]
            ind = indicators.iloc[i]
            if pending:
                pending = False
                entry = float(bar["open"])
                atr_at_decision = float(indicators["atr"].iloc[i - 1])
                if not np.isfinite(entry) or entry <= 0:
                    continue
                active = True
                armed = False
                age = 0
                peak = entry
                stop = entry - params.trail_atr * atr_at_decision
            if active:
                age += 1
                # The existing stop is tested before today's close can raise it.
                if (
                    not np.isfinite(bar["low"])
                    or not np.isfinite(bar["close"])
                    or not np.isfinite(ind["atr"])
                    or bar["low"] <= stop
                    or age >= params.hold_bars
                ):
                    active = False
                    continue
                peak = max(peak, float(bar["close"]))
                if bar["close"] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * float(ind["atr"]))
                df.iloc[i, 0] = 1
                df.iloc[i, 1] = held_size
            elif bool(ind["eligible"]):
                pending = True
                held_size = float(ind["target_size"])
                df.iloc[i, 0] = 1
                df.iloc[i, 1] = held_size
        # Decisions at this close become target holdings at the next open.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.02).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "atr": 1.0, "eligible": [True, False, False, False, False],
        "target_size": 0.4,
    }, index=index)
    strategy = GeneratedStrategy()
    params = ReturnBreadthParams()
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert result["size"].iloc[1] == 0.4
    data.loc[index[1], "low"] = 98.0
    stopped = strategy.generate_signals(data, ind, None, params).data
    assert stopped["signal"].tolist() == [0, 1, 0, 0, 0]
    flat = strategy.indicators(data.assign(low=99.0), params)
    assert not flat["eligible"].any()
    assert strategy.warmup_bars(params) == 21


if __name__ == "__main__":
    _self_check()
