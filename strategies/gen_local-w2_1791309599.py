from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnAgeParams:
    window: int = 12
    percentile_window: int = 126
    percentile: float = 0.85
    atr_window: int = 14
    stop_atr: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ReturnAgeParams]):
    strategy_id = "gen_local-w2_1791309599"

    @classmethod
    def params_type(cls):
        return ReturnAgeParams

    def warmup_bars(self, params):
        return int(max(params.window + params.percentile_window + 2, params.atr_window + 1))

    def indicators(self, data, params):
        if min(params.window, params.percentile_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.0 < params.percentile < 1.0 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("Invalid percentile or stop multiplier.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["up"] = out["return"].clip(lower=0)
        out["down"] = -out["return"].clip(upper=0)
        clock = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        for side in ("up", "down"):
            out[side + "_mass"] = out[side].rolling(params.window).sum()
            out[side + "_moment"] = (clock * out[side]).rolling(params.window).sum()
            out[side + "_age"] = clock - out[side + "_moment"] / out[side + "_mass"].replace(0, np.nan)
        out["separation"] = (out["down_age"] - out["up_age"]) / (params.window - 1)
        out["threshold"] = out["separation"].shift(1).rolling(params.percentile_window).quantile(params.percentile)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["entry"] = (
            (out["return"] > 0)
            & (out["separation"] > out["threshold"])
            & (out["separation"].shift(1) <= out["threshold"].shift(1))
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Next-open entry uses only the ATR known at the preceding close.
                    stop = float(data["open"].iloc[i]) - params.stop_atr * float(indicators["atr"].iloc[i - 1])
                close = float(data["close"].iloc[i])
                if not np.isfinite(close) or not np.isfinite(stop) or close < stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif bool(indicators["entry"].iloc[i]) and np.isfinite(indicators["atr"].iloc[i]):
                active = True
                entry_bar = i + 1
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = ReturnAgeParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    ind = pd.DataFrame({"entry": [True, False, False, False, False], "atr": 1.0}, index=index)
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 98.0
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 0, 0, 0]
    assert strategy.indicators(data, params)["entry"].eq(False).all()
