from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    visibility_window: int = 20
    percentile_window: int = 126
    bridge_percentile: float = 0.85
    peak_window: int = 63
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790922258"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if (params.visibility_window < 3 or params.percentile_window < 2
                or params.peak_window < 2 or params.atr_window < 1
                or params.max_hold < 1 or not 0 < params.bridge_percentile < 1
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        return int(max(params.visibility_window + params.percentile_window + 2,
                       params.peak_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).replace([np.inf, -np.inf], np.nan).max(axis=1, skipna=False)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        interior_max = close.shift(1).to_numpy(copy=True)
        current = close.to_numpy()
        span = np.zeros(len(data), dtype=float)
        # Each lag adds an older endpoint to the trailing horizontal-visibility graph.
        for lag in range(2, params.visibility_window + 1):
            older = close.shift(lag).to_numpy()
            visible = (interior_max < np.minimum(current, older)) & (older > current)
            if lag >= 3:
                span = np.where(visible, np.maximum(span, lag), span)
            interior_max = np.maximum(interior_max, older)
        complete = close.rolling(params.visibility_window + 1).count().eq(params.visibility_window + 1)
        bridge_span = pd.Series(span / params.visibility_window, index=data.index).where(complete)
        threshold = bridge_span.shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.bridge_percentile)
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        drawdown = 1.0 - close / peak
        entry = (
            (bridge_span > threshold)
            & (bridge_span.shift(1) <= threshold.shift(1))
            & (close > previous)
            & (drawdown > 0)
            & np.isfinite(atr)
            & (atr > 0)
        ).fillna(False)
        return pd.DataFrame({
            "bridge_span": bridge_span,
            "bridge_threshold": threshold,
            "drawdown": drawdown,
            "atr": atr,
            "entry": entry,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        self.warmup_bars(params)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        desired = np.zeros(len(data), dtype=int)
        active = False
        high_water = -np.inf
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                # The entry decision fills this bar; initialize from its closing price.
                age += 1
                if not valid:
                    active = False
                else:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if close[i] <= stop or age >= params.max_hold:
                        active = False
            elif valid and entry[i]:
                active = True
                high_water = -np.inf
                stop = -np.inf
                age = 0
            desired[i] = int(active)
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100, 105, 104, 102, 101],
        "high": [101, 106, 105, 103, 102],
        "low": [99, 104, 103, 101, 100],
        "close": [100, 105, 104, 102, 101],
        "volume": [1000] * 5,
    }, index=index)
    indicators = pd.DataFrame({"atr": [1, 1, 10, 10, 10],
                               "entry": [True, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, indicators, None, Params(trail_atr=2.0)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    expired = strategy.generate_signals(data, indicators, None,
                                        Params(trail_atr=2.0, max_hold=2)).data
    assert expired["signal"].tolist() == [0, 1, 1, 0, 0]
    valley = data.copy()
    valley["close"] = [10, 8, 7, 9, 9]
    p = Params(visibility_window=3, percentile_window=2, peak_window=3, atr_window=2)
    features = strategy.indicators(valley, p)
    assert features["bridge_span"].iloc[3] == 1.0
    assert features["bridge_span"].iloc[4] == 0.0
    assert not features["entry"].any()


if __name__ == "__main__":
    _self_check()
