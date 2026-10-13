from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PathAreaParams:
    area_window: int = 16
    area_threshold: float = 0.05
    stop_atr: float = 2.5
    trend_window: int = 80
    max_hold: int = 22


class GeneratedStrategy(BaseStrategy[PathAreaParams]):
    strategy_id = "gen_local-w2_1791225378"

    @classmethod
    def params_type(cls):
        return PathAreaParams

    def warmup_bars(self, params: PathAreaParams) -> int:
        return max(params.trend_window, 65 + params.area_window)

    def indicators(self, data: pd.DataFrame, params: PathAreaParams) -> pd.DataFrame:
        close = data["close"]
        previous = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        atr = true_range.rolling(14).mean()
        returns = close.pct_change()
        volatility = returns.rolling(5).std()
        baseline = volatility.rolling(60).mean()
        x = close.pct_change(3) / (atr / close).replace(0, np.nan)
        y = volatility / baseline.replace(0, np.nan) - 1.0
        segments = x.shift(1) * y - y.shift(1) * x
        window = params.area_window
        area = (segments.rolling(window).sum()
                - x.shift(window) * y + y.shift(window) * x) / (2.0 * window)
        out = pd.DataFrame({
            "area": area,
            "momentum": x,
            "stress": y,
            "trend": close.rolling(params.trend_window).mean(),
            "atr": atr,
        }, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PathAreaParams,
    ) -> SignalFrame:
        close = data["close"]
        stressed = indicators["stress"] > 0.15
        ready = ((stressed & (indicators["area"] < -params.area_threshold))
                 | (~stressed & (indicators["area"] > params.area_threshold)
                    & (close > indicators["trend"])))
        ready = (ready & (indicators["momentum"] > 0)).to_numpy(dtype=bool)
        prices = close.to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending_fill = False
        high_water = 0.0
        stop = 0.0
        held = 0
        for i in range(self.warmup_bars(params), len(data)):
            if not active:
                if ready[i] and ready[i - 1]:
                    active = True
                    pending_fill = True
            elif pending_fill:
                high_water = prices[i]
                stop = high_water - params.stop_atr * atr[i]
                held = 1
                pending_fill = False
            else:
                high_water = max(high_water, prices[i])
                stop = max(stop, high_water - params.stop_atr * atr[i])
                held += 1
                if prices[i] <= stop or held >= params.max_hold:
                    active = False
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = PathAreaParams()
    prices = np.r_[np.full(82, 100.0), 101.0, 102.0, 103.0, 99.0, np.full(4, 99.0)]
    index = pd.date_range("2020-01-01", periods=len(prices))
    data = pd.DataFrame({"close": prices}, index=index)
    derived = pd.DataFrame({"area": 1.0, "momentum": 1.0, "stress": 0.0,
                            "trend": 90.0, "atr": 1.0}, index=index)
    result = GeneratedStrategy().generate_signals(data, derived, None, params).data
    assert result["signal"].iloc[82] == 1
    assert result["signal"].iloc[86] == 0
