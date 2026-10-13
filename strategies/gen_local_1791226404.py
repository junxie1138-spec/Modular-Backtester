from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpiryParams:
    shock_z: float = 1.2
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ExpiryParams]):
    strategy_id = "gen_local_1791226404"

    @classmethod
    def params_type(cls) -> type[ExpiryParams]:
        return ExpiryParams

    def warmup_bars(self, params: ExpiryParams) -> int:
        return 200

    def indicators(self, data: pd.DataFrame, params: ExpiryParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0))
        weight = 1.0
        fd = log_close.copy()
        for lag in range(1, 16):
            weight *= (lag - 1 - 0.4) / lag
            fd = fd + weight * log_close.shift(lag)

        mean = fd.rolling(30, min_periods=30).mean()
        std = fd.rolling(30, min_periods=30).std()
        z = (fd - mean) / std.where(std > 0)
        previous_close = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        ma200 = close.rolling(200, min_periods=200).mean()

        out = pd.DataFrame(index=data.index)
        out["ready"] = np.isfinite(z) & np.isfinite(atr) & (atr > 0) & np.isfinite(ma200) & np.isfinite(close)
        out["z"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["atr"] = atr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["ma200"] = ma200.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["expiry_wednesday"] = (data.index.dayofweek == 2) & (data.index.day >= 13) & (data.index.day <= 19)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ExpiryParams,
    ) -> SignalFrame:
        if not np.isfinite(params.shock_z) or params.shock_z <= 0:
            raise ValueError("shock_z must be positive and finite")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")

        close = data["close"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma200 = indicators["ma200"].to_numpy(dtype=float)
        ready = indicators["ready"].to_numpy(dtype=bool)
        expiry = indicators["expiry_wednesday"].to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        age = 0
        high_water = low_water = stop = np.nan

        for i in range(len(data)):
            if side:
                age += 1
                if not ready[i]:
                    side = 0
                elif side == 1:
                    high_water = close[i] if age == 1 else max(high_water, close[i])
                    candidate = high_water - params.trail_atr * atr[i]
                    stop = candidate if age == 1 else max(stop, candidate)
                    if close[i] <= stop or age >= params.hold_bars:
                        side = 0
                else:
                    low_water = close[i] if age == 1 else min(low_water, close[i])
                    candidate = low_water + params.trail_atr * atr[i]
                    stop = candidate if age == 1 else min(stop, candidate)
                    if close[i] >= stop or age >= params.hold_bars:
                        side = 0
            elif ready[i] and expiry[i] and i >= self.warmup_bars(params) - 1:
                if close[i] > ma200[i] and z[i] <= -params.shock_z:
                    side = 1
                elif close[i] < ma200[i] and z[i] >= params.shock_z:
                    side = -1
                if side:
                    age = 0
                    high_water = low_water = stop = np.nan
            decisions[i] = side

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2024-01-01", periods=4)
    sample = pd.DataFrame({"close": [100.0, 102.0, 99.0, 100.0]}, index=index)
    features = pd.DataFrame({
        "z": [-2.0, 0.0, 0.0, 0.0],
        "atr": [1.0] * 4,
        "ma200": [90.0] * 4,
        "ready": [True] * 4,
        "expiry_wednesday": [True, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    strategy.warmup_bars = lambda params: 1
    assert strategy.generate_signals(sample, features, None, ExpiryParams()).data["signal"].tolist() == [0, 1, 1, 0]
