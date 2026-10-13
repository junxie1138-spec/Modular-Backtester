from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    frac_order: float = 0.4
    frac_window: int = 40
    norm_window: int = 60
    vol_window: int = 20
    atr_window: int = 14
    entry_z: float = 0.35
    reset_z: float = -0.75
    atr_multiple: float = 3.0
    hold_bars: int = 18
    target_vol: float = 0.12

    def __post_init__(self):
        if min(self.frac_window, self.norm_window, self.vol_window, self.atr_window) < 2 or self.hold_bars < 1:
            raise ValueError("Lookbacks must be at least two and holding period positive.")
        if not (0.0 < self.frac_order < 1.0 and self.reset_z < 0.0 < self.entry_z):
            raise ValueError("Invalid fractional order or hysteresis thresholds.")
        if not (np.isfinite(self.atr_multiple) and np.isfinite(self.target_vol) and self.atr_multiple > 0.0 and self.target_vol > 0.0):
            raise ValueError("ATR multiplier and volatility target must be finite and positive.")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791259922"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(200, params.frac_window + params.norm_window - 1, params.vol_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["previous_close"] = close.shift(1)
        out["returns"] = (close / out["previous_close"].replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["annual_vol"] = out["returns"].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        # Finite causal expansion of (1 - lag)**d on close levels.
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.frac_window):
            weight *= (lag - 1.0 - params.frac_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        out["fractional_mean"] = fractional.rolling(params.norm_window, min_periods=params.norm_window).mean()
        out["fractional_std"] = fractional.rolling(params.norm_window, min_periods=params.norm_window).std(ddof=0)
        denominator = out["fractional_std"].where(out["fractional_std"] > 1e-12)
        out["fractional_z"] = ((fractional - out["fractional_mean"]) / denominator).replace([np.inf, -np.inf], np.nan)
        # The simulator sizes only on entry; this is an entry-time risk budget.
        out["entry_size"] = (params.target_vol / out["annual_vol"].clip(lower=1e-4)).clip(lower=0.01, upper=1.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        armed = False
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        values = indicators[["fractional_z", "ma200", "atr", "annual_vol"]].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        for i, (z, ma, atr, vol) in enumerate(values):
            close = closes[i]
            valid = np.isfinite([close, z, ma, atr, vol]).all() and close > 0.0 and atr > 0.0 and vol >= 0.0
            regime = valid and close > ma
            if active:
                if valid:
                    # First update is the first holding bar, after the entry decision.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.atr_multiple * atr)
                if not regime or close <= stop or i - entry_bar >= params.hold_bars:
                    active = False
                    armed = False
                else:
                    raw[i] = 1
                continue
            if not regime:
                armed = False
                continue
            if z <= params.reset_z:
                armed = True
            if armed and z >= params.entry_z:
                active = True
                armed = False
                entry_bar = i
                high_water = -np.inf
                stop = -np.inf
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["entry_size"].shift(1).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.01, upper=1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [100.0, 100.0, 102.0, 108.0, 107.0, 103.0, 102.0, 101.0]}, index=index)
    indicators = pd.DataFrame({
        "fractional_z": [-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "ma200": 90.0,
        "atr": [1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0],
        "annual_vol": 0.2,
        "entry_size": 0.6,
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, GeneratedParams()).data
    # ATR expansion cannot loosen the 105 stop; the exit and size are delayed once.
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert result["size"].tolist() == [1.0] + [0.6] * 7
