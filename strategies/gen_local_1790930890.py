from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    position_window: int = 20
    percentile_window: int = 126
    entry_quantile: float = 0.30
    atr_window: int = 14
    trail_atr: float = 2.0
    max_hold: int = 10

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError("fractional_order must lie between zero and one")
        if min(self.fractional_window, self.position_window,
               self.percentile_window, self.atr_window) < 2:
            raise ValueError("indicator windows must be at least two")
        if not 0.0 < self.entry_quantile < 1.0:
            raise ValueError("entry_quantile must lie between zero and one")
        if self.trail_atr <= 0.0 or self.max_hold < 1:
            raise ValueError("trail_atr and max_hold must be positive")


class GeneratedStrategy(BaseStrategy[TideParams]):
    strategy_id = "gen_local_1790930890"

    @classmethod
    def params_type(cls):
        return TideParams

    def warmup_bars(self, params: TideParams) -> int:
        return int(max(
            params.fractional_window + params.position_window
            + params.percentile_window + 2,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: TideParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        # Fixed-width causal fractional difference: w_j = -w_(j-1)*(d-j+1)/j.
        frac = close.copy()
        weight = 1.0
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            frac = frac + weight * close.shift(lag)
        out["fractional_close"] = frac
        out["fractional_low"] = frac.rolling(params.position_window).min()
        out["fractional_high"] = frac.rolling(params.position_window).max()
        width = out["fractional_high"] - out["fractional_low"]
        out["position"] = (frac - out["fractional_low"]) / width.replace(0.0, np.nan)
        out["entry_threshold"] = (
            out["position"].shift(1).rolling(params.percentile_window)
            .quantile(params.entry_quantile)
        )
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["entry"] = (
            (out["position"] > out["entry_threshold"])
            & (out["position"].shift(1) <= out["entry_threshold"].shift(1))
            & (frac > frac.shift(1))
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: TideParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = stop = -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0.0:
                active = False
                held = 0
                high_water = stop = -np.inf
                continue
            if active:
                # This is a filled holding bar; the decision bar is excluded.
                held += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or held >= params.max_hold:
                    active = False
                    held = 0
                    high_water = stop = -np.inf
                    continue
            elif entries[i]:
                active = True
                held = 0
                high_water = stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = TideParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [100.0, 104.0, 103.0, 101.0, 100.0]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 10.0, 10.0, 10.0],
        "entry": [True, False, False, False, False],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    params.max_hold = 2
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
