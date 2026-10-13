from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    drawdown_window: int = 20
    trend_window: int = 8
    atr_window: int = 14
    innovation_threshold: float = 0.2
    stop_atr: float = 0.8


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790863444"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.drawdown_window, params.atr_window + 1, params.trend_window + 5))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.drawdown_window, params.trend_window, params.atr_window) < 1:
            raise ValueError("Windows must be positive.")
        if not (np.isfinite(params.innovation_threshold) and params.innovation_threshold > 0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Threshold and stop multiplier must be positive and finite.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["change"] = close.diff()
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - close.shift(1)).abs(),
             (data["low"] - close.shift(1)).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean().replace(0.0, np.nan)
        out["fast_slope"] = out["change"].rolling(3).mean()
        out["latent_slope"] = out["fast_slope"].rolling(params.trend_window).mean()
        out["expected_slope"] = out["latent_slope"].shift(1)
        out["innovation"] = out["fast_slope"] - out["expected_slope"]
        out["strength"] = out["innovation"] / out["atr"]
        out["previous_strength"] = out["strength"].shift(1)
        out["peak"] = close.rolling(params.drawdown_window).max()
        out["trough"] = close.rolling(params.drawdown_window).min()
        out["drawdown_atr"] = (out["peak"] - close) / out["atr"]
        out["recovery"] = (close - out["trough"]) / (out["peak"] - out["trough"]).replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        ready = pd.Series(np.isfinite(indicators.to_numpy()).all(axis=1), index=data.index)
        entry = (
            ready
            & (indicators["expected_slope"] < 0)
            & (indicators["fast_slope"] > 0)
            & (indicators["drawdown_atr"] >= 1.0)
            & indicators["recovery"].between(0.1, 0.6)
            & (indicators["strength"] > params.innovation_threshold)
            & (indicators["previous_strength"] <= params.innovation_threshold)
        ).fillna(False).to_numpy()
        strength = indicators["strength"].to_numpy()
        atr = indicators["atr"].to_numpy()
        close = data["close"].to_numpy()
        valid = ready.to_numpy() & np.isfinite(close) & (close > 0)
        df = pd.DataFrame({"signal": 0, "size": 0.25}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        active = False
        age = 0
        high_water = stop = 0.0
        entry_size = 0.25
        for i in range(len(data)):
            if active:
                age += 1
                if not valid[i]:
                    active = False
                else:
                    high_water = max(high_water, float(close[i]))
                    stop = max(stop, high_water - params.stop_atr * float(atr[i]))
                    if close[i] <= stop or age >= 2:
                        active = False
            elif valid[i] and entry[i]:
                active = True
                age = 0
                # Anchor the trail at the entry decision's close.
                high_water = float(close[i])
                stop = high_water - params.stop_atr * float(atr[i])
                entry_size = float(np.clip(strength[i] / (3.0 * params.innovation_threshold), 0.25, 1.0))
            if active:
                signals[i] = 1
                sizes[i] = entry_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 98.0,
         "close": [100.0, 100.0, 99.0, 100.0, 101.0, 100.0], "volume": 1000.0},
        index=index,
    )
    assert strategy.indicators(data, params).index.equals(index)
    ind = pd.DataFrame(
        {"expected_slope": -0.1, "fast_slope": 0.1, "drawdown_atr": 2.0,
         "recovery": 0.3, "strength": [0.0, 0.3, 0.0, 0.6, 0.0, 0.0],
         "previous_strength": 0.0, "atr": [1.0, 1.0, 10.0, 1.0, 1.0, 1.0]},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 1, 1]
    assert np.allclose(result.loc[index[[2, 4, 5]], "size"], [0.5, 1.0, 1.0])
    assert result["size"].gt(0).all()
