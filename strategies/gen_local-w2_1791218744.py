from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    complexity_window: int = 42
    trend_complexity: float = 0.45
    reversion_complexity: float = 0.75
    momentum_window: int = 5
    noise_window: int = 20
    min_efficiency: float = 0.35
    pullback_z: float = 0.8
    atr_window: int = 14
    trail_atr: float = 3.0
    hold_bars: int = 18

    def __post_init__(self):
        if min(self.complexity_window, self.noise_window, self.atr_window) < 2:
            raise ValueError("Rolling windows must be at least two.")
        if self.momentum_window < 1 or not 15 <= self.hold_bars <= 20:
            raise ValueError("Invalid momentum window or holding horizon.")
        if not 0 <= self.trend_complexity < self.reversion_complexity <= 1:
            raise ValueError("Complexity thresholds must be ordered in [0, 1].")
        if not 0 <= self.min_efficiency <= 1 or self.pullback_z <= 0 or self.trail_atr <= 0:
            raise ValueError("Invalid efficiency, pullback or trailing-stop parameter.")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791218744"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return int(max(200, params.complexity_window + 4,
                       params.noise_window + 1, params.momentum_window + 1,
                       params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        triples = np.column_stack([out["return"].shift(2),
                                   out["return"].shift(1), out["return"]])
        ranks = np.argsort(np.argsort(triples, axis=1, kind="stable"),
                           axis=1, kind="stable")
        pattern = pd.Series(9 * ranks[:, 0] + 3 * ranks[:, 1] + ranks[:, 2],
                            index=data.index, dtype=float)
        pattern = pattern.where(np.isfinite(triples).all(axis=1))
        out["pattern"] = pattern
        out["previous_pattern"] = pattern.shift(1)
        valid = pattern.notna() & out["previous_pattern"].notna()
        entropy = pd.Series(0.0, index=data.index)
        window = params.complexity_window
        codes = (5, 7, 11, 15, 19, 21)
        for previous in codes:
            marginal = (out["previous_pattern"] == previous).astype(float).where(valid)
            marginal = marginal.rolling(window, min_periods=window).mean()
            for current in codes:
                joint = ((out["previous_pattern"] == previous) & (pattern == current))
                probability = joint.astype(float).where(valid).rolling(window, min_periods=window).mean()
                ratio = probability.div(marginal.replace(0.0, np.nan))
                term = probability * np.log(ratio.where(ratio > 0.0, 1.0))
                entropy -= term
        # Overlapping triples permit three successor permutations per pattern.
        out["complexity"] = (entropy / np.log(3.0)).clip(0.0, 1.0)
        out["momentum"] = out["return"].rolling(params.momentum_window, min_periods=params.momentum_window).sum()
        out["path"] = out["return"].abs().rolling(params.momentum_window, min_periods=params.momentum_window).sum()
        out["efficiency"] = out["momentum"].abs().div(out["path"].replace(0.0, np.nan))
        out["noise"] = out["return"].rolling(params.noise_window, min_periods=params.noise_window).std()
        out["pullback_score"] = out["momentum"].div((out["noise"] * np.sqrt(params.momentum_window)).replace(0.0, np.nan))
        out["true_range"] = pd.concat([data["high"] - data["low"],
                                      (data["high"] - close.shift(1)).abs(),
                                      (data["low"] - close.shift(1)).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["trend_entry"] = ((out["complexity"] <= params.trend_complexity)
                              & (out["momentum"] > 0.0)
                              & (out["efficiency"] >= params.min_efficiency))
        out["reversion_entry"] = ((out["complexity"] >= params.reversion_complexity)
                                  & (out["pullback_score"] <= -params.pullback_z)
                                  & (out["return"] > 0.0))
        out["entry"] = ((close > out["ma200"]) & (out["atr"] > 0.0)
                        & (out["trend_entry"] | out["reversion_entry"])).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        prices = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            valid = np.isfinite(prices[i]) and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(regime[i])
            if active:
                age += 1
                if valid:
                    high_water = prices[i] if np.isnan(high_water) else max(high_water, prices[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not valid or prices[i] <= stop or prices[i] <= regime[i] or age >= params.hold_bars:
                    active = False
                signals[i] = int(active)
            elif valid and entries[i]:
                active = True
                # The first in-trade close occurs on the next bar after the fill.
                high_water = np.nan
                stop = -np.inf
                age = 0
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 105., 103., 101., 99., 102.]}, index=index)
    indicators = pd.DataFrame({"atr": [1., 1., 3., 3., 3., 1.],
                               "ma200": 90., "entry": [True, False, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    index = pd.date_range("2020-01-01", periods=260)
    close = 100. + np.arange(260) * 0.05 + np.sin(np.arange(260) * 0.7)
    data = pd.DataFrame({"open": close, "high": close + 1., "low": close - 1.,
                         "close": close, "volume": 1000.}, index=index)
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:240], strategy.indicators(data.iloc[:240], params))
    assert full["complexity"].iloc[:params.complexity_window + 3].isna().all()
    assert full["complexity"].iloc[params.complexity_window + 3:].notna().all()


if __name__ == "__main__":
    _self_check()
