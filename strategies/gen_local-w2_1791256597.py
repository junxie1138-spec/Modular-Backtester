from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 16
    extrema_window: int = 20
    atr_window: int = 14
    entropy_drop: float = 0.08
    stretch_atr: float = 1.5
    trail_atr: float = 1.8
    holding_bars: int = 4

    def __post_init__(self):
        if min(self.pattern_window, self.extrema_window, self.atr_window) < 3:
            raise ValueError("Lookbacks must be at least three bars.")
        if not 3 <= self.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (self.entropy_drop, self.stretch_atr, self.trail_atr)):
            raise ValueError("Thresholds must be finite and positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791256597"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.pattern_window + 6,
                       params.extrema_window + 2, params.atr_window + 2))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["previous_close"] = close.shift(1)
        out["older_close"] = close.shift(2)
        out["prior_peak"] = close.rolling(params.extrema_window,
                                         min_periods=params.extrema_window).max().shift(1)
        out["prior_trough"] = close.rolling(params.extrema_window,
                                           min_periods=params.extrema_window).min().shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        # Pairwise comparisons encode six ordinal permutations; ties keep time order.
        older, previous = out["older_close"], out["previous_close"]
        valid = np.isfinite(older) & np.isfinite(previous) & np.isfinite(close)
        pattern = ((older > previous).astype(int) * 4
                   + (older > close).astype(int) * 2
                   + (previous > close).astype(int))
        out["pattern"] = pattern.where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            frequency = (out["pattern"] == code).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window).mean()
            out[f"pattern_frequency_{code}"] = frequency
            entropy = entropy - frequency * np.log(frequency.clip(lower=1e-12))
        out["entropy"] = entropy / np.log(6.0)
        out["entropy_drop"] = out["entropy"].shift(3) - out["entropy"]
        out["previous_atr"] = out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["drawdown_atr"] = (
            out["prior_peak"] - previous) / out["previous_atr"]
        out["drawup_atr"] = (
            previous - out["prior_trough"]) / out["previous_atr"]
        ready = np.isfinite(out["atr"]) & (out["atr"] > 0)
        contraction = out["entropy_drop"] >= params.entropy_drop
        out["long_setup"] = (ready & contraction
                             & (out["drawdown_atr"] >= params.stretch_atr)
                             & (close > previous)).fillna(False)
        out["short_setup"] = (ready & contraction
                              & (out["drawup_atr"] >= params.stretch_atr)
                              & (close < previous)).fillna(False)
        out["long_confirmed"] = out["long_setup"] & out["long_setup"].shift(
            1, fill_value=False)
        out["short_confirmed"] = out["short_setup"] & out["short_setup"].shift(
            1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_confirmed"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_confirmed"].fillna(False).to_numpy(dtype=bool)
        position, age = 0, 0
        highest, floor = -np.inf, -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position, age = 0, 0
                highest, floor = -np.inf, -np.inf
                continue
            if position:
                # This bar follows the entry decision: its open is the first fill.
                age += 1
                highest = max(highest, close)
                floor = max(floor, highest - params.trail_atr * atr)
                # Both directions use the required high-water floor.
                # For shorts this realizes a sufficiently large downward recovery.
                if close <= floor or age >= params.holding_bars:
                    position, age = 0, 0
                    highest, floor = -np.inf, -np.inf
                    continue
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                age = 0
                highest, floor = -np.inf, -np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=8)
    close = pd.Series([100., 100., 105., 104., 103., 102., 102., 102.], index=index)
    bars = pd.DataFrame({"open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1000.})
    features = pd.DataFrame({
        "atr": [1., 1., 1., 5., 5., 5., 5., 5.],
        "long_confirmed": [False, True, False, False, False, False, False, False],
        "short_confirmed": False
    }, index=index)
    result = strategy.generate_signals(bars, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    bars.loc[:, ["open", "high", "low", "close"]] = 100.
    features["atr"] = 1.
    result = strategy.generate_signals(bars, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    short = features.copy()
    short["short_confirmed"] = short["long_confirmed"]
    short["long_confirmed"] = False
    result = strategy.generate_signals(bars, short, None, params).data
    assert result["signal"].tolist() == [0, 0, -1, -1, -1, -1, 0, 0]
    index = pd.date_range("2021-01-01", periods=80)
    close = pd.Series(np.arange(100., 180.), index=index)
    bars = pd.DataFrame({"open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1000.})
    features = strategy.indicators(bars, params)
    assert abs(features["entropy"].iloc[-1]) < 1e-12
    assert not features["long_confirmed"].iloc[:strategy.warmup_bars(params)].any()
    assert features.iloc[:50].equals(strategy.indicators(bars.iloc[:50], params))
