from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PatternParams:
    pattern_window: int = 18
    complexity_ceiling: float = 0.72
    directional_floor: float = 0.25
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[PatternParams]):
    strategy_id = "gen_local_1790873427"

    @classmethod
    def params_type(cls):
        return PatternParams

    def warmup_bars(self, params: PatternParams) -> int:
        return int(max(200, params.pattern_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: PatternParams) -> pd.DataFrame:
        if params.pattern_window < 3 or params.atr_window < 1 or params.hold_bars < 1:
            raise ValueError("Window lengths and holding period must be positive.")
        if not (0 < params.complexity_ceiling <= 1 and 0 < params.directional_floor <= 1):
            raise ValueError("Complexity and directional thresholds must be in (0, 1].")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")
        close = data["close"]
        a, b, c = close.shift(2), close.shift(1), close
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        valid &= (a != b) & (a != c) & (b != c)
        pattern = (4 * (a > b).astype(int) + 2 * (a > c).astype(int)
                   + (b > c).astype(int)).astype(float).where(valid)
        out = pd.DataFrame(index=data.index)
        out["pattern"] = pattern
        probabilities = []
        for code in (0, 1, 3, 4, 6, 7):
            name = "p_" + str(code)
            out[name] = (pattern == code).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
            probabilities.append(name)
        out["complexity"] = ((1.0 - out[probabilities].pow(2).sum(
            axis=1, min_count=6
        )) * 6.0 / 5.0).clip(0.0, 1.0)
        out["direction"] = out["p_0"] - out["p_7"]
        out["ma_200"] = close.rolling(200, min_periods=200).mean()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: PatternParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        orderly = indicators["complexity"].le(params.complexity_ceiling)
        ready = np.isfinite(indicators[[
            "complexity", "direction", "ma_200", "atr"
        ]]).all(axis=1) & indicators["atr"].gt(0) & np.isfinite(data["close"])
        long_entry = (ready & orderly & data["close"].gt(indicators["ma_200"])
                      & indicators["direction"].ge(params.directional_floor))
        short_entry = (ready & orderly & data["close"].lt(indicators["ma_200"])
                       & indicators["direction"].le(-params.directional_floor))
        desired = np.select([long_entry, short_entry], [1, -1], default=0)
        raw = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry, distance = np.nan, np.nan
        for i in range(len(data)):
            if side:
                # The preceding close's decision fills at this bar's open.
                if np.isnan(entry):
                    entry = opens[i]
                age += 1
                invalid = not (np.isfinite(entry) and np.isfinite(close[i]))
                stopped = side * (close[i] - entry) < -distance
                if invalid or stopped or age >= params.hold_bars:
                    side, age, entry = 0, 0, np.nan
            elif desired[i]:
                side = int(desired[i])
                age, entry = 0, np.nan
                # Lock ATR at the decision close; never update the stop distance.
                distance = params.stop_atr * atr[i]
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = PatternParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    index = pd.date_range("2020-01-01", periods=5)
    for side in (1, -1):
        data = pd.DataFrame({
            "open": 100.0, "high": 110.0, "low": 90.0,
            "close": [100.0, 100.0 + side, 100.0 - 5 * side, 100.0, 100.0],
            "volume": 1000.0,
        }, index=index)
        derived = pd.DataFrame({
            "complexity": 0.2, "direction": float(side),
            "ma_200": 100.0 - 10 * side, "atr": 2.0,
        }, index=index)
        signals = strategy.generate_signals(data, derived, ctx, params).data
        assert signals["signal"].iloc[:4].tolist() == [0, side, side, 0]
        assert signals["size"].gt(0).all()
        short_data = data.iloc[:2]
        prefix = strategy.generate_signals(short_data, derived.iloc[:2], ctx, params).data
        assert signals.iloc[:2].equals(prefix)
    trend_index = pd.date_range("2020-01-01", periods=230)
    close = np.arange(100.0, 330.0)
    trend = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    }, index=trend_index)
    derived = strategy.indicators(trend, params)
    assert derived["complexity"].dropna().eq(0.0).all()
    signals = strategy.generate_signals(trend, derived, ctx, params).data["signal"]
    assert signals.iloc[:200].eq(0).all()
    assert signals.iloc[200:208].eq(1).all()
    assert signals.iloc[208] == 0


if __name__ == "__main__":
    _self_check()
