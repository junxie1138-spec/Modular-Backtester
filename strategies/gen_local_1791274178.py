from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    channel_bars: int = 12
    migration_bars: int = 5
    atr_bars: int = 14
    entry_threshold: float = 0.025
    trail_atr: float = 3.0
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791274178"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.channel_bars + params.migration_bars, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.channel_bars, params.migration_bars, params.atr_bars, params.max_hold) < 1:
            raise ValueError("Lookbacks and holding limit must be positive.")
        if not np.isfinite(params.entry_threshold) or params.entry_threshold <= 0:
            raise ValueError("Entry threshold must be positive and finite.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("ATR multiplier must be positive and finite.")
        out = pd.DataFrame(index=data.index)
        out["ceiling"] = data["high"].rolling(params.channel_bars).max()
        out["floor"] = data["low"].rolling(params.channel_bars).min()
        out["width"] = out["ceiling"] - out["floor"]
        out["old_width"] = out["width"].shift(params.migration_bars).replace(0, np.nan)
        out["upper_move"] = out["ceiling"].diff(params.migration_bars)
        out["lower_move"] = out["floor"].diff(params.migration_bars)
        out["contraction"] = (1.0 - out["width"] / out["old_width"]).clip(0.0, 1.0)
        out["position"] = ((data["close"] - out["floor"]) / out["width"].replace(0, np.nan)).clip(0.0, 1.0)
        out["score"] = (out["upper_move"] + out["lower_move"]) / out["old_width"] * out["contraction"]
        out["inward"] = (out["upper_move"] <= 0) & (out["lower_move"] >= 0)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.35, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        scores = indicators["score"].to_numpy(dtype=float)
        positions = indicators["position"].to_numpy(dtype=float)
        inward = indicators["inward"].fillna(False).to_numpy(dtype=bool)
        direction, age = 0, 0
        high_water, stop, entry_size = -np.inf, -np.inf, 0.35
        for i in range(n):
            close, atr, score, position = closes[i], atrs[i], scores[i], positions[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                direction, age = 0, 0
            elif direction:
                age += 1
                # Reflected closes give shorts the same upward-ratcheting high-water rule.
                high_water = max(high_water, direction * close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if direction * close <= stop or age >= params.max_hold:
                    direction, age = 0, 0
            elif inward[i] and np.isfinite(score) and np.isfinite(position):
                if score >= params.entry_threshold and position >= 0.65:
                    direction = 1
                elif score <= -params.entry_threshold and position <= 0.35:
                    direction = -1
                if direction:
                    age = 0
                    # First in-trade close is the next bar, after the delayed entry.
                    high_water, stop = -np.inf, -np.inf
                    strength = float(np.clip(abs(score) / params.entry_threshold, 1.0, 3.0))
                    entry_size = 0.35 + 0.30 * (strength - 1.0)
            signals[i] = direction
            sizes[i] = entry_size if direction else 0.35
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.35).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=9)
    close = np.array([100, 104, 102, 100, 100, 96, 98, 100, 100], dtype=float)
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
        index=index,
    )
    manual = pd.DataFrame(
        {"atr": [1, 1, 10, 1, 1, 1, 10, 1, 1],
         "score": [0.05, 0, 0, 0, -0.10, 0, 0, 0, 0],
         "position": [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
         "inward": True},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, manual, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert np.isclose(result["size"].iloc[1], 0.65)
    assert np.isclose(result["size"].iloc[5], 0.95)
    params.max_hold = 2
    timed = strategy.generate_signals(data, manual, ctx, params).data
    assert timed["signal"].iloc[:4].tolist() == [0, 1, 1, 0]
    flat = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0.0},
        index=pd.date_range("2020-01-01", periods=50),
    )
    derived = strategy.indicators(flat, params)
    assert not strategy.generate_signals(flat, derived, ctx, params).data["signal"].any()
