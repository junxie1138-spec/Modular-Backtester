from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    depth_trigger: float = 0.015
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1790980299"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        # Peak window, occupancy window, then one lag: 60 + 20.
        return 80

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not np.isfinite(params.depth_trigger) or not 0 < params.depth_trigger < 1:
            raise ValueError("depth_trigger must be between zero and one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive")
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["close"].rolling(60, min_periods=60).max()
        out["depth"] = 1.0 - data["close"] / out["peak"].where(out["peak"] > 0)
        out["previous_depth"] = out["depth"].shift(1)
        out["underwater"] = (out["depth"] >= params.depth_trigger / 2.0).astype(float).where(out["depth"].notna())
        out["occupancy"] = out["underwater"].rolling(20, min_periods=20).mean().shift(1)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        valid = out[["depth", "previous_depth", "occupancy", "atr"]].notna().all(axis=1) & (out["atr"] > 0)
        crossing = (out["depth"] >= params.depth_trigger) & (out["previous_depth"] < params.depth_trigger)
        out["setup"] = np.where(valid & crossing, np.where(out["occupancy"] >= 0.7, 1, -1), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        values = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry = None
        best, stop, armed = 0.0, -np.inf, False
        for i in range(len(data)):
            opening, high, low, closing = values[i]
            valid = np.isfinite(values[i]).all() and min(values[i]) > 0 and high >= max(opening, closing) and low <= min(opening, closing) and np.isfinite(atr[i]) and atr[i] > 0
            if not valid:
                side, entry = 0, None
                continue
            if side:
                # A decision made yesterday enters at today's open.
                if entry is None:
                    entry = opening
                    best = side * entry
                age += 1
                # Test the previously established stop before tightening it.
                adverse = side * (low if side == 1 else high)
                if adverse <= stop or age >= 2:
                    side, entry = 0, None
                else:
                    favorable = side * (high if side == 1 else low)
                    best = max(best, favorable)
                    armed = armed or best - side * entry >= 0.005 * entry
                    if armed:
                        # Signed stop only rises: long floor rises, short ceiling falls.
                        stop = max(stop, side * entry, best - params.trail_atr * atr[i])
                        # A newly tightened stop becomes effective next session.
            elif setups[i]:
                side, age, entry = int(setups[i]), 0, None
                stop, armed = -np.inf, False
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    close = np.array([100, 101, 100, 100, 99, 100, 100, 100, 100, 100], dtype=float)
    data = pd.DataFrame({"open": 100.0, "high": np.maximum(100.0, close) + 0.1, "low": np.minimum(100.0, close) - 0.1, "close": close, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"setup": [1, 0, 0, -1, 0, 0, 1, 0, 0, 0], "atr": 0.2}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=80)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1, 1, 0]
    assert (result["size"] > 0).all()
    for n in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:n], indicators.iloc[:n], ctx, params).data
        pd.testing.assert_frame_equal(prefix, result.iloc[:n])
    assert strategy.indicators(data, params)["setup"].eq(0).all()
