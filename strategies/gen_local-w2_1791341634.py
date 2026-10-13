from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPressureParams:
    fractional_order: float = 0.45
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[GapPressureParams]):
    strategy_id = "gen_local-w2_1791341634"

    @classmethod
    def params_type(cls) -> type[GapPressureParams]:
        return GapPressureParams

    def warmup_bars(self, params: GapPressureParams) -> int:
        # 64 fractional lags, 20-bar centering, and one crossing lag.
        return 84

    def indicators(self, data: pd.DataFrame, params: GapPressureParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["previous_close"] = close.shift(1)
        out["gap"] = data["open"] - out["previous_close"]
        out["body"] = close - data["open"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        weights = np.ones(64, dtype=float)
        for lag in range(1, 64):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        # ponytail: fixed 64-term memory; extend only if truncation materially changes signals.
        fractional_close = close.copy()
        for lag in range(1, 64):
            fractional_close = fractional_close + weights[lag] * close.shift(lag)
        out["fractional_close"] = fractional_close
        out["fractional_center"] = fractional_close.rolling(20, min_periods=20).mean()
        out["fractional_pressure"] = fractional_close - out["fractional_center"]
        out["previous_pressure"] = out["fractional_pressure"].shift(1)
        out["gap_flow"] = out["gap"].rolling(5, min_periods=5).sum()
        out["body_flow"] = out["body"].rolling(5, min_periods=5).sum()
        ready = np.isfinite(out[["atr", "fractional_pressure", "previous_pressure", "gap_flow", "body_flow"]]).all(axis=1) & out["atr"].gt(0.0)
        out["long_entry"] = (
            ready
            & out["previous_pressure"].le(0.0)
            & out["fractional_pressure"].gt(0.0)
            & out["gap_flow"].gt(0.5 * out["atr"])
            & out["body_flow"].lt(0.0)
        ).fillna(False)
        out["short_entry"] = (
            ready
            & out["previous_pressure"].ge(0.0)
            & out["fractional_pressure"].lt(0.0)
            & out["gap_flow"].lt(-0.5 * out["atr"])
            & out["body_flow"].gt(0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapPressureParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        long_entry = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        short_entry = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        extreme = stop = np.nan
        for i, price in enumerate(close):
            if not np.isfinite(price) or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                position = age = 0
                extreme = stop = np.nan
                continue
            if position:
                # The first held bar is the bar after the entry decision.
                age += 1
                if position == 1:
                    extreme = price if age == 1 else max(extreme, price)
                    candidate = extreme - params.trail_atr * atr[i]
                    stop = candidate if age == 1 else max(stop, candidate)
                    stopped = price <= stop
                else:
                    # Mirror the highest-close trail for short positions.
                    extreme = price if age == 1 else min(extreme, price)
                    candidate = extreme + params.trail_atr * atr[i]
                    stop = candidate if age == 1 else min(stop, candidate)
                    stopped = price >= stop
                if stopped or age >= 18:
                    position = age = 0
                    extreme = stop = np.nan
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                age = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapPressureParams()
    for direction, prices in [(1, [100.0, 110.0, 109.0, 106.0, 106.0]), (-1, [100.0, 90.0, 91.0, 94.0, 94.0])]:
        index = pd.date_range("2020-01-01", periods=5)
        data = pd.DataFrame({"close": prices}, index=index)
        indicators = pd.DataFrame({
            "atr": [1.0, 1.0, 10.0, 1.0, 1.0],
            "long_entry": [direction == 1, False, False, False, False],
            "short_entry": [direction == -1, False, False, False, False],
        }, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0]
        assert result["size"].gt(0.0).all()
    index = pd.date_range("2020-01-01", periods=21)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "long_entry": [True] + [False] * 20, "short_entry": False}, index=index)
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == [0] + [1] * 18 + [0] * 2


if __name__ == "__main__":
    _self_check()
