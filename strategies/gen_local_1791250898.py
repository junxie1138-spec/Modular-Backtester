from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    lookback: int = 12
    stop_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local_1791250898"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return max(params.lookback, 15)

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        if params.lookback < 4 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("lookback must be >= 4 and stop_atr must be positive")
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        upper = pd.Series(1.0, index=data.index)
        lower = pd.Series(1.0, index=data.index)
        ceiling = close.shift(1)
        floor = close.shift(1)
        # Endpoint edges in ordinary and vertically inverted horizontal visibility graphs.
        # An edge exists only when both endpoints strictly exceed every intervening node.
        for lag in range(2, params.lookback):
            past = close.shift(lag)
            upper += ((close > ceiling) & (past > ceiling)).astype(float)
            lower += ((close < floor) & (past < floor)).astype(float)
            ceiling = pd.concat([ceiling, past], axis=1).max(axis=1, skipna=False)
            floor = pd.concat([floor, past], axis=1).min(axis=1, skipna=False)
        window_min = close.rolling(params.lookback, min_periods=params.lookback).min()
        window_max = close.rolling(params.lookback, min_periods=params.lookback).max()
        location = (close - window_min) / (window_max - window_min).replace(0, np.nan)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        atr = true_range.rolling(14, min_periods=14).mean()
        valid = (
            close.rolling(params.lookback, min_periods=params.lookback).count().eq(params.lookback)
            & np.isfinite(atr) & atr.gt(0) & np.isfinite(location)
        )
        imbalance = (lower - upper) / (lower + upper)
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[valid & location.le(0.25) & imbalance.ge(0.25) & lower.ge(3)] = 1
        candidate.loc[valid & location.ge(0.75) & imbalance.le(-0.25) & upper.ge(3)] = -1
        return pd.DataFrame({
            "upper_degree": upper.where(valid),
            "lower_degree": lower.where(valid),
            "location": location,
            "topology_imbalance": imbalance.where(valid),
            "true_range": true_range,
            "atr": atr,
            "candidate": candidate,
        }, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: TopologyParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        fixed_distance = np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    # Next-open entry with the adverse five-basis-point slippage in the config.
                    entry_price = opens[i] * (1.0 + side * 0.0005)
                invalid = not np.isfinite(entry_price) or not np.isfinite(closes[i])
                stopped = side * (closes[i] - entry_price) < -fixed_distance
                expired = i - entry_bar + 1 >= 2
                if invalid or stopped or expired:
                    side = 0
                    # Keep this close flat; a fresh entry requires a later decision.
            elif candidates[i] and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(candidates[i])
                entry_bar = i + 1
                fixed_distance = params.stop_atr * atr[i]
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = object.__new__(GeneratedStrategy)
    params = TopologyParams()
    index = pd.date_range("2020-01-01", periods=6)
    for direction in (1, -1):
        for stopped in (False, True):
            close = np.full(6, 100.0)
            if stopped:
                close[2] -= direction * 2.0
            data = pd.DataFrame({
                "open": 100.0, "high": close + 3.0, "low": close - 3.0,
                "close": close, "volume": 1000.0,
            }, index=index)
            features = pd.DataFrame({
                "candidate": [0, direction, 0, 0, 0, 0], "atr": 1.0,
            }, index=index)
            result = strategy.generate_signals(data, features, None, params).data
            expected = [0, 0, direction, 0 if stopped else direction, 0, 0]
            assert result["signal"].tolist() == expected
            assert result["size"].gt(0).all()
    index = pd.date_range("2020-02-01", periods=40)
    close = 100.0 + np.sin(np.arange(40)) * 2.0
    data = pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1000.0,
    }, index=index)
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:30], strategy.indicators(data.iloc[:30], params))
    assert full["candidate"].iloc[:strategy.warmup_bars(params) - 1].eq(0).all()
