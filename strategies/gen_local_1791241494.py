from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    path_window: int = 8
    area_threshold: float = 0.02
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791241494"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.path_window + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.path_window < 2 or params.hold_bars < 1:
            raise ValueError("path_window must be >= 2 and hold_bars >= 1")
        if not (np.isfinite(params.area_threshold) and params.area_threshold > 0):
            raise ValueError("area_threshold must be finite and positive")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Require 0 < min_size <= max_size <= 1")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).where(lambda x: x > 0)
        opening = data["open"].where(lambda x: x > 0)
        out["gap"] = np.log(opening / previous).replace([np.inf, -np.inf], np.nan)
        out["session"] = np.log(data["close"].where(lambda x: x > 0) / opening).replace([np.inf, -np.inf], np.nan)
        window = params.path_window
        out["gap_path"] = out["gap"].fillna(0.0).cumsum()
        out["session_path"] = out["session"].fillna(0.0).cumsum()
        # Antisymmetric second signature level, including gap-before-session order.
        out["cross_increment"] = (
            out["gap_path"].shift(1).fillna(0.0) * out["session"]
            - out["session_path"].shift(1).fillna(0.0) * out["gap"]
            + out["gap"] * out["session"]
        )
        out["gap_net"] = out["gap"].rolling(window).sum()
        out["session_net"] = out["session"].rolling(window).sum()
        out["area"] = 0.5 * (
            out["cross_increment"].rolling(window).sum()
            - out["gap_path"].shift(window).fillna(0.0) * out["session_net"]
            + out["session_path"].shift(window).fillna(0.0) * out["gap_net"]
        )
        out["path_length"] = (out["gap"].abs() + out["session"].abs()).rolling(window).sum()
        out["score"] = (2.0 * out["area"] / out["path_length"].pow(2).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["direction"] = np.sign(out["gap_net"]).fillna(0).astype(int)
        out["eligible"] = (
            out["score"].ge(params.area_threshold)
            & (out["gap_net"] * out["session_net"]).gt(0)
            & (out["gap"] * out["gap_net"]).gt(0)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        direction = indicators["direction"].fillna(0).to_numpy(dtype=int)
        strength = (indicators["score"] / params.area_threshold).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        entry_sizes = (
            params.min_size
            + (params.max_size - params.min_size) * (strength - 1.0).clip(0.0, 2.0) / 2.0
        ).to_numpy(dtype=float)
        side, exit_bar, held_size = 0, -1, params.min_size
        start = max(self.warmup_bars(params), int(ctx.warmup_bars))
        for i in range(start, len(data)):
            if side:
                if i >= exit_bar:
                    side = 0
                    continue
            elif eligible[i] and direction[i] != 0:
                side = int(direction[i])
                exit_bar = i + params.hold_bars
                held_size = float(entry_sizes[i])
            signals[i] = side
            if side:
                sizes[i] = held_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GeneratedParams(path_window=3, hold_bars=5)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert np.allclose(derived["score"].dropna(), -1.0 / (4 * params.path_window))
    test = pd.DataFrame({"eligible": False, "direction": 0, "score": 0.0}, index=index)
    for entry, side, strength in [(4, 1, 2.0), (11, -1, 3.0)]:
        test.iloc[entry] = [True, side, params.area_threshold * strength]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, test, ctx, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[5:10], expected[12:17] = 1, -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].between(params.min_size, params.max_size).all()
    assert np.allclose(result["size"].iloc[5:10], 0.60)
    assert np.allclose(result["size"].iloc[12:17], 0.95)


if __name__ == "__main__":
    _self_check()
