from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    compression_limit: float = 0.8
    degree_threshold: int = 2
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790892368"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        if not 0.0 < params.compression_limit <= 1.0:
            raise ValueError("compression_limit must be in (0, 1]")
        if params.degree_threshold not in (1, 2, 3, 4):
            raise ValueError("degree_threshold must be an integer from 1 to 4")
        if params.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be 1 or 2")
        return 7

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        high = data["high"].replace([np.inf, -np.inf], np.nan)
        low = data["low"].replace([np.inf, -np.inf], np.nan)
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        out["candle_range"] = (high - low).where(high > low)
        out["prior_mean3"] = out["candle_range"].rolling(3, min_periods=3).mean().shift(1)
        out["prior_mean6"] = out["candle_range"].rolling(6, min_periods=6).mean().shift(1)
        out["compression"] = out["prior_mean3"] / out["prior_mean6"]
        out["close_pressure"] = (2.0 * close - high - low) / out["candle_range"]

        # Horizontal visibility requires every intervening vertex below both endpoints.
        # Adjacent edges are universal, so only nonadjacent edges carry information.
        for name, values in (("upper_degree", high), ("lower_degree", -low)):
            degree = pd.Series(0.0, index=data.index)
            for lag in range(2, 6):
                ceiling = np.minimum(values, values.shift(lag))
                visible = values.notna() & values.shift(lag).notna()
                for middle in range(1, lag):
                    visible &= values.shift(middle) < ceiling
                degree += visible.astype(float)
            out[name] = degree.where(values.rolling(6, min_periods=6).count().eq(6))
        out["graph_balance"] = out["upper_degree"] - out["lower_degree"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        compressed = indicators["compression"].le(params.compression_limit)
        long_entry = (
            compressed
            & indicators["graph_balance"].ge(params.degree_threshold)
            & indicators["close_pressure"].ge(0.35)
        )
        short_entry = (
            compressed
            & indicators["graph_balance"].le(-params.degree_threshold)
            & indicators["close_pressure"].le(-0.35)
        )
        ready = indicators[["compression", "graph_balance", "close_pressure"]].notna().all(axis=1)
        candidates = np.select(
            [(ready & long_entry).to_numpy(), (ready & short_entry).to_numpy()],
            [1, -1],
            default=0,
        )
        signals = np.zeros(len(data), dtype=int)
        position, exit_at = 0, -1
        for i, candidate in enumerate(candidates):
            if position and i >= exit_at:
                position = 0
            elif position == 0 and candidate:
                position = int(candidate)
                exit_at = i + params.holding_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=7)
    for direction in (1, -1):
        supplied = pd.DataFrame({"compression": 0.5, "graph_balance": 3.0 * direction, "close_pressure": 0.5 * direction}, index=index)
        for hold, expected in ((1, [0, 1, 0, 1, 0, 1, 0, 1]), (2, [0, 1, 1, 0, 1, 1, 0, 1])):
            params = VisibilityParams(holding_bars=hold)
            result = strategy.generate_signals(data, supplied, ctx, params).data
            assert result["signal"].tolist() == [direction * value for value in expected]
            assert result["size"].gt(0).all()
        supplied.iloc[0] = np.nan
        assert strategy.generate_signals(data, supplied, ctx, VisibilityParams()).data["signal"].iloc[1] == 0
    actual = strategy.indicators(data, VisibilityParams())
    assert actual["compression"].iloc[:6].isna().all()
    assert actual["upper_degree"].iloc[-1] == 0
    assert strategy.generate_signals(data, actual, ctx, VisibilityParams()).data["signal"].eq(0).all()
    graph_data = data.iloc[:6].copy()
    graph_data["high"] = [5.0, 1.0, 2.0, 1.0, 1.0, 6.0]
    graph_data["low"] = 0.0
    assert strategy.indicators(graph_data, VisibilityParams())["upper_degree"].iloc[-1] == 2


if __name__ == "__main__":
    _self_check()
