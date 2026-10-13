from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_bars: int = 5
    baseline_bars: int = 20
    compression_ratio: float = 0.85
    cycle_threshold: float = 0.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791338824"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.graph_bars, params.baseline_bars + 3)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.graph_bars < 3 or params.baseline_bars < 1:
            raise ValueError("graph_bars >= 3 and baseline_bars >= 1 required")
        if not (0 < params.compression_ratio < 1 and 0 <= params.cycle_threshold <= 1):
            raise ValueError("Invalid compression_ratio or cycle_threshold")
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        width = ((data["high"] - data["low"]) / close).where(
            (data["low"] > 0) & (data["high"] >= data["low"])
        ).replace([np.inf, -np.inf], np.nan)
        n = params.graph_bars
        lag = [width.shift(i) for i in range(n)]
        excess_edges = pd.Series(0.0, index=data.index)
        # Every adjacent pair is connected; count only edges creating independent cycles.
        # ponytail: O(n^3) shift comparisons; use a visibility stack if windows become large.
        for left in range(2, n):
            for right in range(left - 1):
                visible = lag[left].notna() & lag[right].notna()
                for middle in range(right + 1, left):
                    chord = lag[left] + (lag[right] - lag[left]) * (
                        (left - middle) / (left - right)
                    )
                    visible &= lag[middle] < chord
                excess_edges += visible.astype(float)
        valid = width.rolling(n, min_periods=n).count().eq(n)
        cycle_density = (excess_edges / ((n - 1) * (n - 2) / 2)).where(valid)
        baseline = width.shift(3).rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).mean()
        return pd.DataFrame({
            "close": close,
            "range_fraction": width,
            "cycle_density": cycle_density,
            "compression": width.rolling(3, min_periods=3).mean() / baseline.replace(0, np.nan),
            "pressure_ma": close.rolling(3, min_periods=3).mean(),
            "regime_ma": close.rolling(200, min_periods=200).mean(),
        }, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        entry_condition = (
            indicators.notna().all(axis=1)
            & (indicators["close"] > indicators["regime_ma"])
            & (indicators["close"] > indicators["pressure_ma"])
            & (indicators["compression"] < params.compression_ratio)
            & (indicators["cycle_density"] >= params.cycle_threshold)
        ).fillna(False)
        # Hold while the same entry condition stays true; its reversal is the sole exit.
        df = pd.DataFrame({"signal": entry_condition.astype(int), "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params()
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    close = pd.Series(100.0 + np.arange(240) * 0.1, index=index)
    width = 0.04 * np.exp(-np.arange(240) / 40.0)
    data = pd.DataFrame({
        "open": close, "high": close * (1 + width / 2),
        "low": close * (1 - width / 2), "close": close, "volume": 1000000.0,
    }, index=index)
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert strategy.warmup_bars(params) >= 200
    assert np.isclose(indicators["cycle_density"].iloc[-1], 1.0)
    assert result["signal"].iloc[:200].eq(0).all()
    assert result["signal"].iloc[200:].eq(1).all()
    assert result["size"].gt(0).all()
    prefix = strategy.indicators(data.iloc[:220], params)
    pd.testing.assert_frame_equal(prefix, indicators.iloc[:220])
    flipped = indicators.copy()
    flipped.loc[index[-2], "cycle_density"] = 0.0
    exited = strategy.generate_signals(data, flipped, None, params).data
    assert exited["signal"].iloc[-2] == 1 and exited["signal"].iloc[-1] == 0
    missing = data.copy()
    missing.loc[index[-2], "high"] = np.nan
    missing_signals = strategy.generate_signals(
        missing, strategy.indicators(missing, params), None, params
    ).data
    assert missing_signals["signal"].iloc[-1] == 0
    empty = data.iloc[:0]
    assert strategy.generate_signals(
        empty, strategy.indicators(empty, params), None, params
    ).data.empty


if __name__ == "__main__":
    _self_check()
