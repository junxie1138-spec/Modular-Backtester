from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    topology_window: int = 16
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791285705"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(4 * params.topology_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        w = int(params.topology_window)
        if w < 4:
            raise ValueError("topology_window must be at least 4")
        close = data["close"].astype(float)
        degree = pd.Series(0.0, index=data.index)
        maximum_slope = pd.Series(-np.inf, index=data.index)
        # Natural visibility: intervening backward slopes must be smaller.
        for lag in range(1, w + 1):
            slope = (close.shift(lag) - close) / lag
            degree += (slope.notna() & (slope > maximum_slope)).astype(float)
            maximum_slope = pd.Series(
                np.maximum(maximum_slope.to_numpy(), slope.fillna(-np.inf).to_numpy()),
                index=data.index,
            )
        degree = degree.where(close.rolling(w + 1, min_periods=w + 1).count().eq(w + 1))
        span = (
            data["high"].rolling(w, min_periods=w).max()
            - data["low"].rolling(w, min_periods=w).min()
        ) / close.where(close > 0)
        baseline = span.rolling(3 * w, min_periods=3 * w).median()
        out = pd.DataFrame(index=data.index)
        out["visibility_degree"] = degree
        out["range_span"] = span
        out["range_baseline"] = baseline
        out["compressed_before_release"] = (
            span.shift(1) < 0.70 * baseline.shift(1)
        ).fillna(False)
        out["new_visibility_hub"] = (
            (degree >= np.ceil(0.60 * w))
            & (degree.shift(1) <= np.floor(0.30 * w))
        ).fillna(False)
        out["up_close"] = (close > close.shift(1)).fillna(False)
        out["entry"] = (
            out["compressed_before_release"]
            & out["new_visibility_hub"]
            & out["up_close"]
            & baseline.shift(1).notna()
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        hold = int(params.holding_bars)
        if hold < 1:
            raise ValueError("holding_bars must be positive")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                else:
                    exit_bar = -1
                continue
            if entries[i]:
                signals[i] = 1
                exit_bar = i + hold
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry": [True] * 12}, index=index)
    result = GeneratedStrategy.generate_signals(
        data, indicators, None, VisibilityParams(holding_bars=3)
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    assert result["size"].gt(0).all()
    computed = GeneratedStrategy.indicators(data, VisibilityParams(topology_window=4))
    assert computed.index.equals(data.index)
    assert not computed["entry"].any()


if __name__ == "__main__":
    _self_check()
