from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    graph_window: int = 16
    percentile_window: int = 126
    entry_quantile: float = 0.80
    profit_target: float = 0.035
    max_hold: int = 18
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791331361"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.graph_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not (
            params.graph_window >= 3 and params.percentile_window >= 2
            and 0.0 < params.entry_quantile < 1.0
            and params.profit_target > 0.0 and params.max_hold >= 1
            and np.isfinite(params.size) and params.size > 0.0
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        gap = (data["open"] / data["close"].shift(1).replace(0.0, np.nan) - 1.0)
        gap = gap.replace([np.inf, -np.inf], np.nan)
        magnitude = gap.abs()
        barrier = pd.Series(-np.inf, index=data.index)
        span_sum = pd.Series(0.0, index=data.index)
        squared_span_sum = span_sum.copy()
        # Horizontal visibility: both endpoints exceed every intervening gap.
        for distance in range(1, params.graph_window):
            older = magnitude.shift(distance)
            visible = (magnitude > barrier) & (older > barrier)
            span_sum += visible.astype(float) * distance
            squared_span_sum += visible.astype(float) * distance * distance
            barrier = pd.Series(np.maximum(barrier, older), index=data.index)
        complete = magnitude.rolling(params.graph_window).count().eq(params.graph_window)
        reach = squared_span_sum / span_sum.replace(0.0, np.nan) / (params.graph_window - 1)
        out["gap"] = gap
        out["visibility_reach"] = reach.where(complete)
        intraday = data["close"] / data["open"].replace(0.0, np.nan) - 1.0
        out["gap_rejection"] = (-np.sign(gap) * intraday / magnitude.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        out["reach_threshold"] = out["visibility_reach"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.entry_quantile)
        out["rejection_threshold"] = out["gap_rejection"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.entry_quantile)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        entry = (
            ready
            & (indicators["visibility_reach"] > indicators["reach_threshold"])
            & (indicators["gap_rejection"] > indicators["rejection_threshold"])
            & (indicators["gap_rejection"] > 0.0)
        ).to_numpy()
        gaps = indicators["gap"].to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # Entry is filled at this bar's open, never the decision close.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_price else np.nan
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    position = 0
                raw[i] = position
                continue
            if entry[i] and np.isfinite(gaps[i]) and gaps[i] != 0.0:
                position = -1 if gaps[i] > 0.0 else 1
                entry_bar = i + 1
                entry_price = np.nan
                raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GeneratedParams(max_hold=3)
    index = pd.date_range("2020-01-01", periods=32)
    data = pd.DataFrame(
        {"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.loc[index[2], "close"] = 104.0
    data.loc[index[9], "close"] = 96.0
    indicators = pd.DataFrame(
        {"gap": -0.01, "visibility_reach": 0.0, "gap_rejection": 1.0,
         "reach_threshold": 0.5, "rejection_threshold": 0.5},
        index=index,
    )
    indicators.loc[index[[1, 8, 15]], "visibility_reach"] = 1.0
    indicators.loc[index[8], "gap"] = 0.01
    strategy = GeneratedStrategy()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].iloc[:4].tolist() == [0, 0, 1, 0]
    assert signals["signal"].iloc[8:11].tolist() == [0, -1, 0]
    assert signals["signal"].iloc[15:20].tolist() == [0, 1, 1, 1, 0]
    assert signals["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
