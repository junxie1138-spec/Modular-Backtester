from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    graph_bars: int = 60
    percentile_bars: int = 126
    hub_percentile: float = 0.8
    profit_target: float = 0.025
    max_hold_bars: int = 8

    def __post_init__(self):
        if self.graph_bars < 5 or self.percentile_bars < 2:
            raise ValueError("Graph and percentile windows are too short.")
        if not 0.0 < self.hub_percentile < 1.0:
            raise ValueError("hub_percentile must be between zero and one.")
        if self.profit_target <= 0.0 or self.max_hold_bars < 1:
            raise ValueError("Profit target and holding period must be positive.")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1790984152"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.graph_bars + params.percentile_bars + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        volume = data["volume"].astype(float).where(
            np.isfinite(data["volume"]) & (data["volume"] > 0.0)
        )
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        barrier = pd.Series(-np.inf, index=data.index)
        degree = pd.Series(0.0, index=data.index)
        # Same-weekday nodes form separate horizontal visibility graphs.
        # An edge exists only if all intervening nodes are below both endpoints.
        for lag in range(1, params.graph_bars + 1):
            past = volume.shift(lag)
            same_day = weekday.eq(weekday.shift(lag))
            visible = same_day & volume.notna() & past.notna()
            visible &= (volume > barrier) & (past > barrier)
            degree += visible.astype(float) * (lag / params.graph_bars)
            barrier = barrier.where(~same_day, np.maximum(barrier, past.fillna(np.inf)))
        complete = volume.notna().rolling(
            params.graph_bars + 1, min_periods=params.graph_bars + 1
        ).sum().eq(params.graph_bars + 1)
        out["weekday_visibility_degree"] = degree.where(complete)
        out["hub_threshold"] = out["weekday_visibility_degree"].shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.hub_percentile)
        out["hub_release"] = (
            out["weekday_visibility_degree"].shift(1)
            - out["weekday_visibility_degree"]
        )
        out["entry"] = (
            (out["weekday_visibility_degree"].shift(1) > out["hub_threshold"].shift(1))
            & (out["hub_release"] > 0.0)
            & (data["close"] > data["open"])
            & (data["volume"] < data["volume"].shift(1))
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if active:
                # Yesterday's entry decision fills at today's open.
                if age == 0:
                    entry_price = opens[i]
                age += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                target_hit = valid_entry and closes[i] >= entry_price * (1.0 + params.profit_target)
                if not valid_entry or target_hit or age >= params.max_hold_bars:
                    active = False
            elif entries[i]:
                active = True
                age = 0
            signals[i] = int(active)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.loc[index[2], "close"] = 103.0
    indicators = pd.DataFrame({"entry": False}, index=index)
    indicators.loc[index[[0, 5]], "entry"] = True
    strategy = GeneratedStrategy()
    params = StrategyParams(profit_target=0.02, max_hold_bars=3)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert result["size"].gt(0.0).all()
    params = StrategyParams(graph_bars=5, percentile_bars=2)
    derived = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(derived.iloc[:10], strategy.indicators(data.iloc[:10], params))
    assert not derived["entry"].iloc[:strategy.warmup_bars(params) - 1].any()


if __name__ == "__main__":
    _self_check()
