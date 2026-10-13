from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapContagionParams:
    cluster_bars: int = 6
    rank_bars: int = 126
    crowding_percentile: float = 0.80
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapContagionParams]):
    strategy_id = "gen_local_1791277109"

    @classmethod
    def params_type(cls):
        return GapContagionParams

    def warmup_bars(self, params: GapContagionParams) -> int:
        return int(max(200, params.cluster_bars + params.rank_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GapContagionParams) -> pd.DataFrame:
        if (params.cluster_bars < 2 or params.rank_bars < 2
                or not 0 < params.crowding_percentile <= 1
                or not np.isfinite(params.profit_target)
                or params.profit_target <= 0 or params.max_hold_bars < 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = (data["open"] / out["previous_close"].where(
            out["previous_close"] > 0) - 1).replace([np.inf, -np.inf], np.nan)
        valid = out["gap"].notna() & data["close"].notna()
        out["unrepaired"] = (
            (out["gap"] < 0) & (data["close"] < out["previous_close"])
        ).astype(float).where(valid)
        out["prevalence"] = out["unrepaired"].rolling(
            params.cluster_bars, min_periods=params.cluster_bars).mean()
        out["crowding_rank"] = out["prevalence"].rolling(
            params.rank_bars, min_periods=params.rank_bars).rank(pct=True)
        out["previous_rank"] = out["crowding_rank"].shift(1)
        out["previous_prevalence"] = out["prevalence"].shift(1)
        out["previous_unrepaired"] = out["unrepaired"].shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["entry"] = (
            (out["previous_rank"] >= params.crowding_percentile)
            & (out["prevalence"] < out["previous_prevalence"])
            & (out["previous_unrepaired"] == 1)
            & (out["gap"] >= 0)
            & (data["close"] > data["open"])
            & (data["close"] > out["ma200"])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapContagionParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        desired = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                # Targets use observed closes; exit orders fill next bar.
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1 + params.profit_target)
                )
                time_hit = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_hit:
                    active = False
                    continue
            elif i >= self.warmup_bars(params) - 1 and entries[i]:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
            desired[i] = int(active)
        df["signal"] = desired
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=210, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 114.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index)
    data.iloc[201, data.columns.get_loc("open")] = 110.0
    data.iloc[202, data.columns.get_loc("close")] = 113.0
    indicators = pd.DataFrame({"entry": False}, index=index)
    indicators.iloc[200, 0] = True
    strategy = GeneratedStrategy()
    ctx = StrategyContext("SPY", "1d", 200)
    for params in (
        GapContagionParams(),
        GapContagionParams(profit_target=0.5, max_hold_bars=2),
    ):
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].iloc[200:204].tolist() == [0, 1, 1, 0]
        assert set(result["signal"].unique()) <= {0, 1}
        assert (result["size"] > 0).all()
    computed = strategy.indicators(data, GapContagionParams())
    assert computed.index.equals(data.index)
    assert not computed["entry"].iloc[:199].any()


if __name__ == "__main__":
    _self_check()
