from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OverflowParams:
    capacity_bars: int = 6
    cluster_bars: int = 3
    overflow_budget: float = 1.0
    dominance: float = 0.75


class GeneratedStrategy(BaseStrategy[OverflowParams]):
    strategy_id = "gen_local-w2_1791268319"

    @classmethod
    def params_type(cls) -> type[OverflowParams]:
        return OverflowParams

    def warmup_bars(self, params: OverflowParams) -> int:
        if params.capacity_bars < 1 or params.cluster_bars < 1:
            raise ValueError("Lookbacks must be positive")
        if params.capacity_bars + params.cluster_bars > 10:
            raise ValueError("Combined warmup must not exceed ten bars")
        if not np.isfinite(params.overflow_budget) or params.overflow_budget <= 0:
            raise ValueError("Overflow budget must be finite and positive")
        if not np.isfinite(params.dominance) or not 0.5 < params.dominance <= 1.0:
            raise ValueError("Dominance must be in (0.5, 1.0]")
        return int(params.capacity_bars + params.cluster_bars)

    def indicators(self, data: pd.DataFrame, params: OverflowParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        volume = data["volume"].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = volume.where(volume >= 0)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["capacity"] = volume.shift(1).rolling(
            params.capacity_bars, min_periods=params.capacity_bars
        ).median()
        out["close_change"] = close.diff()
        out["relative_volume"] = volume / out["capacity"].where(out["capacity"] > 0)
        out["excess_volume"] = (out["relative_volume"] - 1.0).clip(lower=0)
        out["signed_excess"] = np.sign(out["close_change"]) * out["excess_volume"]
        out["overflow"] = out["signed_excess"].rolling(
            params.cluster_bars, min_periods=params.cluster_bars
        ).sum()
        out["total_excess"] = out["excess_volume"].rolling(
            params.cluster_bars, min_periods=params.cluster_bars
        ).sum()
        out["dominance"] = out["overflow"] / out["total_excess"].where(
            out["total_excess"] > 0
        )
        long_entry = (
            (out["overflow"] >= params.overflow_budget)
            & (out["dominance"] >= params.dominance)
        ).fillna(False)
        short_entry = (
            (out["overflow"] <= -params.overflow_budget)
            & (out["dominance"] <= -params.dominance)
        ).fillna(False)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OverflowParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, event in enumerate(indicators["entry"].fillna(0).to_numpy(dtype=int)):
            if position != 0 and event == -position:
                position = 0
            elif position == 0 and event != 0:
                position = int(event)
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = OverflowParams()
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    close = np.arange(30, dtype=float) + 100.0
    data = pd.DataFrame({
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": [100.0] * 6 + [180.0] * 3 + [100.0] * 21,
    }, index=index)
    derived = strategy.indicators(data, params)
    assert strategy.warmup_bars(params) == 9
    assert derived["entry"].iloc[:8].eq(0).all()
    assert derived["entry"].iloc[8] == 1
    events = pd.DataFrame({"entry": [1, 1, 0, -1, -1, 0]}, index=index[:6])
    result = strategy.generate_signals(data.iloc[:6], events, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, -1]
    assert result.data["size"].gt(0).all()
