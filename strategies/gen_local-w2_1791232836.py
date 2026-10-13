from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeSpillParams:
    reclaim_fraction: float = 0.4
    contraction_ratio: float = 0.85
    min_drawdown: float = 0.006


class GeneratedStrategy(BaseStrategy[RangeSpillParams]):
    strategy_id = "gen_local-w2_1791232836"

    @classmethod
    def params_type(cls):
        return RangeSpillParams

    def warmup_bars(self, params: RangeSpillParams) -> int:
        return 6

    def indicators(self, data: pd.DataFrame, params: RangeSpillParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["range"] = data["high"] - data["low"]
        out["downside_spill"] = (data["low"].shift(1) - data["low"]).clip(lower=0)
        out["spill_reference"] = out["downside_spill"].rolling(2, min_periods=2).max().shift(1)
        out["low_reference"] = data["low"].rolling(2, min_periods=2).min().shift(1)
        out["high_reference"] = data["high"].rolling(2, min_periods=2).max().shift(1)
        out["range_reference"] = out["range"].rolling(2, min_periods=2).max().shift(1)
        out["peak"] = data["high"].rolling(6, min_periods=6).max()
        out["drawdown"] = 1.0 - data["close"] / out["peak"].replace(0, np.nan)
        out["reclaim"] = (data["low"] - out["low_reference"]) / out["spill_reference"].replace(0, np.nan)
        out["contraction"] = out["range"] / out["range_reference"].replace(0, np.nan)
        out["upper_capped"] = data["high"] < out["high_reference"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeSpillParams,
    ) -> SignalFrame:
        ready = indicators[["drawdown", "reclaim", "contraction", "spill_reference"]].notna().all(axis=1)
        entry = (
            ready
            & (indicators["drawdown"] >= params.min_drawdown)
            & (indicators["spill_reference"] > 0)
            & (indicators["reclaim"] >= params.reclaim_fraction)
            & (indicators["reclaim"] < 1.0)
            & (indicators["contraction"] <= params.contraction_ratio)
            & indicators["upper_capped"]
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # The position persists precisely while entry remains true; its reversal exits.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    high = np.array([110, 109, 108, 107, 106, 105, 103, 102.5, 103, 104], dtype=float)
    low = np.array([108, 107, 106, 105, 104, 101, 102.4, 102.3, 101.5, 103], dtype=float)
    data = pd.DataFrame({"open": (high + low) / 2, "high": high, "low": low,
                         "close": (high + low) / 2, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = RangeSpillParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    assert (result["size"] > 0).all()
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    assert strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
