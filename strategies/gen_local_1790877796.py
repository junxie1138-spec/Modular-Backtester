from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownBreakoutParams:
    peak_window: int = 63
    depth_window: int = 15
    expansion_window: int = 8
    min_depth: float = 0.025
    recovery_fraction: float = 0.60
    expansion_share: float = 0.58


class GeneratedStrategy(BaseStrategy[DrawdownBreakoutParams]):
    strategy_id = "gen_local_1790877796"

    @classmethod
    def params_type(cls):
        return DrawdownBreakoutParams

    def warmup_bars(self, params: DrawdownBreakoutParams) -> int:
        return int(max(params.peak_window + params.depth_window - 1,
                       params.expansion_window + 1))

    def indicators(self, data: pd.DataFrame, params: DrawdownBreakoutParams) -> pd.DataFrame:
        if min(params.peak_window, params.depth_window, params.expansion_window) < 2:
            raise ValueError("Lookback windows must be at least two bars.")
        if not (0 < params.min_depth < 1 and 0 < params.recovery_fraction < 1
                and 0.5 < params.expansion_share < 1):
            raise ValueError("Invalid depth, recovery, or expansion thresholds.")
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["high"].rolling(params.peak_window).max()
        out["depth"] = (1.0 - data["close"] / out["peak"].where(out["peak"] > 0)).clip(lower=0)
        out["deepest"] = out["depth"].rolling(params.depth_window).max()
        out["recovery"] = 1.0 - out["depth"] / out["deepest"].where(out["deepest"] > 0)
        out["up_expansion"] = data["high"].diff().clip(lower=0)
        out["down_expansion"] = (-data["low"].diff()).clip(lower=0)
        out["up_total"] = out["up_expansion"].rolling(params.expansion_window).sum()
        out["down_total"] = out["down_expansion"].rolling(params.expansion_window).sum()
        out["total_expansion"] = out["up_total"] + out["down_total"]
        out["expansion_share"] = out["up_total"] / out["total_expansion"].where(out["total_expansion"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DrawdownBreakoutParams) -> SignalFrame:
        # Recovery breaks out of the recent underwater envelope.
        recovered = (indicators["deepest"].ge(params.min_depth)
                     & indicators["recovery"].ge(params.recovery_fraction))
        expanding = indicators["expansion_share"].ge(params.expansion_share)
        # The same AND condition governs entry and reversal exit.
        df = pd.DataFrame(index=data.index)
        df["signal"] = (recovered & expanding).fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = DrawdownBreakoutParams()
    close = np.r_[np.full(80, 100.0), np.linspace(99, 90, 10),
                  np.linspace(91, 103, 13), np.linspace(102, 94, 9)]
    data = pd.DataFrame({"open": close, "high": close + 0.1,
                         "low": close - 0.1, "close": close, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=len(close), freq="B"))
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d",
                          warmup_bars=strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    result = strategy.generate_signals(data, ind, ctx, params).data
    expected = ((ind["deepest"] >= params.min_depth)
                & (ind["recovery"] >= params.recovery_fraction)
                & (ind["expansion_share"] >= params.expansion_share))
    assert np.array_equal(result["signal"], expected.shift(1).fillna(False).astype(int))
    assert result["signal"].eq(1).any() and result["signal"].diff().eq(-1).any()
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
    assert result["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()
    prefix = data.iloc[:-5]
    prefix_ind = strategy.indicators(prefix, params)
    prefix_result = strategy.generate_signals(prefix, prefix_ind, ctx, params).data
    pd.testing.assert_frame_equal(result.loc[prefix.index], prefix_result)
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    flat_ind = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, flat_ind, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
