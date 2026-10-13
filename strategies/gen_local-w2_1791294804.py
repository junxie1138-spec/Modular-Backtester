from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CaptureParams:
    rank_window: int = 60
    entry_percentile: float = 0.70

    def __post_init__(self):
        if self.rank_window < 2 or not 0.0 < self.entry_percentile < 1.0:
            raise ValueError("rank_window must be >= 2 and entry_percentile must lie in (0, 1)")


class GeneratedStrategy(BaseStrategy[CaptureParams]):
    strategy_id = "gen_local-w2_1791294804"

    @classmethod
    def params_type(cls):
        return CaptureParams

    def warmup_bars(self, params: CaptureParams) -> int:
        return params.rank_window + 4

    def indicators(self, data: pd.DataFrame, params: CaptureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda x: x > 0)
        out["daily_return"] = (data["close"] / out["previous_close"] - 1.0).replace([np.inf, -np.inf], np.nan)
        out["upside_opportunity"] = (data["high"] / out["previous_close"] - 1.0).clip(lower=0.0).replace([np.inf, -np.inf], np.nan)
        out["net_gain"] = out["daily_return"].rolling(4, min_periods=4).sum()
        out["upside_budget"] = out["upside_opportunity"].rolling(4, min_periods=4).sum()
        out["capture"] = (out["net_gain"] / out["upside_budget"].where(out["upside_budget"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["capture_percentile"] = out["capture"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CaptureParams) -> SignalFrame:
        condition = (indicators["capture_percentile"] > params.entry_percentile) & (indicators["net_gain"] > 0.0)
        # Stay long while the entry condition holds; its reversal is the only exit.
        df = pd.DataFrame(index=data.index)
        df["signal"] = condition.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = CaptureParams(rank_window=12)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=100)
    t = np.arange(len(index), dtype=float)
    close = 100.0 * np.exp(0.001 * t + 0.025 * np.sin(t / 4.0))
    data = pd.DataFrame({"open": close, "high": close * 1.006, "low": close * 0.994, "close": close, "volume": 1000000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert indicators["capture_percentile"].first_valid_index() == index[params.rank_window + 3]
    assert result["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()
    assert result["signal"].isin([0, 1]).all() and result["signal"].eq(1).any()
    assert result["size"].gt(0).all()
    prefix = data.iloc[:70]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(prefix_result, result.iloc[:70])
    controlled = pd.DataFrame({"capture_percentile": [np.nan, 0.8, 0.9, 0.6, 0.85], "net_gain": [np.nan, 0.02, 0.02, 0.02, -0.01]}, index=index[:5])
    signals = strategy.generate_signals(data.iloc[:5], controlled, ctx, params).data["signal"]
    assert signals.tolist() == [0, 0, 1, 1, 0]
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    flat_result = strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data
    assert flat_result["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
