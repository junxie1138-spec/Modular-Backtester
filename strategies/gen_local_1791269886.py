from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 20
    complexity_window: int = 6
    percentile_window: int = 126
    drawdown_percentile: float = 0.75
    expansion_percentile: float = 0.70


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791269886"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.peak_window, params.complexity_window + 3) + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.peak_window, params.complexity_window, params.percentile_window) < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not 0.0 < params.drawdown_percentile < 1.0 or not 0.0 < params.expansion_percentile < 1.0:
            raise ValueError("Percentiles must be strictly between zero and one")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        a, b, c = close.shift(2), close.shift(1), close
        motif = pd.Series(np.select([
            (a < b) & (b < c),
            (a < c) & (c < b),
            (b < a) & (a < c),
            (b < c) & (c < a),
            (c < a) & (a < b),
            (c < b) & (b < a),
        ], [0, 1, 2, 3, 4, 5], default=np.nan), index=data.index)
        transition = motif.shift(1) * 6 + motif
        complete = transition.notna().rolling(params.complexity_window, min_periods=params.complexity_window).sum().eq(params.complexity_window)
        diversity = pd.Series(0.0, index=data.index)
        for code in range(36):
            count = transition.eq(code).astype(float).where(transition.notna()).rolling(params.complexity_window, min_periods=params.complexity_window).sum()
            diversity = diversity + count.gt(0).astype(float)

        out["motif"] = motif
        out["transition"] = transition
        out["diversity"] = diversity.where(complete)
        out["expansion"] = out["diversity"].diff()
        out["expansion_threshold"] = out["expansion"].rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.expansion_percentile).shift(1)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["drawdown_threshold"] = out["drawdown"].rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.drawdown_percentile).shift(1)
        out["close_change"] = close.diff()
        out["recent_decline"] = out["close_change"].shift(1).lt(0) | out["close_change"].shift(2).lt(0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        required = ["expansion", "expansion_threshold", "drawdown", "drawdown_threshold", "close_change"]
        ready = indicators[required].notna().all(axis=1)
        entry = (
            ready
            & indicators["drawdown"].gt(0)
            & indicators["drawdown"].ge(indicators["drawdown_threshold"])
            & indicators["expansion"].gt(0)
            & indicators["expansion"].gt(indicators["expansion_threshold"])
            & indicators["close_change"].gt(0)
            & indicators["recent_decline"]
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Maintain the long position only while the entry condition remains true.
        # Its reversal is the sole exit; the recent decline gate allows at most two rising closes.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    sample_close = pd.Series([100.0, 97.0, 98.0, 96.0, 99.0, 95.0, 97.0, 98.0, 94.0, 96.0, 97.0, 93.0] * 20, index=pd.date_range("2020-01-01", periods=240))
    sample = pd.DataFrame({"open": sample_close, "high": sample_close + 1, "low": sample_close - 1, "close": sample_close, "volume": 1000.0})
    params = RecoveryParams(peak_window=5, complexity_window=3, percentile_window=8, drawdown_percentile=0.5, expansion_percentile=0.5)
    strategy = GeneratedStrategy()
    derived = strategy.indicators(sample, params)
    signals = strategy.generate_signals(sample, derived, None, params).data
    eligible = (derived["drawdown"].gt(0) & derived["drawdown"].ge(derived["drawdown_threshold"]) & derived["expansion"].gt(0) & derived["expansion"].gt(derived["expansion_threshold"]) & derived["close_change"].gt(0) & derived["recent_decline"]).fillna(False)
    assert eligible.any()
    assert signals["signal"].equals(eligible.astype(int).shift(1).fillna(0).astype(int))
    assert signals["signal"].isin([0, 1]).all() and signals["size"].gt(0).all()
    assert not signals["signal"].rolling(3).sum().eq(3).any()
