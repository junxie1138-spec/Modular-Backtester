from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RetainedAccelerationParams:
    roc_bars: int = 4
    retention_bars: int = 8
    percentile_bars: int = 126
    entry_percentile: float = 0.65


class GeneratedStrategy(BaseStrategy[RetainedAccelerationParams]):
    strategy_id = "gen_local-w2_1791302748"

    @classmethod
    def params_type(cls):
        return RetainedAccelerationParams

    def warmup_bars(self, params: RetainedAccelerationParams) -> int:
        if min(params.roc_bars, params.percentile_bars) < 1 or params.retention_bars < 2:
            raise ValueError("Lookbacks must be positive; retention_bars must be at least two")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one")
        return int(params.roc_bars + params.retention_bars + params.percentile_bars + 1)

    def indicators(self, data: pd.DataFrame, params: RetainedAccelerationParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["positive_acceleration"] = out["acceleration"].clip(lower=0)
        window = params.retention_bars
        out["gross_push"] = out["positive_acceleration"].rolling(window).sum()
        out["net_gain"] = out["acceleration"].rolling(window).sum()
        out["largest_push"] = out["positive_acceleration"].rolling(window).max()
        denominator = out["gross_push"].replace(0.0, np.nan)
        out["retained_fraction"] = out["net_gain"] / denominator
        out["distributed_fraction"] = 1.0 - out["largest_push"] / denominator
        out["score"] = (
            out["retained_fraction"] * out["distributed_fraction"]
        ).mask(out["gross_push"].eq(0.0), 0.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["threshold"] = out["score"].shift(1).rolling(
            params.percentile_bars
        ).quantile(params.entry_percentile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RetainedAccelerationParams,
    ) -> SignalFrame:
        qualifies = (
            indicators["roc"].gt(0.0)
            & indicators["score"].gt(0.0)
            & indicators["score"].gt(indicators["threshold"])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # A false qualifying condition reverses the entry and exits the position.
        df["signal"] = qualifies.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RetainedAccelerationParams()
    strategy = GeneratedStrategy()
    t = np.arange(500, dtype=float)
    close = 100.0 * np.exp(np.cumsum(
        0.001 + 0.004 * np.sin(t / 13.0) + 0.001 * np.sin(t / 3.0)
    ))
    data = pd.DataFrame({
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1000000.0,
    }, index=pd.date_range("2020-01-01", periods=len(t), freq="B"))
    ind = strategy.indicators(data, params)
    result = strategy.generate_signals(data, ind, None, params).data
    expected = (
        ind["roc"].gt(0) & ind["score"].gt(0)
        & ind["score"].gt(ind["threshold"])
    ).astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert result["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()
    assert result["signal"].eq(1).any()
    assert result["signal"].isin([0, 1]).all()
    assert result["size"].gt(0).all()
    pd.testing.assert_frame_equal(strategy.indicators(data.iloc[:350], params), ind.iloc[:350])
