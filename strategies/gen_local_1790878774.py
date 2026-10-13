from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DistributedDriftParams:
    drift_window: int = 5
    rank_window: int = 63
    entry_percentile: float = 0.75


class GeneratedStrategy(BaseStrategy[DistributedDriftParams]):
    strategy_id = "gen_local_1790878774"

    @classmethod
    def params_type(cls):
        return DistributedDriftParams

    def warmup_bars(self, params: DistributedDriftParams) -> int:
        return params.drift_window + params.rank_window + 2

    def indicators(self, data: pd.DataFrame, params: DistributedDriftParams) -> pd.DataFrame:
        if params.drift_window < 2 or params.rank_window < 2:
            raise ValueError("Lookback windows must be at least two.")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        window = params.drift_window
        out["net_return"] = out["return"].rolling(window, min_periods=window).sum()
        out["largest_gain"] = out["return"].clip(lower=0).rolling(window, min_periods=window).max()
        out["total_motion"] = out["return"].abs().rolling(window, min_periods=window).sum()
        out["distributed_drift"] = (
            (out["net_return"] - out["largest_gain"])
            / out["total_motion"].where(out["total_motion"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["strength_percentile"] = out["distributed_drift"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method="average", pct=True)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DistributedDriftParams,
    ) -> SignalFrame:
        qualified = (
            indicators["distributed_drift"].gt(0)
            & indicators["strength_percentile"].ge(params.entry_percentile)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Two consecutive qualifying closes enter; the first failed close exits.
        df["signal"] = (qualified & qualified.shift(1, fill_value=False)).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = DistributedDriftParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "distributed_drift": [np.nan, 0.2, 0.2, 0.2, -0.1, 0.2, 0.2, 0.2],
        "strength_percentile": [np.nan, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1]
    assert result["size"].gt(0).all()
