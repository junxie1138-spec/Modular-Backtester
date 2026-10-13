from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VarianceShareParams:
    cycle_bars: int = 8
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[VarianceShareParams]):
    strategy_id = "gen_local_1791194466"

    @classmethod
    def params_type(cls):
        return VarianceShareParams

    def warmup_bars(self, params: VarianceShareParams) -> int:
        return max(2, int(params.cycle_bars)) + 63

    def indicators(self, data: pd.DataFrame, params: VarianceShareParams) -> pd.DataFrame:
        n = max(2, int(params.cycle_bars))
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["squared_return"] = out["return"].pow(2)
        out["downside_squared_return"] = out["return"].clip(upper=0).pow(2)
        out["total_variance"] = out["squared_return"].rolling(n, min_periods=n).sum()
        out["downside_variance"] = out["downside_squared_return"].rolling(n, min_periods=n).sum()
        out["downside_share"] = out["downside_variance"].div(out["total_variance"].where(out["total_variance"] > 0))
        out["share_percentile"] = out["downside_share"].rolling(63, min_periods=63).rank(method="average", pct=True)
        out["annual_vol"] = out["return"].rolling(n, min_periods=n).std(ddof=0) * np.sqrt(252.0)
        out["exposure"] = (float(params.target_vol) / out["annual_vol"].clip(lower=0.01)).clip(lower=0.01, upper=0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VarianceShareParams) -> SignalFrame:
        percentile = indicators["share_percentile"]
        entries = np.select([percentile >= 0.8, percentile <= 0.2], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["exposure"].shift(1).fillna(0.01).clip(lower=0.01, upper=0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"share_percentile": [0.9, 0.5, 0.1, 0.1, 0.5, 0.9, 0.9], "exposure": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, VarianceShareParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.allclose(result["size"], [0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    derived = GeneratedStrategy.indicators(None, data, VarianceShareParams())
    assert derived["share_percentile"].isna().all()
    assert GeneratedStrategy.warmup_bars(None, VarianceShareParams()) == 71


if __name__ == "__main__":
    _self_check()
