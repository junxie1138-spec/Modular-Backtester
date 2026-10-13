from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeReuseParams:
    cluster_bars: int = 4
    rank_bars: int = 60
    compression_cutoff: float = 0.30
    pressure_floor: float = 0.10


class GeneratedStrategy(BaseStrategy[RangeReuseParams]):
    strategy_id = "gen_local_1790931162"

    @classmethod
    def params_type(cls):
        return RangeReuseParams

    def warmup_bars(self, params: RangeReuseParams) -> int:
        return int(params.cluster_bars + params.rank_bars - 1)

    def indicators(self, data: pd.DataFrame, params: RangeReuseParams) -> pd.DataFrame:
        if params.cluster_bars < 2 or params.rank_bars < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0.0 < params.compression_cutoff < 1.0:
            raise ValueError("compression_cutoff must lie between zero and one.")
        if not 0.0 <= params.pressure_floor < 1.0:
            raise ValueError("pressure_floor must lie between zero and one.")
        out = pd.DataFrame(index=data.index)
        n = params.cluster_bars
        out["bar_range"] = (data["high"] - data["low"]).clip(lower=0.0)
        out["range_sum"] = out["bar_range"].rolling(n, min_periods=n).sum()
        out["envelope_high"] = data["high"].rolling(n, min_periods=n).max()
        out["envelope_low"] = data["low"].rolling(n, min_periods=n).min()
        out["envelope_span"] = out["envelope_high"] - out["envelope_low"]
        denominator = out["range_sum"].replace(0.0, np.nan)
        out["reuse_ratio"] = out["envelope_span"] / denominator
        out["compression_rank"] = out["reuse_ratio"].rolling(
            params.rank_bars, min_periods=params.rank_bars
        ).rank(pct=True)
        out["body"] = data["close"] - data["open"]
        out["body_sum"] = out["body"].rolling(n, min_periods=n).sum()
        out["pressure"] = (out["body_sum"] / denominator).clip(-1.0, 1.0)
        out["ready"] = (
            out["compression_rank"].notna() & out["pressure"].notna()
        ).astype(int)
        out["compression_rank"] = out["compression_rank"].fillna(1.0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeReuseParams,
    ) -> SignalFrame:
        rank = indicators["compression_rank"]
        pressure = indicators["pressure"]
        entry = (
            indicators["ready"].eq(1)
            & rank.le(params.compression_cutoff)
            & pressure.gt(params.pressure_floor)
        )
        # Eligibility reversal is the sole exit; no timed or price-based exit.
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry.astype(int)
        compression_strength = (
            1.0 - rank / params.compression_cutoff
        ).clip(0.0, 1.0)
        buying_strength = (
            (pressure - params.pressure_floor) / (1.0 - params.pressure_floor)
        ).clip(0.0, 1.0)
        df["size"] = (
            0.20 + 0.75 * np.sqrt(compression_strength * buying_strength)
        ).shift(1).fillna(0.20).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RangeReuseParams()
    index = pd.date_range("2020-01-01", periods=100)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    computed = strategy.indicators(data, params)
    assert computed.index.equals(data.index)
    assert np.isfinite(computed.to_numpy()).all()
    assert computed["ready"].iloc[:strategy.warmup_bars(params) - 1].eq(0).all()
    assert strategy.generate_signals(data, computed, None, params).data["signal"].eq(0).all()
    probe = pd.DataFrame(
        {"compression_rank": [0.10, 0.20, 0.10, 0.70],
         "pressure": [0.20, 0.80, -0.10, 0.30],
         "ready": [1, 1, 1, 1]}, index=index[:4]
    )
    result = strategy.generate_signals(data.iloc[:4], probe, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    assert result["size"].between(0.20, 0.95).all()
    assert result["size"].iloc[1] != result["size"].iloc[2]
