from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class LiquidityParams:
    smooth_bars: int = 5
    rank_bars: int = 63
    memory_bars: int = 10
    vol_bars: int = 20
    entry_percentile: float = 0.8
    hold_bars: int = 8
    target_vol: float = 0.12
    max_size: float = 0.95

    def __post_init__(self):
        if min(self.smooth_bars, self.rank_bars, self.memory_bars, self.vol_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if self.hold_bars < 1 or not 0.5 < self.entry_percentile < 1.0:
            raise ValueError("Invalid hold duration or percentile.")
        if not 0.0 < self.target_vol or not 0.0 < self.max_size <= 1.0:
            raise ValueError("Invalid volatility target or exposure cap.")


class GeneratedStrategy(BaseStrategy[LiquidityParams]):
    strategy_id = "gen_local_1790856573"

    @classmethod
    def params_type(cls) -> type[LiquidityParams]:
        return LiquidityParams

    def warmup_bars(self, params: LiquidityParams) -> int:
        return int(max(params.smooth_bars + params.rank_bars + params.memory_bars + 1,
                       params.vol_bars + 1))

    def indicators(self, data: pd.DataFrame, params: LiquidityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["relative_range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"]) / close
        out["dollar_volume"] = close * data["volume"].where(data["volume"] > 0)
        out["liquidity"] = (out["dollar_volume"] / out["relative_range"]).replace(
            [np.inf, -np.inf], np.nan)
        out["smoothed_liquidity"] = out["liquidity"].rolling(params.smooth_bars).mean()
        out["liquidity_rank"] = out["smoothed_liquidity"].rolling(
            params.rank_bars).rank(pct=True)
        out["prior_rank"] = out["liquidity_rank"].shift(1)
        out["recent_low_rank"] = out["prior_rank"].rolling(params.memory_bars).min()
        out["momentum"] = close.pct_change(params.smooth_bars, fill_method=None)
        out["annual_vol"] = out["return"].rolling(params.vol_bars).std(ddof=0) * np.sqrt(252.0)
        out["entry_ready"] = (
            (out["liquidity_rank"] >= params.entry_percentile)
            & (out["prior_rank"] < params.entry_percentile)
            & (out["recent_low_rank"] <= 1.0 - params.entry_percentile)
            & (out["momentum"] > 0)
            & (out["annual_vol"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: LiquidityParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.max_size, dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        vols = indicators["annual_vol"].to_numpy(dtype=float)
        exit_bar = -1
        entry_size = params.max_size
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signals[i] = 1
                sizes[i] = entry_size
            elif ready[i] and np.isfinite(vols[i]) and vols[i] > 0:
                entry_size = float(np.clip(params.target_vol / vols[i], 1e-6, params.max_size))
                exit_bar = i + params.hold_bars
                signals[i] = 1
                sizes[i] = entry_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = LiquidityParams()
    strategy = GeneratedStrategy()
    sample = pd.DataFrame(index=pd.date_range("2020-01-01", periods=20))
    derived = pd.DataFrame({"entry_ready": True, "annual_vol": 0.2}, index=sample.index)
    derived.loc[sample.index[1:], "annual_vol"] = 0.4
    result = strategy.generate_signals(sample, derived, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] + [1] * 8 + [0, 1]
    assert np.allclose(result["size"].iloc[1:9], 0.6)
    assert np.allclose(result["size"].iloc[10:18], 0.3)
    assert result["size"].gt(0).all()
