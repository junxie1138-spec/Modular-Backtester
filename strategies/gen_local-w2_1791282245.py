from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 4
    smooth_bars: int = 3
    vol_bars: int = 20
    threshold: float = 0.08
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791282245"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.vol_bars + 1, params.roc_bars + params.smooth_bars + 1) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.roc_bars, params.smooth_bars) < 1 or params.vol_bars < 2:
            raise ValueError("Lookbacks must be positive; vol_bars must be at least two.")
        if not np.isfinite(params.threshold) or params.threshold < 0:
            raise ValueError("threshold must be finite and nonnegative.")
        if not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive.")
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(close).diff()
        out["gap"] = np.log(opening / close.shift(1))
        out["roc"] = out["return"].rolling(params.roc_bars).sum()
        out["acceleration"] = out["roc"].diff()
        out["gap_acceleration"] = out["gap"] - out["gap"].shift(params.roc_bars)
        out["session_acceleration"] = out["acceleration"] - out["gap_acceleration"]
        out["smoothed_acceleration"] = out["session_acceleration"].rolling(params.smooth_bars).mean()
        out["daily_vol"] = out["return"].rolling(params.vol_bars).std(ddof=0)
        out["annual_vol"] = out["daily_vol"] * np.sqrt(252.0)
        out["score"] = out["smoothed_acceleration"] / (np.sqrt(2.0) * out["daily_vol"].clip(lower=1e-6))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        bullish = indicators["score"].gt(params.threshold).fillna(False)
        confirmed = bullish & bullish.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        # A false bullish predicate immediately exits; reentry needs two fresh confirmations.
        df["signal"] = confirmed.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        exposure = (params.target_vol / indicators["annual_vol"].clip(lower=1e-6)).clip(lower=0.01, upper=0.95)
        df["size"] = exposure.shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    features = pd.DataFrame(
        {"score": [np.nan, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
         "annual_vol": [0.2] * 8}, index=index,
    )
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(
        data, features, StrategyContext("SPY", "1d", 0), Params(),
    ).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1]
    assert np.allclose(result["size"].iloc[1:], 0.6)
    assert np.isfinite(result["size"]).all() and result["size"].gt(0).all()
