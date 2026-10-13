from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VarianceParams:
    vol_window: int = 16
    reversal_threshold: float = 0.5
    target_daily_vol: float = 0.008
    min_size: float = 0.05
    max_size: float = 0.95

    def __post_init__(self):
        if self.vol_window < 2 or self.reversal_threshold <= 0:
            raise ValueError("Require vol_window >= 2 and reversal_threshold > 0")
        if not 0 < self.min_size <= self.max_size <= 1:
            raise ValueError("Require 0 < min_size <= max_size <= 1")
        if not np.isfinite(self.target_daily_vol) or self.target_daily_vol <= 0:
            raise ValueError("Require a finite positive target_daily_vol")
        if not np.isfinite(self.reversal_threshold):
            raise ValueError("Require a finite reversal_threshold")


class GeneratedStrategy(BaseStrategy[VarianceParams]):
    strategy_id = "gen_local-w2_1790941498"

    @classmethod
    def params_type(cls):
        return VarianceParams

    def warmup_bars(self, params: VarianceParams) -> int:
        return params.vol_window + 2

    def indicators(self, data: pd.DataFrame, params: VarianceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        out["lag_vol"] = out["return"].rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0).shift(1).where(lambda s: s > 1e-8)
        out["signed_energy"] = out["return"] * out["return"].abs()
        out["energy_change"] = (
            -out["signed_energy"].diff() / out["lag_vol"].pow(2)
        ).replace([np.inf, -np.inf], np.nan)
        out["magnitude"] = out["energy_change"].abs()
        out["confidence"] = out["magnitude"] / (
            params.reversal_threshold + out["magnitude"]
        )
        out["target_size"] = (
            params.target_daily_vol / out["lag_vol"] * out["confidence"]
        ).clip(params.min_size, params.max_size).fillna(params.min_size)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VarianceParams,
    ) -> SignalFrame:
        score = indicators["energy_change"].fillna(0.0).to_numpy()
        targets = indicators["target_size"].to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        entry_size = params.min_size
        for i, value in enumerate(score):
            desired = 1 if value >= params.reversal_threshold else (
                -1 if value <= -params.reversal_threshold else 0
            )
            if position == 0:
                if desired:
                    position = desired
                    entry_size = float(targets[i])
            elif desired == -position:
                position = 0
            signals[i] = position
            sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = VarianceParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    close = 100 * np.exp(np.cumsum(0.01 * np.sin(np.arange(80))))
    data = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000000.0,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    test_indicators = pd.DataFrame({
        "energy_change": [2, 0, -2, -2, 0, 2, 2, 0],
        "target_size": np.arange(2, 10) / 10,
    }, index=index[:8])
    result = strategy.generate_signals(data.iloc[:8], test_indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert np.allclose(result["size"], [0.05, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8])
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:50], params)
    assert np.allclose(full.iloc[:50], prefix, equal_nan=True)
    output = strategy.generate_signals(data, full, ctx, params).data
    partial = strategy.generate_signals(data.iloc[:50], prefix, ctx, params).data
    assert output.iloc[:50].equals(partial)
    assert output["signal"].isin([-1, 0, 1]).all()
    assert output["size"].between(params.min_size, params.max_size).all()
    assert output["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()
