from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DiffusionParams:
    drift_bars: int = 4
    volatility_bars: int = 20
    entry_threshold: float = 0.25
    target_volatility: float = 0.12


class GeneratedStrategy(BaseStrategy[DiffusionParams]):
    strategy_id = "gen_local-w2_1791316997"

    @classmethod
    def params_type(cls):
        return DiffusionParams

    def warmup_bars(self, params: DiffusionParams) -> int:
        return max(params.drift_bars + 1, params.volatility_bars + 2)

    def indicators(self, data: pd.DataFrame, params: DiffusionParams) -> pd.DataFrame:
        if params.drift_bars < 2 or params.volatility_bars < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not np.isfinite(params.entry_threshold) or params.entry_threshold < 0:
            raise ValueError("Entry threshold must be finite and nonnegative.")
        if not np.isfinite(params.target_volatility) or params.target_volatility <= 0:
            raise ValueError("Target volatility must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["squared_return"] = out["return"].pow(2)
        out["adjacent_product"] = out["return"].abs() * out["return"].shift(1).abs()
        out["realized_variance"] = out["squared_return"].rolling(params.volatility_bars, min_periods=params.volatility_bars).mean()
        out["bipower_variance"] = (np.pi / 2.0) * out["adjacent_product"].rolling(params.volatility_bars, min_periods=params.volatility_bars).mean()
        out["drift"] = out["return"].rolling(params.drift_bars, min_periods=params.drift_bars).sum()
        out["score"] = out["drift"] / np.sqrt(params.drift_bars * out["bipower_variance"].clip(lower=1e-12))
        out["continuous_share"] = (out["bipower_variance"] / out["realized_variance"].clip(lower=1e-12)).clip(0.0, 1.0)
        out["annual_volatility"] = np.sqrt(252.0 * out["realized_variance"].clip(lower=1e-12))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DiffusionParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        ready = indicators[["score", "continuous_share", "annual_volatility"]].notna().all(axis=1)
        # The same condition opens and maintains the position; its reversal exits.
        df["signal"] = (ready & indicators["score"].gt(params.entry_threshold)).astype(int)
        excess = (indicators["score"] - params.entry_threshold).clip(lower=0.0)
        conviction = 0.15 + 0.85 * excess / (1.0 + excess)
        exposure = params.target_volatility / indicators["annual_volatility"]
        df["size"] = (exposure * conviction * indicators["continuous_share"]).replace([np.inf, -np.inf], np.nan).fillna(0.01).clip(0.01, 0.95)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = DiffusionParams()
    strategy = GeneratedStrategy()
    returns = np.tile(np.r_[np.full(8, 0.004), np.full(8, -0.004)], 8)
    close = 100.0 * np.exp(np.cumsum(returns))
    data = pd.DataFrame({key: close for key in ("open", "high", "low", "close")}, index=pd.date_range("2020-01-01", periods=len(close)))
    data["volume"] = 1000000.0
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    expected = indicators["score"].gt(params.entry_threshold).astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert set(result["signal"].unique()) == {0, 1}
    assert (result["signal"].diff() == -1).any()
    assert result["size"].between(0.01, 0.95).all()
    prefix = data.iloc[:75]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(result.iloc[:75], prefix_result)
