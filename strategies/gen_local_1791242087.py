from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalSlopeParams:
    slope_window: int = 5
    noise_window: int = 20
    innovation_threshold: float = 0.1


class GeneratedStrategy(BaseStrategy[SeasonalSlopeParams]):
    strategy_id = "gen_local_1791242087"

    @classmethod
    def params_type(cls) -> type[SeasonalSlopeParams]:
        return SeasonalSlopeParams

    @staticmethod
    def warmup_bars(params: SeasonalSlopeParams) -> int:
        return int(max(params.slope_window, params.noise_window) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalSlopeParams) -> pd.DataFrame:
        if params.slope_window < 2 or params.noise_window < 2:
            raise ValueError("Lookback windows must be at least two bars.")
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold < 0:
            raise ValueError("Innovation threshold must be finite and nonnegative.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        out["log_return"] = out["log_close"].diff()
        coordinates = np.arange(params.slope_window, dtype=float)
        weights = coordinates - coordinates.mean()
        denominator = float(np.dot(weights, weights))
        out["trend_slope"] = sum(
            out["log_close"].shift(lag) * weights[params.slope_window - 1 - lag]
            for lag in range(params.slope_window)
        ) / denominator
        out["slope_innovation"] = out["trend_slope"].diff()
        out["return_noise"] = out["log_return"].rolling(
            params.noise_window, min_periods=params.noise_window
        ).std(ddof=0)
        out["normalized_innovation"] = (
            out["slope_innovation"] / out["return_noise"].where(out["return_noise"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["weekday"] = data.index.dayofweek
        out["seasonal_window"] = out["weekday"].isin([1, 2])
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalSlopeParams,
    ) -> SignalFrame:
        innovation = indicators["normalized_innovation"]
        eligible = (
            indicators["seasonal_window"]
            & innovation.notna()
            & innovation.gt(params.innovation_threshold)
        )
        df = pd.DataFrame(index=data.index)
        # The same predicate enters and maintains exposure; its reversal exits.
        # Tuesday/Wednesday eligibility limits each position to two trading bars.
        df["signal"] = eligible.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.bdate_range("2020-01-01", periods=160)
    steps = np.arange(len(index), dtype=float)
    close = 100.0 * np.exp(0.001 * steps + 0.02 * np.sin(steps / 2.0))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1000000.0},
        index=index,
    )
    params = SeasonalSlopeParams()
    indicators = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    expected = (
        indicators["seasonal_window"]
        & indicators["normalized_innovation"].gt(params.innovation_threshold)
    ).astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert result["signal"].isin([0, 1]).all() and result["signal"].sum() > 0
    assert result["size"].gt(0).all()
    assert not result["signal"].rolling(3).sum().eq(3).any()
    truncated = GeneratedStrategy.indicators(data.iloc[:100], params)
    pd.testing.assert_frame_equal(indicators.iloc[:100], truncated)
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    flat_indicators = GeneratedStrategy.indicators(flat, params)
    flat_signals = GeneratedStrategy.generate_signals(flat, flat_indicators, None, params).data
    assert flat_signals["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
