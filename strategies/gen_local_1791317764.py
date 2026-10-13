from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalSlopeParams:
    smoothing: int = 3
    innovation_threshold: float = 0.10


class GeneratedStrategy(BaseStrategy[SeasonalSlopeParams]):
    strategy_id = "gen_local_1791317764"

    @classmethod
    def params_type(cls) -> type[SeasonalSlopeParams]:
        return SeasonalSlopeParams

    def warmup_bars(self, params: SeasonalSlopeParams) -> int:
        if not isinstance(params.smoothing, int) or not 1 <= params.smoothing <= 3:
            raise ValueError("smoothing must be an integer between 1 and 3")
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold < 0:
            raise ValueError("innovation_threshold must be finite and nonnegative")
        return 10

    def indicators(self, data: pd.DataFrame, params: SeasonalSlopeParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["log_return"] = out["log_close"].diff()
        out["weekday"] = data.index.dayofweek
        # Three equally spaced log prices have OLS slope (last - first) / 2.
        out["latent_slope"] = (out["log_close"] - out["log_close"].shift(2)) / 2.0
        out["seasonal_reference"] = np.nan
        for lag in (3, 4, 5):
            candidate = out["latent_slope"].shift(lag).where(
                out["weekday"].eq(out["weekday"].shift(lag))
            )
            out[f"reference_{lag}"] = candidate
            out["seasonal_reference"] = out["seasonal_reference"].combine_first(candidate)
        out["innovation"] = out["latent_slope"] - out["seasonal_reference"]
        out["smoothed_innovation"] = out["innovation"].rolling(
            params.smoothing, min_periods=params.smoothing
        ).mean()
        out["smoothed_slope"] = out["latent_slope"].rolling(
            params.smoothing, min_periods=params.smoothing
        ).mean()
        out["noise"] = out["log_return"].rolling(9, min_periods=9).std(ddof=0)
        out["innovation_score"] = (
            out["smoothed_innovation"] / out["noise"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalSlopeParams,
    ) -> SignalFrame:
        entry_condition = (
            indicators["innovation_score"].gt(params.innovation_threshold)
            & indicators["smoothed_slope"].gt(0.0)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # The desired position becomes flat exactly when the entry condition flips false.
        df["signal"] = entry_condition.astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.bdate_range("2020-01-01", periods=160)
    close = 100.0 * np.exp(np.cumsum(0.003 * np.sin(np.arange(160) / 8.0)))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    strategy = GeneratedStrategy()
    params = SeasonalSlopeParams()
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, None, params).data
    expected = ((derived["innovation_score"] > params.innovation_threshold) & (derived["smoothed_slope"] > 0)).astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert result["signal"].iloc[:10].eq(0).all()
    assert result["signal"].isin([0, 1]).all() and result["signal"].sum() > 0
    assert result["size"].gt(0).all()
    prefix = data.iloc[:100]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    assert prefix_result.equals(result.iloc[:100])
    assert strategy.warmup_bars(params) <= 10
