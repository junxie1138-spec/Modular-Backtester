from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    horizon: int = 18
    estimation_window: int = 63
    variance_threshold: float = 1.15


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790875516"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.estimation_window + params.horizon))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.horizon < 2 or params.estimation_window < 2:
            raise ValueError("horizon and estimation_window must be at least 2")
        if not np.isfinite(params.variance_threshold) or params.variance_threshold <= 0:
            raise ValueError("variance_threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(lambda x: x > 0)
        out["log_return"] = np.log(close).diff()
        out["horizon_return"] = out["log_return"].rolling(
            params.horizon, min_periods=params.horizon
        ).sum()
        out["daily_std"] = out["log_return"].rolling(
            params.estimation_window, min_periods=params.estimation_window
        ).std(ddof=0)
        out["horizon_std"] = out["horizon_return"].rolling(
            params.estimation_window, min_periods=params.estimation_window
        ).std(ddof=0)
        denominator = (params.horizon * out["daily_std"].pow(2)).replace(0.0, np.nan)
        out["variance_ratio"] = out["horizon_std"].pow(2) / denominator
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        ready = indicators[["variance_ratio", "horizon_return", "ma200"]].notna().all(axis=1)
        entry = (
            ready
            & (indicators["variance_ratio"] > params.variance_threshold)
            & (indicators["horizon_return"] > 0.0)
            & (data["close"] > indicators["ma200"])
        )
        df = pd.DataFrame(index=data.index)
        # The position exits exactly when the complete entry condition flips false.
        df["signal"] = entry.fillna(False).astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=420, freq="B")
    close = 100.0 * np.exp(np.cumsum(0.001 + 0.003 * np.sin(np.arange(420) / 15.0)))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1000000.0}, index=index)
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = ((indicators["variance_ratio"] > params.variance_threshold)
                & (indicators["horizon_return"] > 0.0)
                & (data["close"] > indicators["ma200"])).astype(int)
    pd.testing.assert_series_equal(result["signal"], expected.shift(1).fillna(0).astype(int), check_names=False)
    assert result["signal"].eq(1).any() and result["signal"].eq(0).any()
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
    prefix = data.iloc[:300]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    pd.testing.assert_frame_equal(result.iloc[:300], prefix_result)
