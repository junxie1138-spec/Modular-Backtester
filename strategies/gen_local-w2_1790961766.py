from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DepthRecurrenceParams:
    fit_window: int = 30
    min_depth: float = 0.01


class GeneratedStrategy(BaseStrategy[DepthRecurrenceParams]):
    strategy_id = "gen_local-w2_1790961766"

    @classmethod
    def params_type(cls) -> type[DepthRecurrenceParams]:
        return DepthRecurrenceParams

    def warmup_bars(self, params: DepthRecurrenceParams) -> int:
        return int(params.fit_window) + 12

    def indicators(self, data: pd.DataFrame, params: DepthRecurrenceParams) -> pd.DataFrame:
        if params.fit_window < 3 or not np.isfinite(params.min_depth) or params.min_depth <= 0:
            raise ValueError("fit_window must be >= 3 and min_depth must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["close"].rolling(10, min_periods=10).max()
        out["depth"] = 1.0 - data["close"] / out["peak"].where(out["peak"] > 0)
        out["change"] = out["depth"].diff()
        out["lag1"] = out["change"].shift(1)
        out["lag2"] = out["change"].shift(2)
        w = int(params.fit_window)
        complete = out[["change", "lag1", "lag2"]].notna().all(axis=1)
        y = out["change"].where(complete)
        x1 = out["lag1"].where(complete)
        x2 = out["lag2"].where(complete)
        out["mean_y"] = y.rolling(w, min_periods=w).mean()
        out["mean_1"] = x1.rolling(w, min_periods=w).mean()
        out["mean_2"] = x2.rolling(w, min_periods=w).mean()
        out["var_1"] = x1.rolling(w, min_periods=w).var(ddof=0)
        out["var_2"] = x2.rolling(w, min_periods=w).var(ddof=0)
        out["cov_12"] = x1.rolling(w, min_periods=w).cov(x2, ddof=0)
        out["cov_1y"] = x1.rolling(w, min_periods=w).cov(y, ddof=0)
        out["cov_2y"] = x2.rolling(w, min_periods=w).cov(y, ddof=0)
        out["determinant"] = out["var_1"] * out["var_2"] - out["cov_12"] ** 2
        denominator = out["determinant"].where(
            out["determinant"] > 1e-8 * out["var_1"] * out["var_2"]
        )
        out["beta_1"] = (out["cov_1y"] * out["var_2"] - out["cov_2y"] * out["cov_12"]) / denominator
        out["beta_2"] = (out["cov_2y"] * out["var_1"] - out["cov_1y"] * out["cov_12"]) / denominator
        out["intercept"] = out["mean_y"] - out["beta_1"] * out["mean_1"] - out["beta_2"] * out["mean_2"]
        out["forecast_change"] = out["intercept"] + out["beta_1"] * out["change"] + out["beta_2"] * out["lag1"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DepthRecurrenceParams) -> SignalFrame:
        entry = (
            (indicators["depth"] >= params.min_depth)
            & (indicators["beta_2"] < 0.0)
            & (indicators["forecast_change"] < 0.0)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Remain long only while the entry condition holds; its reversal exits.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = DepthRecurrenceParams()
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    close = 100.0 + 3.0 * np.sin(np.arange(240) * 1.1)
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000000.0}, index=index)
    indicators = GeneratedStrategy.indicators(None, data, params)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    entry = ((indicators["depth"] >= params.min_depth) & (indicators["beta_2"] < 0.0) & (indicators["forecast_change"] < 0.0)).fillna(False)
    expected = entry.astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert result.index.equals(data.index) and result["size"].gt(0).all()
    assert result["signal"].isin([0, 1]).all()
    assert result["signal"].iloc[:GeneratedStrategy.warmup_bars(None, params)].eq(0).all()
    prefix = data.iloc[:180]
    earlier = GeneratedStrategy.indicators(None, prefix, params)
    assert np.allclose(earlier.to_numpy(), indicators.iloc[:180].to_numpy(), equal_nan=True)
