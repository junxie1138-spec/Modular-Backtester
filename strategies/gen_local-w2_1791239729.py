from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendInnovationParams:
    trend_window: int = 5
    range_window: int = 20
    threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[TrendInnovationParams]):
    strategy_id = "gen_local-w2_1791239729"

    @classmethod
    def params_type(cls) -> type[TrendInnovationParams]:
        return TrendInnovationParams

    def warmup_bars(self, params: TrendInnovationParams) -> int:
        return max(params.trend_window, params.range_window) + 1

    def indicators(self, data: pd.DataFrame, params: TrendInnovationParams) -> pd.DataFrame:
        if params.trend_window < 2 or params.range_window < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        n = params.trend_window
        close = data["close"].astype(float)
        midpoint = (n - 1) / 2.0
        weighted = pd.Series(0.0, index=data.index)
        for lag in range(n):
            weighted = weighted + (midpoint - lag) * close.shift(lag)
        slope = weighted / (n * (n * n - 1) / 12.0)
        level = close.rolling(n, min_periods=n).mean()
        forecast = level.shift(1) + ((n + 1) / 2.0) * slope.shift(1)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        scale = true_range.rolling(params.range_window, min_periods=params.range_window).mean().shift(1)
        scale = scale.where(scale > 0)
        out = pd.DataFrame({
            "regression_level": level,
            "latent_slope": slope,
            "forecast": forecast,
            "true_range": true_range,
            "range_scale": scale,
            "forecast_breakout": (close - forecast) / scale,
            "slope_innovation": n * slope.diff() / scale,
        }, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: TrendInnovationParams) -> SignalFrame:
        scores = indicators[["forecast_breakout", "slope_innovation"]]
        eligible = scores.notna().all(axis=1) & scores.gt(params.threshold).all(axis=1)
        # The same predicate enters and holds; its reversal is the only exit.
        df = pd.DataFrame(index=data.index)
        df["signal"] = eligible.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = TrendInnovationParams()
    close = np.full(80, 100.0)
    close[40:42] = [105.0, 108.0]
    data = pd.DataFrame({"open": close, "high": close + 1.0,
                         "low": close - 1.0, "close": close, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=len(close)))
    ind = GeneratedStrategy.indicators(None, data, params)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].iloc[40] == 0
    assert result["signal"].iloc[41] == 1
    assert result["signal"].iloc[43] == 0
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
    prefix = GeneratedStrategy.indicators(None, data.iloc[:50], params)
    pd.testing.assert_frame_equal(ind.iloc[:50], prefix)


if __name__ == "__main__":
    _self_check()
