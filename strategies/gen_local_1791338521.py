from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PathParams:
    path_bars: int = 4
    compression_ratio: float = 0.85


class GeneratedStrategy(BaseStrategy[PathParams]):
    strategy_id = "gen_local_1791338521"

    @classmethod
    def params_type(cls):
        return PathParams

    def warmup_bars(self, params: PathParams) -> int:
        return 20 + params.path_bars

    def indicators(self, data: pd.DataFrame, params: PathParams) -> pd.DataFrame:
        n = params.path_bars
        if n < 2 or not 0.0 < params.compression_ratio <= 1.0:
            raise ValueError("path_bars must be >= 2; compression_ratio must be in (0, 1].")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["relative_range"] = ((data["high"] - data["low"]) / close).where(
            data["high"] > data["low"]
        )
        out["price_step"] = np.log(close).diff()
        out["range_step"] = np.log(out["relative_range"]).diff()
        x, y = out["price_step"], out["range_step"]
        out["recent_range"] = out["relative_range"].rolling(n, min_periods=n).mean()
        out["prior_range"] = out["relative_range"].shift(n).rolling(20, min_periods=20).mean()
        out["compression"] = out["recent_range"] / out["prior_range"].replace(0.0, np.nan)
        out["price_variation"] = x.abs().rolling(n, min_periods=n).sum()
        out["range_variation"] = y.abs().rolling(n, min_periods=n).sum()
        out["net_return"] = x.rolling(n, min_periods=n).sum()
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature of the price/range path.
        for lag in range(1, n):
            cross = x.shift(lag) * y - y.shift(lag) * x
            area = area + 0.5 * cross.rolling(n - lag, min_periods=n - lag).sum()
        out["signed_area"] = area
        denominator = out["price_variation"] * out["range_variation"]
        out["normalized_area"] = area / denominator.replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PathParams,
    ) -> SignalFrame:
        entry = (
            (indicators["compression"] < params.compression_ratio)
            & (indicators["normalized_area"] > 0.05)
            & (indicators["net_return"] > 0.0)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # The desired position returns to cash exactly when entry flips false.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, params = GeneratedStrategy(), PathParams()
    index = pd.date_range("2020-01-01", periods=32)
    close = pd.Series(100.0 + np.sin(np.arange(32)) + 0.1 * np.arange(32), index=index)
    width = pd.Series(2.0 + 0.5 * np.cos(np.arange(32)), index=index)
    data = pd.DataFrame({
        "open": close, "high": close + width, "low": close - width,
        "close": close, "volume": 1000.0,
    }, index=index)
    out = strategy.indicators(data, params)
    x = out["price_step"].iloc[-params.path_bars:].to_numpy()
    y = out["range_step"].iloc[-params.path_bars:].to_numpy()
    expected = 0.5 * sum(
        x[i] * y[j] - y[i] * x[j]
        for i in range(params.path_bars) for j in range(i + 1, params.path_bars)
    )
    assert np.isclose(out["signed_area"].iloc[-1], expected)
    pd.testing.assert_frame_equal(
        out.iloc[:28], strategy.indicators(data.iloc[:28], params)
    )
    gates = pd.DataFrame({
        "compression": [np.nan, 0.5, 0.5, 1.0, 0.5],
        "normalized_area": [np.nan, 0.2, 0.2, 0.2, -0.2],
        "net_return": [np.nan, 0.01, 0.01, 0.01, 0.01],
    }, index=index[:5])
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data.iloc[:5], gates, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert (result["size"] > 0.0).all()
    assert strategy.warmup_bars(params) >= 20 + params.path_bars


if __name__ == "__main__":
    _self_check()
