from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownCorrelationParams:
    peak_window: int = 63
    correlation_window: int = 42
    deep_drawdown: float = 0.025
    shallow_drawdown: float = 0.005
    forecast_threshold: float = 0.0005
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[DrawdownCorrelationParams]):
    strategy_id = "gen_local-w2_1791271393"

    @classmethod
    def params_type(cls) -> type[DrawdownCorrelationParams]:
        return DrawdownCorrelationParams

    @staticmethod
    def warmup_bars(params: DrawdownCorrelationParams) -> int:
        return int(max(params.peak_window, params.correlation_window + 3))

    @staticmethod
    def indicators(data: pd.DataFrame, params: DrawdownCorrelationParams) -> pd.DataFrame:
        if params.peak_window < 2 or params.correlation_window < 3:
            raise ValueError("Lookback windows are too short")
        if not 0 <= params.shallow_drawdown < params.deep_drawdown < 1:
            raise ValueError("Drawdown thresholds must be ordered within [0, 1)")
        if not np.isfinite(params.forecast_threshold) or params.forecast_threshold <= 0:
            raise ValueError("forecast_threshold must be finite and positive")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        close = data["close"].astype(float).where(lambda s: np.isfinite(s) & s.gt(0))
        out = pd.DataFrame(index=data.index)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["return"] = close.pct_change(fill_method=None)
        out["lagged_return"] = out["return"].shift(1)
        out["correlation"] = out["return"].rolling(
            params.correlation_window, min_periods=params.correlation_window
        ).corr(out["lagged_return"]).clip(-1.0, 1.0)
        out["fitted_correlation"] = out["correlation"].shift(1)
        out["forecast"] = out["fitted_correlation"] * out["return"]
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownCorrelationParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        depth = indicators["drawdown"]
        forecast = indicators["forecast"]
        valid = np.isfinite(depth) & np.isfinite(forecast)
        long_entry = valid & depth.ge(params.deep_drawdown) & forecast.ge(params.forecast_threshold)
        short_entry = valid & depth.le(params.shallow_drawdown) & forecast.le(-params.forecast_threshold)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(entries):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signals[i] = position
                continue
            if candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "drawdown": [0.04] * 8,
        "forecast": [0.001, -0.001, -0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    }, index=index)
    params = DrawdownCorrelationParams()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    indicators["drawdown"] = 0.001
    indicators["forecast"] = -0.001
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, -1, -1, 0, -1]
    indicators["forecast"] = np.nan
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].eq(0).all()
