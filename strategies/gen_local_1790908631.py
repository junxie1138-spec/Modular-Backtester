from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_span: int = 8
    correlation_window: int = 40
    horizon: int = 4
    threshold: float = 0.12


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790908631"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(max(2, params.graph_span) + 2 * max(1, params.horizon) + max(3, params.correlation_window) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        span = max(2, int(params.graph_span))
        horizon = max(1, int(params.horizon))
        window = max(3, int(params.correlation_window))
        returns = np.log(data["close"].where(data["close"] > 0)).diff()
        returns = returns.replace([np.inf, -np.inf], np.nan)
        upper_degree = pd.Series(0.0, index=data.index)
        lower_degree = pd.Series(0.0, index=data.index)
        for lag in range(2, span + 1):
            endpoint = returns.shift(lag)
            upper_visible = returns.notna() & endpoint.notna()
            lower_visible = upper_visible.copy()
            for step in range(1, lag):
                intermediate = returns.shift(step)
                chord = returns + (endpoint - returns) * (step / lag)
                upper_visible &= intermediate < chord
                lower_visible &= intermediate > chord
            upper_degree += upper_visible.astype(float)
            lower_degree += lower_visible.astype(float)
        valid = returns.rolling(span + 1, min_periods=span + 1).count().eq(span + 1)
        upper_degree = upper_degree.where(valid)
        lower_degree = lower_degree.where(valid)
        topology = (upper_degree - lower_degree) / (span - 1)
        cycle = topology.rolling(horizon, min_periods=horizon).mean()
        autocorrelation = cycle.rolling(window, min_periods=window).corr(cycle.shift(horizon))
        autocorrelation = autocorrelation.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        forecast = cycle * autocorrelation
        return pd.DataFrame({
            "log_return": returns,
            "upper_degree": upper_degree,
            "lower_degree": lower_degree,
            "topology": topology,
            "cycle": cycle,
            "autocorrelation": autocorrelation,
            "forecast": forecast,
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        threshold = max(abs(float(params.threshold)), 1e-12)
        forecast = indicators["forecast"].reindex(data.index).to_numpy(dtype=float)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for bar, value in enumerate(forecast):
            if np.isfinite(value):
                if value > threshold:
                    position = 1
                elif value < -threshold:
                    position = 0
            positions[bar] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame({"forecast": [np.nan, 0.3, 0.05, -0.3, -0.05, 0.3, -0.3, 0.0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 1, 0]
    assert result["size"].gt(0).all()
    flat = GeneratedStrategy.indicators(data, params)
    assert flat["forecast"].isna().all()
