from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityReboundParams:
    graph_window: int = 12
    correlation_window: int = 30
    min_visibility: float = 0.65
    min_negative_correlation: float = 0.10
    min_forecast: float = 0.001
    profit_target: float = 0.008
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[VisibilityReboundParams]):
    strategy_id = "gen_local-w2_1791328245"

    @classmethod
    def params_type(cls) -> type[VisibilityReboundParams]:
        return VisibilityReboundParams

    @staticmethod
    def warmup_bars(params: VisibilityReboundParams) -> int:
        if params.graph_window < 3 or params.correlation_window < 3:
            raise ValueError("Lookbacks must be at least three bars")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")
        if not 0.0 <= params.min_visibility <= 1.0:
            raise ValueError("min_visibility must be between zero and one")
        if not 0.0 <= params.min_negative_correlation < 1.0:
            raise ValueError("Invalid correlation threshold")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive")
        if not np.isfinite(params.min_forecast) or params.min_forecast < 0.0:
            raise ValueError("min_forecast must be finite and nonnegative")
        return int(max(params.graph_window + 1, params.correlation_window + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityReboundParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        close = data["close"].astype(float).where(data["close"] > 0.0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        inverted = -returns
        degree = pd.Series(0.0, index=data.index)
        obstruction = pd.Series(-np.inf, index=data.index)
        # A past node is visible when its connecting slope exceeds every intervening slope.
        for lag in range(1, params.graph_window):
            slope = (inverted.shift(lag) - inverted) / lag
            degree += (slope.notna() & inverted.notna() & (slope > obstruction)).astype(float)
            obstruction = pd.Series(np.fmax(obstruction.to_numpy(), slope.to_numpy()), index=data.index)
        complete = returns.rolling(params.graph_window, min_periods=params.graph_window).count().eq(params.graph_window)
        visibility = (degree / (params.graph_window - 1)).where(complete)
        lagged_returns = returns.shift(1)
        correlation = returns.rolling(params.correlation_window, min_periods=params.correlation_window).corr(lagged_returns)
        correlation = correlation.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        mean_return = returns.rolling(params.correlation_window, min_periods=params.correlation_window).mean()
        forecast = mean_return + correlation * (returns - mean_return)
        topology_agrees = (returns < 0.0) & (visibility >= params.min_visibility)
        autocorrelation_agrees = (correlation < -params.min_negative_correlation) & (forecast > params.min_forecast)
        entry = (topology_agrees & autocorrelation_agrees).fillna(False)
        return pd.DataFrame({
            "return": returns,
            "inverted_return": inverted,
            "visibility_degree": degree.where(complete),
            "visibility": visibility,
            "lagged_return": lagged_returns,
            "autocorrelation": correlation,
            "mean_return": mean_return,
            "forecast": forecast,
            "topology_agrees": topology_agrees.fillna(False),
            "autocorrelation_agrees": autocorrelation_agrees.fillna(False),
            "entry": entry,
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityReboundParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = 0.0
        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if entry_bar < 0 and i > 0 and decisions[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    entry_bar = i
                    entry_price = opens[i]
                else:
                    continue
            if entry_bar >= 0:
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_stop = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_stop:
                    # Close-based exits fill at the next open, like all decisions.
                    entry_bar = -1
                    entry_price = 0.0
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}, index=index)
    forced = pd.DataFrame({"entry": [True, False, False, False, False]}, index=index)
    params = VisibilityReboundParams()
    result = GeneratedStrategy.generate_signals(bars, forced, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    bars.loc[index[1], "close"] = 101.0
    result = GeneratedStrategy.generate_signals(bars, forced, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    derived = GeneratedStrategy.indicators(bars, params)
    assert not derived["entry"].any()
    assert result["size"].gt(0.0).all()
