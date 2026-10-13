from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 20
    percentile_window: int = 126
    entry_percentile: float = 0.80
    volatility_window: int = 20
    target_volatility: float = 0.10
    atr_window: int = 14
    stop_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791277529"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(max(params.graph_window + params.percentile_window,
                       params.volatility_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if (params.graph_window < 3 or params.percentile_window < 2
                or params.volatility_window < 2 or params.atr_window < 1
                or params.holding_bars < 1 or not 0 < params.entry_percentile < 1
                or params.stop_atr <= 0 or params.target_volatility <= 0):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0))
        minimum_slope = pd.Series(np.inf, index=data.index)
        weighted_edges = pd.Series(0.0, index=data.index)
        # Each edge passes above every intermediate log-price node.
        for lag in range(1, params.graph_window):
            slope = (log_close - log_close.shift(lag)) / lag
            visible = slope.notna() & slope.lt(minimum_slope)
            weighted_edges += lag * visible.astype(float)
            minimum_slope = np.minimum(minimum_slope, slope)
        complete = log_close.rolling(params.graph_window).count().eq(params.graph_window)
        reach = (weighted_edges / (params.graph_window * (params.graph_window - 1) / 2)).where(complete)
        threshold = reach.shift(1).rolling(params.percentile_window).quantile(params.entry_percentile)
        returns = close.pct_change(fill_method=None)
        volatility = returns.rolling(params.volatility_window).std(ddof=0) * np.sqrt(252.0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1).where(previous_close.notna())
        atr = true_range.rolling(params.atr_window).mean()
        trend = log_close - log_close.shift(params.graph_window - 1)
        return pd.DataFrame({
            "reach": reach,
            "reach_threshold": threshold,
            "trend": trend,
            "annual_volatility": volatility,
            "atr": atr
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        candidates = (ready & indicators["reach"].gt(indicators["reach_threshold"])
                      & indicators["trend"].gt(0) & indicators["atr"].gt(0)
                      & indicators["annual_volatility"].gt(0)).to_numpy()
        sizes = (params.target_volatility / indicators["annual_volatility"].where(
            indicators["annual_volatility"] > 0)).replace(
            [np.inf, -np.inf], np.nan).fillna(0.95).clip(lower=0.000001, upper=0.95)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        entry_bar = -1
        entry_atr = np.nan
        fixed_stop = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                active = bool(np.isfinite(opens[i]) and opens[i] > 0)
                if active:
                    entry_bar = i
                    # Anchor to the next bar's opening fill; freeze the ATR
                    # from the preceding decision bar. Never move this stop.
                    fixed_stop = opens[i] - params.stop_atr * entry_atr
            if active:
                if (not np.isfinite(closes[i]) or closes[i] < fixed_stop
                        or i - entry_bar + 1 >= params.holding_bars):
                    active = False
                else:
                    signals[i] = 1
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                signals[i] = 1
                entry_atr = atrs[i]
                pending = True
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = VisibilityParams(stop_atr=2.0)
    index = pd.date_range("2020-01-01", periods=5)
    close = np.array([100.0, 120.0, 99.0, 97.0, 100.0])
    data = pd.DataFrame({"open": 100.0, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({
        "reach": [1.0, 0.0, 0.0, 0.0, 0.0],
        "reach_threshold": 0.5, "trend": 1.0,
        "annual_volatility": 0.20, "atr": 1.0
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    params.holding_bars = 2
    indicators["atr"] = 100.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
