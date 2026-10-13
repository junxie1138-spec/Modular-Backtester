from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 16
    topology_threshold: float = 0.25
    min_efficiency: float = 0.35
    profit_target: float = 0.018
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791330294"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return max(200, params.graph_window + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.graph_window < 2 or not 3 <= params.holding_bars <= 5:
            raise ValueError("graph_window must be >= 2 and holding_bars must be 3..5")
        if not 0 < params.topology_threshold <= 1 or not 0 <= params.min_efficiency <= 1:
            raise ValueError("Invalid topology or efficiency threshold")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        log_price = np.log(close)
        upper_degree = pd.Series(0.0, index=data.index)
        lower_degree = pd.Series(0.0, index=data.index)
        max_slope = pd.Series(-np.inf, index=data.index)
        min_slope = pd.Series(np.inf, index=data.index)
        valid = log_price.notna()
        # Endpoint edges exist when all intervening nodes lie below the chord.
        # Reflected-price edges reverse that inequality.
        for lag in range(1, params.graph_window + 1):
            slope = (log_price.shift(lag) - log_price) / lag
            valid &= slope.notna()
            upper_degree += (slope > max_slope).astype(float)
            lower_degree += (slope < min_slope).astype(float)
            max_slope = pd.Series(np.maximum(max_slope.to_numpy(), slope.to_numpy()), index=data.index)
            min_slope = pd.Series(np.minimum(min_slope.to_numpy(), slope.to_numpy()), index=data.index)
        result = pd.DataFrame(index=data.index)
        result["upper_degree"] = upper_degree.where(valid)
        result["lower_degree"] = lower_degree.where(valid)
        result["topology"] = ((upper_degree - lower_degree) / params.graph_window).where(valid)
        result["ma200"] = close.rolling(200, min_periods=200).mean()
        result["displacement"] = log_price.diff(params.graph_window)
        result["path_length"] = log_price.diff().abs().rolling(
            params.graph_window, min_periods=params.graph_window
        ).sum()
        result["efficiency"] = result["displacement"].abs().div(
            result["path_length"].replace(0.0, np.nan)
        )
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        close = data["close"]
        efficient = indicators["efficiency"] >= params.min_efficiency
        longs = (
            efficient
            & (indicators["topology"] >= params.topology_threshold)
            & (indicators["displacement"] > 0)
            & (close > indicators["ma200"])
        )
        shorts = (
            efficient
            & (indicators["topology"] <= -params.topology_threshold)
            & (indicators["displacement"] < 0)
            & (close < indicators["ma200"])
        )
        candidates = np.select([longs.fillna(False), shorts.fillna(False)], [1, -1], default=0)
        decisions = np.zeros(len(data), dtype=int)
        closes = close.to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        position = 0
        entry_decision = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # Read the entry open only once its bar has arrived.
                if i == entry_decision + 1:
                    entry_price = opens[i]
                invalid = (
                    not np.isfinite(entry_price) or entry_price <= 0
                    or not np.isfinite(closes[i]) or closes[i] <= 0
                )
                age = i - entry_decision
                # Targets are close-confirmed; liquidation fills next bar.
                profit = position * (closes[i] / entry_price - 1.0) if not invalid else 0.0
                if invalid or profit >= params.profit_target or age >= params.holding_bars:
                    position = 0
                    entry_price = np.nan
                decisions[i] = position
            elif np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                if position:
                    entry_decision = i
                decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"efficiency": 1.0, "topology": 1.0, "displacement": 1.0, "ma200": 90.0},
        index=index,
    )
    signals = strategy.generate_signals(data, indicators, None, params).data["signal"].tolist()
    assert signals == [0, 1, 1, 1, 1, 0, 1, 1]
    data.iloc[2, data.columns.get_loc("close")] = 102.0
    signals = strategy.generate_signals(data, indicators, None, params).data["signal"].tolist()
    assert signals[:4] == [0, 1, 1, 0]
    data["close"] = 100.0
    data.iloc[2, data.columns.get_loc("close")] = 98.0
    indicators["topology"] = -1.0
    indicators["displacement"] = -1.0
    indicators["ma200"] = 110.0
    signals = strategy.generate_signals(data, indicators, None, params).data["signal"].tolist()
    assert signals[:4] == [0, -1, -1, 0]
    curve = pd.DataFrame({"close": np.exp(np.linspace(0.0, 1.0, 260) ** 2)})
    upward = strategy.indicators(curve, params)
    downward = strategy.indicators(curve.assign(close=1.0 / curve["close"]), params)
    assert upward["topology"].iloc[-1] > params.topology_threshold
    assert downward["topology"].iloc[-1] < -params.topology_threshold
    flat = strategy.indicators(curve.assign(close=100.0), params)
    assert (flat["topology"].dropna() == 0.0).all()
    assert flat["efficiency"].isna().all()
    assert strategy.warmup_bars(VisibilityParams(graph_window=240)) == 241


if __name__ == "__main__":
    _self_check()
