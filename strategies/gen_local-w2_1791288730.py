from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    roc_bars: int = 5
    volatility_bars: int = 20
    acceleration_bars: int = 2
    acceleration_threshold: float = 0.15
    target_volatility: float = 0.12
    minimum_entry_size: float = 0.35
    maximum_size: float = 0.95
    profit_target: float = 0.025
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791288730"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.roc_bars, params.volatility_bars) + params.acceleration_bars + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.roc_bars, params.acceleration_bars, params.holding_bars) < 1 or params.volatility_bars < 2:
            raise ValueError("Lookbacks and holding period must be positive; volatility requires two bars.")
        if not (0 < params.minimum_entry_size <= params.maximum_size <= 1):
            raise ValueError("Entry and maximum sizes must satisfy 0 < minimum <= maximum <= 1.")
        if params.target_volatility <= 0 or params.profit_target <= 0 or params.acceleration_threshold <= 0:
            raise ValueError("Volatility target, profit target, and acceleration threshold must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["daily_volatility"] = out["return"].rolling(params.volatility_bars, min_periods=params.volatility_bars).std(ddof=0)
        out["annual_volatility"] = out["daily_volatility"] * np.sqrt(252.0)
        out["roc"] = close.pct_change(periods=params.roc_bars, fill_method=None)
        denominator = out["daily_volatility"].where(out["daily_volatility"] > 0) * np.sqrt(float(params.roc_bars))
        out["risk_momentum"] = (out["roc"] / denominator).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["risk_momentum"].diff(params.acceleration_bars) / float(params.acceleration_bars)
        out["raw_exposure"] = (params.target_volatility / out["annual_volatility"].where(out["annual_volatility"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["exposure"] = out["raw_exposure"].clip(lower=0.01, upper=params.maximum_size).fillna(0.01)
        capacity = out["raw_exposure"].ge(params.minimum_entry_size)
        long_entry = capacity & out["risk_momentum"].gt(0) & out["acceleration"].gt(params.acceleration_threshold)
        short_entry = capacity & out["risk_momentum"].lt(0) & out["acceleration"].lt(-params.acceleration_threshold)
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        out["confirmed"] = out["candidate"].where(out["candidate"].eq(out["candidate"].shift(1)), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        position = 0
        pending_entry = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if pending_entry:
                entry_price = opens[i]
                entry_bar = i
                pending_entry = False
            if position:
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and valid_close else np.nan
                expired = i - entry_bar + 1 >= params.holding_bars
                if not valid_entry or gain >= params.profit_target or expired:
                    position = 0
                    decisions[i] = 0
                    continue
            elif confirmed[i] != 0:
                position = int(confirmed[i])
                pending_entry = True
            decisions[i] = position
        df["signal"] = decisions
        df["size"] = indicators["exposure"].shift(1).fillna(0.01).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"confirmed": [0] + [1] * 11, "exposure": 0.8}, index=index)
    result = strategy.generate_signals(bars, derived, None, params).data
    assert result["signal"].iloc[:2].eq(0).all()
    assert result["signal"].iloc[2:10].eq(1).all()
    assert result["signal"].iloc[10] == 0
    bars.loc[index[3], "close"] = 103.0
    result = strategy.generate_signals(bars, derived, None, params).data
    assert result["signal"].iloc[3] == 1 and result["signal"].iloc[4] == 0
    derived["confirmed"] = [0] + [-1] * 11
    bars.loc[index[3], "close"] = 97.0
    result = strategy.generate_signals(bars, derived, None, params).data
    assert result["signal"].iloc[3] == -1 and result["signal"].iloc[4] == 0
