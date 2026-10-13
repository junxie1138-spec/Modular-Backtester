from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 6
    volatility_window: int = 20
    innovation_threshold: float = 0.25
    holding_bars: int = 8
    target_volatility: float = 0.12
    max_size: float = 0.95

    def __post_init__(self):
        if self.slope_window < 2 or self.volatility_window < 2 or self.holding_bars < 1:
            raise ValueError("Windows must be >= 2 and holding_bars >= 1.")
        if not np.isfinite(self.innovation_threshold) or self.innovation_threshold <= 0:
            raise ValueError("innovation_threshold must be finite and positive.")
        if not np.isfinite(self.target_volatility) or self.target_volatility <= 0:
            raise ValueError("target_volatility must be finite and positive.")
        if not np.isfinite(self.max_size) or not 0 < self.max_size <= 1:
            raise ValueError("max_size must be in (0, 1].")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791233608"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(params.slope_window, params.volatility_window) + 3)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        midpoint = ((data["high"] + data["low"]) / 2).where(
            (data["high"] > 0) & (data["low"] > 0)
        )
        out["close_return"] = np.log(close).diff()
        out["midpoint_return"] = np.log(midpoint).diff()
        out["overnight_return"] = np.log(opening / close.shift(1))
        out["latent_slope"] = out["midpoint_return"].rolling(
            params.slope_window, min_periods=params.slope_window
        ).mean()
        out["daily_volatility"] = out["close_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out["slope_innovation"] = out["midpoint_return"] - out["latent_slope"].shift(1)
        out["innovation_score"] = out["slope_innovation"] / out[
            "daily_volatility"
        ].shift(1).where(out["daily_volatility"].shift(1) > 0)
        out["annual_volatility"] = out["daily_volatility"] * np.sqrt(252.0)
        out["target_size"] = (
            params.target_volatility / out["annual_volatility"].where(
                out["annual_volatility"] > 0
            )
        ).clip(lower=0.01, upper=params.max_size)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        eligible = (
            (indicators["latent_slope"] > 0)
            & (indicators["innovation_score"] > params.innovation_threshold)
            & (indicators["overnight_return"] <= 0)
            & indicators["target_size"].notna()
        ).fillna(False)
        confirmed = (eligible & eligible.shift(1, fill_value=False)).to_numpy()
        candidate_sizes = indicators["target_size"].to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.max_size, dtype=float)
        exit_bar = -1
        entry_size = params.max_size
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signals[i] = 1
                sizes[i] = entry_size
            elif confirmed[i]:
                exit_bar = i + params.holding_bars
                entry_size = float(candidate_sizes[i])
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params(holding_bars=8)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=18, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    actual = strategy.indicators(data, params)
    assert actual.index.equals(index)
    flat = strategy.generate_signals(data, actual, None, params).data
    assert (flat["signal"] == 0).all() and (flat["size"] > 0).all()
    forced = pd.DataFrame(
        {"latent_slope": 0.01, "innovation_score": 1.0,
         "overnight_return": -0.001, "target_size": 0.4},
        index=index,
    )
    forced.iloc[0, forced.columns.get_loc("innovation_score")] = 0.0
    forced.iloc[3:, forced.columns.get_loc("innovation_score")] = -1.0
    result = strategy.generate_signals(data, forced, None, params).data
    expected = np.zeros(len(index), dtype=int)
    expected[3:11] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result.loc[result["signal"] == 1, "size"] == 0.4).all()
