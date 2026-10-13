from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    gap_window: int = 8
    baseline_window: int = 32
    energy_window: int = 5
    entry_threshold: float = 0.8
    coarse_threshold: float = 0.5
    min_size: float = 0.2
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790983809"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.baseline_window + 1, params.gap_window + 1,
                       8 + params.energy_window))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (params.gap_window < 1 or params.baseline_window < 2
                or params.energy_window < 1 or params.entry_threshold <= 0
                or not 0 < params.coarse_threshold < 1
                or not 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).where(lambda x: x > 0)
        opening = data["open"].where(lambda x: x > 0)
        closing = data["close"].where(lambda x: x > 0)
        out["gap"] = np.log(opening / previous)
        out["intraday"] = np.log(closing / opening)
        for scale in (1, 2, 4):
            out[f"mean_{scale}"] = out["gap"].rolling(scale).mean()
            out[f"haar_{scale}"] = np.sqrt(scale / 2.0) * (
                out[f"mean_{scale}"] - out[f"mean_{scale}"].shift(scale))
            out[f"energy_{scale}"] = (
                out[f"haar_{scale}"].pow(2).rolling(params.energy_window).mean())
        out["total_energy"] = out[["energy_1", "energy_2", "energy_4"]].sum(
            axis=1, min_count=3)
        out["coarse_share"] = out["energy_4"] / out["total_energy"].where(
            out["total_energy"] > 0)
        out["gap_sum"] = out["gap"].rolling(params.gap_window).sum()
        out["gap_volatility"] = out["gap"].rolling(params.baseline_window).std(ddof=0)
        out["tension"] = out["gap_sum"] / (
            out["gap_volatility"].where(out["gap_volatility"] > 0)
            * np.sqrt(params.gap_window))
        out["counterforce"] = out["intraday"].rolling(params.gap_window).sum()
        out["strength"] = out["tension"].abs() * out["coarse_share"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        tension = indicators["tension"]
        eligible = (
            tension.abs().ge(params.entry_threshold)
            & indicators["coarse_share"].ge(params.coarse_threshold)
            & (tension * indicators["counterforce"]).lt(0)
        ).fillna(False)
        desired = pd.Series(0, index=data.index, dtype=int)
        desired.loc[eligible & tension.lt(0)] = 1
        desired.loc[eligible & tension.gt(0)] = -1
        strength = indicators["strength"].replace(
            [np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        entry_size = params.min_size + (params.max_size - params.min_size) * (
            1 - np.exp(-strength))
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position, held_size = 0, params.min_size
        for i, direction in enumerate(desired.to_numpy()):
            if position and direction == -position:
                position = 0
            elif position == 0 and direction:
                position = int(direction)
                held_size = float(entry_size.iloc[i])
            signals[i] = position
            sizes[i] = held_size if position else params.min_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), StrategyParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "tension": [-2.0, 0.0, 2.0, 2.0, 0.0],
        "coarse_share": 1.0,
        "counterforce": [1.0, 0.0, -1.0, -1.0, 0.0],
        "strength": [2.0, 0.0, 2.0, 2.0, 0.0],
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1]
    assert result["size"].between(params.min_size, params.max_size).all()
    assert result["size"].iloc[1] > params.min_size
    quiet = strategy.generate_signals(
        data, strategy.indicators(data, params), ctx, params).data
    assert quiet["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
