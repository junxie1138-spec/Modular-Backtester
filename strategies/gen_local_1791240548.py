from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class LoopParams:
    roc_bars: int = 3
    loop_bars: int = 6
    baseline_bars: int = 40
    acceleration_threshold: float = 1.2
    closure_limit: float = 0.25
    displacement_threshold: float = 0.5
    holding_bars: int = 8
    min_size: float = 0.2
    max_size: float = 0.95

    def __post_init__(self):
        if min(self.roc_bars, self.loop_bars, self.holding_bars) < 1 or self.baseline_bars < 2:
            raise ValueError("Lookbacks and holding period must be positive; baseline must be at least two.")
        if not (self.acceleration_threshold > 0 and 0 < self.closure_limit < 1):
            raise ValueError("Invalid acceleration or closure threshold.")
        if not (self.displacement_threshold > 0 and 0 < self.min_size <= self.max_size <= 1):
            raise ValueError("Invalid displacement threshold or size bounds.")


class GeneratedStrategy(BaseStrategy[LoopParams]):
    strategy_id = "gen_local_1791240548"

    @classmethod
    def params_type(cls):
        return LoopParams

    def warmup_bars(self, params: LoopParams) -> int:
        return int(params.roc_bars + 2 + max(params.baseline_bars, params.loop_bars))

    def indicators(self, data: pd.DataFrame, params: LoopParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(periods=p.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["acceleration_energy"] = out["acceleration"].pow(2)
        out["acceleration_scale"] = np.sqrt(
            out["acceleration_energy"].rolling(p.baseline_bars, min_periods=p.baseline_bars).mean()
        ).shift(1).replace(0, np.nan)
        out["acceleration_z"] = out["acceleration"] / out["acceleration_scale"]
        out["absolute_acceleration"] = out["acceleration"].abs()
        out["gross_impulse"] = out["absolute_acceleration"].rolling(p.loop_bars, min_periods=p.loop_bars).sum()
        out["net_impulse"] = out["acceleration"].rolling(p.loop_bars, min_periods=p.loop_bars).sum()
        out["closure"] = out["net_impulse"].abs() / out["gross_impulse"].replace(0, np.nan)
        out["volatility"] = out["return"].rolling(p.baseline_bars, min_periods=p.baseline_bars).std(ddof=0).shift(1).replace(0, np.nan)
        out["displacement"] = out["roc"] / (out["volatility"] * np.sqrt(p.roc_bars))
        out = out.replace([np.inf, -np.inf], np.nan)
        closed = out["closure"].le(p.closure_limit)
        long_entry = closed & out["acceleration_z"].ge(p.acceleration_threshold) & out["displacement"].le(-p.displacement_threshold)
        short_entry = closed & out["acceleration_z"].le(-p.acceleration_threshold) & out["displacement"].ge(p.displacement_threshold)
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        out["conviction"] = (
            (out["acceleration_z"].abs() / (2 * p.acceleration_threshold)).clip(0, 1)
            * (1 - out["closure"]).clip(0, 1)
        ).fillna(0)
        out["entry_size"] = p.min_size + (p.max_size - p.min_size) * out["conviction"]
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: LoopParams,
    ) -> SignalFrame:
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].fillna(params.min_size).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        direction = 0
        exit_bar = -1
        frozen_size = params.min_size
        for i in range(len(data)):
            if direction != 0 and i == exit_bar:
                direction = 0
                continue
            if direction == 0 and candidates[i] != 0:
                direction = int(candidates[i])
                frozen_size = float(entry_sizes[i])
                exit_bar = i + params.holding_bars
            if direction != 0:
                signals[i] = direction
                sizes[i] = frozen_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        # Shift size with the decision so entry conviction is frozen at the fill.
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = LoopParams()
    index = pd.date_range("2020-01-01", periods=20)
    data = pd.DataFrame(index=index)
    candidates = np.ones(20, dtype=int)
    candidates[9] = -1
    indicators = pd.DataFrame({
        "candidate": candidates,
        "entry_size": np.linspace(params.min_size, params.max_size, 20),
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] + [-1] * 8 + [0, 1]
    assert np.allclose(result["size"].iloc[1:9], indicators["entry_size"].iloc[0])
    assert np.allclose(result["size"].iloc[10:18], indicators["entry_size"].iloc[9])
    flat_data = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=100))
    flat = strategy.indicators(flat_data, params)
    assert flat["candidate"].eq(0).all()
    assert np.isfinite(flat["entry_size"]).all()
    assert strategy.warmup_bars(params) >= params.roc_bars + params.baseline_bars + 2


if __name__ == "__main__":
    _self_check()
