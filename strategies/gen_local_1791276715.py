from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_window: int = 20
    reference_window: int = 30
    entry_z: float = 1.2
    minimum_mean_depth: float = 0.008
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1791276715"

    @classmethod
    def params_type(cls):
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        if params.peak_window < 2 or params.reference_window < 2:
            raise ValueError("Lookback windows must be at least two bars.")
        if params.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be one or two.")
        if not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("entry_z must be positive and finite.")
        if not np.isfinite(params.minimum_mean_depth) or params.minimum_mean_depth <= 0:
            raise ValueError("minimum_mean_depth must be positive and finite.")
        # The reference window operates on an already-windowed depth series.
        return int(params.peak_window + params.reference_window - 1)

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        high = data["high"].astype(float).where(data["high"] > 0)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["peak"] = high.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = (1.0 - close / out["peak"]).clip(lower=0.0)
        out["mean_depth"] = out["depth"].rolling(params.reference_window, min_periods=params.reference_window).mean()
        out["depth_std"] = out["depth"].rolling(params.reference_window, min_periods=params.reference_window).std(ddof=0)
        denominator = out["depth_std"].where(out["depth_std"] > 1e-12)
        out["depth_z"] = ((out["depth"] - out["mean_depth"]) / denominator).replace([np.inf, -np.inf], np.nan)
        out["strength"] = (out["depth_z"].abs() / params.entry_z - 1.0).clip(lower=0.0, upper=1.0).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        z = indicators["depth_z"]
        ready = z.notna() & indicators["mean_depth"].ge(params.minimum_mean_depth)
        candidates = np.select(
            [ready & z.ge(params.entry_z), ready & z.le(-params.entry_z)],
            [1, -1],
            default=0,
        ).astype(int)
        entry_sizes = (0.25 + 0.70 * indicators["strength"].fillna(0.0).clip(0.0, 1.0)).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        active = 0
        active_size = 1.0
        exit_bar = -1
        for i in range(len(data)):
            if active != 0:
                if i == exit_bar:
                    active = 0
                    active_size = 1.0
            elif candidates[i] != 0:
                active = int(candidates[i])
                active_size = float(entry_sizes[i])
                exit_bar = i + params.holding_bars
            signals[i] = active
            sizes[i] = active_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = DrawdownParams(entry_z=1.0)
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame(index=index)
    z = pd.Series([1.5, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0], index=index)
    indicators = pd.DataFrame({
        "depth_z": z,
        "mean_depth": 0.02,
        "strength": (z.abs() - 1.0).clip(0.0, 1.0),
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.allclose(result["size"].to_numpy(), [1.0, 0.60, 0.60, 1.0, 0.95, 0.95, 1.0])
    params.holding_bars = 1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, -1, 0, 1, 0]
    prices = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=70))
    derived = strategy.indicators(prices, params)
    flat = strategy.generate_signals(prices, derived, None, params).data
    assert flat["signal"].eq(0).all()
    assert flat["size"].gt(0).all()
    assert strategy.warmup_bars(params) == 49


if __name__ == "__main__":
    _self_check()
