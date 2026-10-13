from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAllocationParams:
    allocation_window: int = 10
    percentile_window: int = 126
    entry_percentile: float = 0.8
    volatility_window: int = 20
    target_volatility: float = 0.12
    holding_bars: int = 8
    max_size: float = 0.95

    def __post_init__(self):
        windows = (self.allocation_window, self.percentile_window, self.volatility_window)
        if any(type(value) is not int or value < 2 for value in windows):
            raise ValueError("Lookback windows must be integers >= 2")
        if type(self.holding_bars) is not int or self.holding_bars < 1:
            raise ValueError("holding_bars must be a positive integer")
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one")
        if not np.isfinite(self.target_volatility) or self.target_volatility <= 0:
            raise ValueError("target_volatility must be finite and positive")
        if not 0.0 < self.max_size <= 1.0:
            raise ValueError("max_size must be between zero and one")


class GeneratedStrategy(BaseStrategy[RangeAllocationParams]):
    strategy_id = "gen_local_1791291067"

    @classmethod
    def params_type(cls):
        return RangeAllocationParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.volatility_window, params.allocation_window + params.percentile_window + 2))

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        valid = (data["low"] > 0) & (data["high"] >= data["low"])
        out["range_energy"] = np.log(data["high"].where(valid) / data["low"].where(valid)).pow(2).replace([np.inf, -np.inf], np.nan)
        out["direction"] = np.sign(data["close"].diff())
        out["signed_energy"] = out["range_energy"] * out["direction"]
        out["energy_sum"] = out["range_energy"].rolling(params.allocation_window).sum()
        out["signed_sum"] = out["signed_energy"].rolling(params.allocation_window).sum()
        out["allocation"] = out["signed_sum"] / out["energy_sum"].replace(0.0, np.nan)
        out["threshold"] = out["allocation"].shift(1).rolling(params.percentile_window).quantile(params.entry_percentile)
        out["drift"] = data["close"].diff(params.allocation_window)
        out["annual_volatility"] = np.sqrt(out["range_energy"].rolling(params.volatility_window).mean() * 252.0 / (4.0 * np.log(2.0)))
        out["risk_size"] = (params.target_volatility / out["annual_volatility"].replace(0.0, np.nan)).clip(upper=params.max_size)
        out["entry"] = (
            (out["allocation"] > out["threshold"])
            & (out["allocation"].shift(1) <= out["threshold"].shift(1))
            & (out["drift"] > 0)
            & out["risk_size"].notna()
            & (out["risk_size"] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame({"signal": 0, "size": float(params.max_size)}, index=data.index)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        sizes = indicators["risk_size"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        weights = np.full(len(data), params.max_size, dtype=float)
        remaining = 0
        entry_size = float(params.max_size)
        for i in range(len(data)):
            if remaining > 0:
                remaining -= 1
                if remaining == 0:
                    continue
            elif entries[i] and np.isfinite(sizes[i]) and sizes[i] > 0:
                remaining = params.holding_bars
                entry_size = float(min(sizes[i], params.max_size))
            if remaining > 0:
                signals[i] = 1
                weights[i] = entry_size
        df["signal"] = signals
        df["size"] = weights
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RangeAllocationParams()
    index = pd.date_range("2020-01-01", periods=2 * (params.holding_bars + 1) + 2)
    close = pd.Series(100.0 + np.arange(len(index)), index=index)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0})
    probe = pd.DataFrame({"entry": True, "risk_size": np.linspace(0.2, 0.8, len(index))}, index=index)
    result = GeneratedStrategy.generate_signals(data, probe, None, params).data
    expected = [int(i > 0 and (i - 1) % (params.holding_bars + 1) < params.holding_bars) for i in range(len(index))]
    assert result["signal"].tolist() == expected
    assert (result["size"] > 0).all()
    assert (result["size"].iloc[1:params.holding_bars + 1] == 0.2).all()
    actual = GeneratedStrategy.indicators(data, params)
    assert not actual["entry"].any()
