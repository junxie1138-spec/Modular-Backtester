from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    fine_share_max: float = 0.25
    position_max: float = 0.65
    profit_target: float = 0.03
    max_hold_bars: int = 10
    size: float = 0.95

    def __post_init__(self):
        if not 0.0 < self.fine_share_max < 1.0:
            raise ValueError("fine_share_max must be between zero and one")
        if not 0.20 < self.position_max < 1.0:
            raise ValueError("position_max must exceed 0.20 and be below one")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if not isinstance(self.max_hold_bars, int) or self.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be a positive integer")
        if not 0.0 < self.size <= 1.0:
            raise ValueError("size must be between zero and one")


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791288216"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params):
        return 21

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for half in (1, 2, 4, 8):
            mean = out["log_close"].rolling(half, min_periods=half).mean()
            # Causal Haar detail: adjacent trailing blocks, unit-norm weights.
            out[f"wavelet_{2 * half}"] = np.sqrt(half / 2.0) * (mean - mean.shift(half))
            out[f"energy_{2 * half}"] = out[f"wavelet_{2 * half}"].pow(2).rolling(5, min_periods=5).mean()
        out["total_energy"] = out[["energy_2", "energy_4", "energy_8", "energy_16"]].sum(axis=1, min_count=4)
        out["fine_share"] = (out["energy_2"] + out["energy_4"]) / out["total_energy"].replace(0.0, np.nan)
        out["channel_low"] = data["low"].rolling(20, min_periods=20).min()
        out["channel_high"] = data["high"].rolling(20, min_periods=20).max()
        out["relative_position"] = (data["close"] - out["channel_low"]) / (out["channel_high"] - out["channel_low"]).replace(0.0, np.nan)
        out["setup"] = (
            (out["fine_share"] < params.fine_share_max)
            & (out["wavelet_16"] < 0)
            & (out["wavelet_4"] > 0)
            & out["relative_position"].between(0.20, params.position_max)
        ).fillna(False)
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame({"signal": 0, "size": float(params.size)}, index=data.index)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        if not (np.isfinite(opens).all() and np.isfinite(closes).all() and (opens > 0).all() and (closes > 0).all()):
            raise ValueError("Execution prices must be finite and positive")
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        holding = False
        entry_bar = -1
        entry_price = 0.0
        for i in range(len(data)):
            if holding:
                if i == entry_bar:
                    entry_price = opens[i]
                # Decisions at this close execute at the following open.
                if closes[i] >= entry_price * (1.0 + params.profit_target) or i - entry_bar + 1 >= params.max_hold_bars:
                    holding = False
                else:
                    signals[i] = 1
            elif confirmed[i]:
                holding = True
                entry_bar = i + 1
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 99.0, "close": [100, 100, 103, 100, 100, 100], "volume": 1000.0}, index=index)
    evidence = pd.DataFrame({"confirmed": [True, True, True, False, False, False]}, index=index)
    result = strategy.generate_signals(data, evidence, None, WaveletParams(profit_target=0.02)).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, evidence, None, WaveletParams(max_hold_bars=2)).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    indicators = strategy.indicators(data, WaveletParams())
    assert not indicators["confirmed"].any()
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
