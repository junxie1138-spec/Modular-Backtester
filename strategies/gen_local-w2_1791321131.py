from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    impulse_threshold: float = 0.75
    recovery_fraction: float = 0.35
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local-w2_1791321131"

    @classmethod
    def params_type(cls) -> type[WaveParams]:
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        # Two-bar ROC, two-bar difference, three-bar sum, one-bar lag.
        return 8

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        if not np.isfinite(params.impulse_threshold) or params.impulse_threshold <= 0:
            raise ValueError("impulse_threshold must be finite and positive")
        if not np.isfinite(params.recovery_fraction) or not 0 < params.recovery_fraction < 1:
            raise ValueError("recovery_fraction must lie between zero and one")
        close = data["close"].astype(float).where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        out["return_1"] = close.pct_change(fill_method=None)
        out["roc_2"] = close.pct_change(2, fill_method=None)
        out["acceleration"] = out["roc_2"].diff(2)
        out["prior_impulse"] = out["acceleration"].rolling(3, min_periods=3).sum().shift(1)
        out["return_scale"] = np.sqrt(out["return_1"].pow(2).rolling(6, min_periods=6).mean())
        out["impulse_strength"] = out["prior_impulse"].abs().div(out["return_scale"].replace(0.0, np.nan))
        out["cancellation_ratio"] = -out["acceleration"].div(out["prior_impulse"].replace(0.0, np.nan))
        out["displacement"] = close.pct_change(6, fill_method=None)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        qualified = ready & (out["impulse_strength"] >= params.impulse_threshold)
        qualified &= out["cancellation_ratio"].between(params.recovery_fraction, 1.0, inclusive="left")
        long_entry = qualified & (out["prior_impulse"] < 0) & (out["displacement"] < 0)
        short_entry = qualified & (out["prior_impulse"] > 0) & (out["displacement"] > 0)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveParams) -> SignalFrame:
        if not 15 <= params.hold_bars <= 20:
            raise ValueError("hold_bars must be between 15 and 20 sessions")
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, candidate in enumerate(entries):
            if position != 0:
                if i - entry_bar >= params.hold_bars:
                    position = 0
                    # Preserve an explicit flat exit bar; ignore its candidate.
                signals[i] = position
                continue
            if candidate != 0:
                position = int(candidate)
                entry_bar = i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = WaveParams()
    index = pd.date_range("2020-01-01", periods=45, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    flat = strategy.indicators(data, params)
    assert flat["entry"].eq(0).all()
    entries = pd.DataFrame({"entry": np.zeros(len(data), dtype=int)}, index=index)
    entries.iloc[0, 0] = 1
    entries.iloc[5, 0] = -1
    entries.iloc[18, 0] = -1
    entries.iloc[19, 0] = -1
    entries.iloc[20, 0] = 1
    result = strategy.generate_signals(data, entries, None, params).data
    assert result.index.equals(data.index)
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[20:38].eq(-1).all()
    assert result["signal"].iloc[[0, 19]].eq(0).all()
    assert result["signal"].iloc[38:].eq(0).all()
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) <= 10


if __name__ == "__main__":
    _self_check()
