from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPressureParams:
    baseline_days: int = 60
    pressure_days: int = 4
    entry_z: float = 1.0
    reversal_z: float = 0.25


class GeneratedStrategy(BaseStrategy[GapPressureParams]):
    strategy_id = "gen_local-w2_1790965281"

    @classmethod
    def params_type(cls):
        return GapPressureParams

    def warmup_bars(self, params):
        return max(200, params.baseline_days + params.pressure_days + 1)

    def indicators(self, data, params):
        if params.baseline_days < 2 or params.pressure_days < 1:
            raise ValueError("Require baseline_days >= 2 and pressure_days >= 1")
        if not np.isfinite(params.entry_z) or not np.isfinite(params.reversal_z) or params.entry_z <= 0 or params.reversal_z < 0:
            raise ValueError("Require finite entry_z > 0 and reversal_z >= 0")
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1)
        out["gap"] = (data["open"] / prior_close.where(prior_close > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out["gap_mean"] = out["gap"].rolling(params.baseline_days, min_periods=params.baseline_days).mean().shift(1)
        out["gap_std"] = out["gap"].rolling(params.baseline_days, min_periods=params.baseline_days).std().shift(1)
        out["gap_surprise"] = (out["gap"] - out["gap_mean"]) / out["gap_std"].where(out["gap_std"] > 1e-12)
        out["pressure"] = out["gap_surprise"].rolling(params.pressure_days, min_periods=params.pressure_days).sum() / np.sqrt(params.pressure_days)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        # Hysteresis: negative pressure arms entry; positive pressure reverses it.
        # The 3-5-session pressure window targets the horizon; exits require reversal.
        out["entry"] = ((out["pressure"] <= -params.entry_z) & (data["close"] > out["ma200"])).fillna(False)
        out["reversal"] = (out["pressure"] >= params.reversal_z).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        positions = np.zeros(len(data), dtype=int)
        holding = False
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        reversals = indicators["reversal"].fillna(False).to_numpy(dtype=bool)
        for i, (entry, reversal) in enumerate(zip(entries, reversals)):
            if holding:
                if reversal:
                    holding = False
            elif entry:
                holding = True
            positions[i] = int(holding)
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapPressureParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame({
        "entry": [False, True, False, False, True, False],
        "reversal": [False, False, False, True, False, False],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 1]
    assert (result["size"] > 0).all()
