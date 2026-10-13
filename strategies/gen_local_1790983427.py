from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StaircaseParams:
    peak_window: int = 20
    retrace_min: float = 0.35
    retrace_max: float = 0.75
    min_peak_gain: float = 0.01
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StaircaseParams]):
    strategy_id = "gen_local_1790983427"

    @classmethod
    def params_type(cls) -> type[StaircaseParams]:
        return StaircaseParams

    def warmup_bars(self, params: StaircaseParams) -> int:
        return int(max(220, 3 * params.peak_window + 1))

    def indicators(self, data: pd.DataFrame, params: StaircaseParams) -> pd.DataFrame:
        if params.peak_window < 2:
            raise ValueError("peak_window must be at least 2")
        if not 0.0 < params.retrace_min < params.retrace_max < 1.0:
            raise ValueError("retracement bounds must satisfy 0 < min < max < 1")
        if not np.isfinite(params.min_peak_gain) or params.min_peak_gain <= 0.0:
            raise ValueError("min_peak_gain must be positive and finite")
        w = params.peak_window
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["ma200_change"] = out["ma200"].diff(20)
        out["peak_recent"] = close.rolling(w, min_periods=w).max()
        out["peak_middle"] = close.shift(w).rolling(w, min_periods=w).max()
        out["peak_old"] = close.shift(2 * w).rolling(w, min_periods=w).max()
        out["recent_step"] = out["peak_recent"] - out["peak_middle"]
        out["older_step"] = out["peak_middle"] - out["peak_old"]
        out["step_budget"] = out[["recent_step", "older_step"]].min(axis=1, skipna=False)
        out["drawdown_depth"] = out["peak_recent"] - close
        out["retrace_fraction"] = out["drawdown_depth"] / out["step_budget"].where(out["step_budget"] > 0.0)
        out["recent_gain"] = out["recent_step"] / out["peak_middle"].where(out["peak_middle"] > 0.0)
        out["older_gain"] = out["older_step"] / out["peak_old"].where(out["peak_old"] > 0.0)
        ready = np.isfinite(out).all(axis=1) & close.gt(0.0)
        out["eligible"] = (
            ready
            & close.gt(out["ma200"])
            & out["ma200_change"].gt(0.0)
            & out["recent_gain"].ge(params.min_peak_gain)
            & out["older_gain"].ge(params.min_peak_gain)
            & out["retrace_fraction"].between(params.retrace_min, params.retrace_max)
        ).fillna(False)
        out["entry"] = out["eligible"] & ~out["eligible"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StaircaseParams) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    # Force the scheduled exit, even when a new entry appears.
                    exit_at = -1
            elif entries[i]:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    candidates = pd.DataFrame({"entry": True}, index=index)
    result = strategy.generate_signals(data, candidates, None, StaircaseParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    result = strategy.generate_signals(data, candidates, None, StaircaseParams(hold_bars=1))
    assert result.data["signal"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
