from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlateauParams:
    peak_lookback: int = 20
    range_lookback: int = 20
    min_depth_ranges: float = 1.5
    max_depth_ranges: float = 4.0
    compression_ratio: float = 1.25
    plateau_tolerance: float = 0.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[PlateauParams]):
    strategy_id = "gen_local-w2_1790995507"

    @classmethod
    def params_type(cls) -> type[PlateauParams]:
        return PlateauParams

    def warmup_bars(self, params: PlateauParams) -> int:
        return int(max(params.peak_lookback + 2, params.range_lookback + 3, 4))

    def indicators(self, data: pd.DataFrame, params: PlateauParams) -> pd.DataFrame:
        if params.peak_lookback < 2 or params.range_lookback < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["close"].rolling(params.peak_lookback, min_periods=params.peak_lookback).max()
        out["daily_range"] = data["high"] - data["low"]
        out["baseline_range"] = out["daily_range"].rolling(params.range_lookback, min_periods=params.range_lookback).mean().shift(2)
        baseline = out["baseline_range"].where(out["baseline_range"] > 0)
        out["drawdown"] = out["peak"] - data["close"]
        out["depth_ranges"] = out["drawdown"] / baseline
        out["depth_change_ranges"] = out["drawdown"].diff().abs() / baseline
        out["two_bar_envelope"] = data["high"].rolling(2, min_periods=2).max() - data["low"].rolling(2, min_periods=2).min()
        out["compression"] = out["two_bar_envelope"] / baseline
        required = out[["depth_ranges", "depth_change_ranges", "compression"]]
        valid = pd.Series(np.isfinite(required.to_numpy()).all(axis=1), index=data.index)
        out["qualifies"] = (
            valid
            & out["depth_ranges"].between(params.min_depth_ranges, params.max_depth_ranges)
            & out["depth_change_ranges"].le(params.plateau_tolerance)
            & out["compression"].le(params.compression_ratio)
        ).fillna(False)
        out["entry_ready"] = out["qualifies"] & out["qualifies"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PlateauParams) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(ready):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Force a flat bar at expiry, even if confirmation persists.
                    exit_bar = -1
            elif enter:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    sample = pd.DataFrame(index=pd.date_range("2020-01-01", periods=8))
    confirmed = pd.DataFrame({"entry_ready": [False, True, True, True, False, False, False, False]}, index=sample.index)
    for holding in (1, 2):
        result = strategy.generate_signals(sample, confirmed, None, PlateauParams(hold_bars=holding)).data
        assert result["signal"].iloc[:2].eq(0).all()
        assert result["signal"].iloc[2:2 + holding].eq(1).all()
        assert result["signal"].iloc[2 + holding] == 0
        assert result["signal"].isin([0, 1]).all()
        assert result["size"].gt(0).all()
