from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCapacityParams:
    capacity_bars: int = 4
    percentile_bars: int = 63
    entry_percentile: float = 0.85
    atr_bars: int = 14
    stop_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapCapacityParams]):
    strategy_id = "gen_local_1791294792"

    @classmethod
    def params_type(cls) -> type[GapCapacityParams]:
        return GapCapacityParams

    @staticmethod
    def warmup_bars(params: GapCapacityParams) -> int:
        return max(params.capacity_bars, params.atr_bars) + params.percentile_bars + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapCapacityParams) -> pd.DataFrame:
        if params.capacity_bars < 1 or params.percentile_bars < 2 or params.atr_bars < 1:
            raise ValueError("Lookbacks must be positive; percentile_bars must be at least two")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0 or params.hold_bars not in (1, 2):
            raise ValueError("stop_atr must be positive and finite; hold_bars must be one or two")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["body"] = data["close"] - data["open"]
        out["gap"] = data["open"] - out["prior_close"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["bear_capacity"] = (-out["body"]).clip(lower=0.0).rolling(params.capacity_bars, min_periods=params.capacity_bars).sum().shift(1)
        out["bull_capacity"] = out["body"].clip(lower=0.0).rolling(params.capacity_bars, min_periods=params.capacity_bars).sum().shift(1)
        out["opposing_capacity"] = out["bear_capacity"].where(out["gap"] > 0.0, out["bull_capacity"])
        out["gap_pressure"] = out["gap"].abs() / (out["opposing_capacity"] + out["prior_atr"]).replace(0.0, np.nan)
        out["pressure_threshold"] = out["gap_pressure"].shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.entry_percentile)
        valid = np.isfinite(out[["gap", "body", "atr", "gap_pressure", "pressure_threshold"]]).all(axis=1) & out["atr"].gt(0.0)
        extreme = valid & out["gap_pressure"].gt(out["pressure_threshold"])
        out["entry_direction"] = np.select([
            extreme & out["gap"].gt(0.0) & out["body"].gt(0.0),
            extreme & out["gap"].lt(0.0) & out["body"].lt(0.0),
        ], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapCapacityParams) -> SignalFrame:
        targets = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].to_numpy(dtype=int)
        position = 0
        entry_price = 0.0
        entry_atr = 0.0
        age = 0
        for i in range(len(data)):
            filled_target = targets[i - 1] if i > 0 else 0
            if filled_target == 0:
                position = 0
                age = 0
            elif position == 0:
                position = int(filled_target)
                entry_price = opens[i]
                entry_atr = atr[i - 1]
                age = 0
            if position != 0:
                age += 1
                stopped = position * (closes[i] - entry_price) <= -params.stop_atr * entry_atr
                invalid = not (np.isfinite(entry_price) and np.isfinite(entry_atr) and np.isfinite(closes[i]))
                targets[i] = 0 if stopped or invalid or age >= params.hold_bars else position
            else:
                targets[i] = entries[i]
        df = pd.DataFrame({"signal": targets, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
