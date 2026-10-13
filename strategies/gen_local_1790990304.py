from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    drawdown_window: int = 20
    range_window: int = 14
    atr_window: int = 14
    drawdown_atr: float = 2.0
    shock_multiple: float = 1.4
    minimum_overlap: float = 0.5
    stop_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790990304"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.drawdown_window, params.range_window + 1, params.atr_window + 1) + 2)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["range"] = data["high"] - data["low"]
        out["previous_range"] = out["range"].shift(1)
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["range_baseline"] = out["range"].rolling(params.range_window, min_periods=params.range_window).mean().shift(1)
        out["recent_peak"] = data["high"].rolling(params.drawdown_window, min_periods=params.drawdown_window).max()
        out["drawdown_units"] = (out["recent_peak"] - data["close"]) / out["atr"].where(out["atr"] > 0)
        out["close_location"] = (data["close"] - data["low"]) / out["range"].where(out["range"] > 0)
        out["intersection_high"] = np.minimum(data["high"], out["previous_high"])
        out["intersection_low"] = np.maximum(data["low"], out["previous_low"])
        out["intersection"] = (out["intersection_high"] - out["intersection_low"]).clip(lower=0)
        out["smaller_range"] = np.minimum(out["range"], out["previous_range"])
        out["overlap"] = out["intersection"] / out["smaller_range"].where(out["smaller_range"] > 0)
        out["shock"] = (
            (out["drawdown_units"] >= params.drawdown_atr)
            & (out["range"] >= params.shock_multiple * out["range_baseline"])
            & (out["range_baseline"] > 0)
            & (data["low"] < out["previous_low"])
            & (data["close"] < out["previous_close"])
            & (out["close_location"] <= 0.35)
        ).fillna(False)
        out["range_translation"] = (
            (data["low"] > out["previous_low"])
            & (data["high"] > out["previous_high"])
            & (out["overlap"] >= params.minimum_overlap)
        ).fillna(False)
        out["entry"] = (
            out["shock"].shift(2, fill_value=False)
            & out["range_translation"].shift(1, fill_value=False)
            & out["range_translation"]
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=bool)
        active = False
        pending = False
        frozen_atr = np.nan
        stop_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_bar = i
                    stop_price = opens[i] - params.stop_atr * frozen_atr
            if active:
                stopped = np.isfinite(closes[i]) and closes[i] < stop_price
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                frozen_atr = atr[i]
                pending = True
                decisions[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
