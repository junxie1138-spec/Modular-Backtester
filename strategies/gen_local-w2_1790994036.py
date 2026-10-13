from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    drawdown_min: float = 0.04
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1790994036"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return 61

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        volume = data["volume"]
        out["previous_close"] = close.shift(1)
        out["prior_peak"] = close.shift(1).rolling(60, min_periods=60).max()
        out["drawdown"] = 1.0 - close.div(out["prior_peak"].replace(0.0, np.nan))
        out["return"] = close.div(out["previous_close"].replace(0.0, np.nan)) - 1.0
        out["baseline_volume"] = volume.shift(1).rolling(20, min_periods=20).median()
        out["capitulation"] = (
            (out["drawdown"] >= params.drawdown_min)
            & (out["return"] <= -0.015)
            & (close < data["open"])
            & (volume >= 1.8 * out["baseline_volume"])
            & (out["baseline_volume"] > 0.0)
        ).fillna(False)
        out["bar_number"] = np.arange(len(data), dtype=float)
        out["anchor_bar"] = out["bar_number"].where(out["capitulation"]).ffill()
        out["anchor_age"] = out["bar_number"] - out["anchor_bar"]
        out["anchor_midpoint"] = ((data["open"] + close) / 2.0).where(out["capitulation"]).ffill()
        out["anchor_volume"] = volume.where(out["capitulation"]).ffill()
        out["quiet_selling"] = (
            (close < out["previous_close"])
            & (volume > 0.0)
            & (volume <= 0.55 * out["anchor_volume"])
            & (out["anchor_age"] >= 1.0)
        ).fillna(False)
        out["recent_quiet_count"] = out["quiet_selling"].astype(int).shift(1).rolling(3, min_periods=3).sum()
        out["midpoint_cross"] = (
            (close > out["anchor_midpoint"])
            & (out["previous_close"] <= out["anchor_midpoint"].shift(1))
            & (out["anchor_bar"] == out["anchor_bar"].shift(1))
        ).fillna(False)
        out["entry_ready"] = (
            out["midpoint_cross"]
            & out["anchor_age"].between(3.0, 10.0)
            & (out["recent_quiet_count"] >= 2.0)
            & (close > data["open"])
            & (close > out["previous_close"])
            & (volume >= 1.2 * out["baseline_volume"])
            & (out["baseline_volume"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        hold = max(1, int(params.hold_bars))
        entries = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        anchors = indicators["anchor_bar"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        exit_bar = -1
        used_anchor = -1.0
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                else:
                    # The shifted zero fills exactly hold bars after entry.
                    exit_bar = -1
                continue
            if entries[i] and np.isfinite(anchors[i]) and anchors[i] != used_anchor:
                signals[i] = 1
                exit_bar = i + hold
                used_anchor = anchors[i]
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
