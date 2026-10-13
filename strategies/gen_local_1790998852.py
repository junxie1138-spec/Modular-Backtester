from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    normalization_window: int = 40
    recovery_window: int = 5
    drawdown_window: int = 63
    minimum_drawdown: float = 0.03
    maximum_drawdown: float = 0.18
    energy_window: int = 10
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790998852"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(
            params.fractional_window + params.normalization_window + params.recovery_window + 2,
            params.energy_window + params.recovery_window + 1,
            params.drawdown_window,
        ))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        log_close = np.log(close)
        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.fractional_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        fractional_close = log_close * weights[0]
        for k in range(1, params.fractional_window):
            fractional_close = fractional_close + log_close.shift(k) * weights[k]
        out["fractional_close"] = fractional_close
        out["fractional_mean"] = fractional_close.rolling(params.normalization_window, min_periods=params.normalization_window).mean()
        out["fractional_std"] = fractional_close.rolling(params.normalization_window, min_periods=params.normalization_window).std(ddof=0)
        out["fractional_z"] = (fractional_close - out["fractional_mean"]) / out["fractional_std"].where(out["fractional_std"] > 0)
        out["previous_fractional_z"] = out["fractional_z"].shift(1)
        out["recent_fractional_trough"] = out["fractional_z"].shift(1).rolling(params.recovery_window, min_periods=params.recovery_window).min()
        out["log_return"] = log_close.diff()
        out["upside_energy"] = out["log_return"].clip(lower=0).pow(2).rolling(params.energy_window, min_periods=params.energy_window).sum()
        out["downside_energy"] = out["log_return"].clip(upper=0).pow(2).rolling(params.energy_window, min_periods=params.energy_window).sum()
        out["total_energy"] = out["upside_energy"] + out["downside_energy"]
        out["upside_share"] = out["upside_energy"] / out["total_energy"].where(out["total_energy"] > 0)
        out["previous_upside_share"] = out["upside_share"].shift(params.recovery_window)
        out["rolling_peak"] = close.rolling(params.drawdown_window, min_periods=params.drawdown_window).max()
        out["drawdown"] = 1.0 - close / out["rolling_peak"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        fractional_repair = (
            (indicators["fractional_z"] > -0.25)
            & (indicators["previous_fractional_z"] <= -0.25)
            & (indicators["recent_fractional_trough"] <= -1.0)
        )
        energy_rotation = (
            (indicators["upside_share"] >= 0.55)
            & (indicators["previous_upside_share"] <= 0.35)
        )
        drawdown_zone = (
            (indicators["drawdown"] >= params.minimum_drawdown)
            & (indicators["drawdown"] <= params.maximum_drawdown)
        )
        ready = indicators[[
            "fractional_z", "previous_fractional_z", "recent_fractional_trough",
            "upside_share", "previous_upside_share", "drawdown",
        ]].notna().all(axis=1)
        entries = (fractional_repair & energy_rotation & drawdown_zone & ready).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=np.int64)
        holding_bars = max(1, int(params.holding_bars))
        exit_bar = -1
        for i, entry in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Force a flat exit bar even if a fresh entry condition exists.
                    exit_bar = -1
            elif entry:
                exit_bar = i + holding_bars
                signal[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
