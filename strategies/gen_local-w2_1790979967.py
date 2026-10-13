from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    fine_scale: int = 2
    recovery_scale: int = 4
    shock_scale: int = 8
    energy_window: int = 3
    drawdown_window: int = 40
    percentile_window: int = 126
    recovery_percentile: float = 0.70
    drawdown_percentile: float = 0.40


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1790979967"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        longest_scale = max(params.fine_scale, params.recovery_scale, params.shock_scale)
        return int(max(
            params.drawdown_window + params.percentile_window,
            2 * longest_scale + params.energy_window + params.percentile_window,
        ))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for name, scale in (
            ("fine", params.fine_scale),
            ("recovery", params.recovery_scale),
            ("shock", params.shock_scale),
        ):
            out[name + "_mean"] = out["log_close"].rolling(scale, min_periods=scale).mean()
            out[name + "_wavelet"] = (
                out[name + "_mean"] - out[name + "_mean"].shift(scale)
            ) * np.sqrt(scale / 2.0)
            out[name + "_energy"] = out[name + "_wavelet"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["upside_recovery_energy"] = out["recovery_wavelet"].clip(lower=0).pow(2).rolling(
            params.energy_window, min_periods=params.energy_window
        ).mean()
        out["total_energy"] = out["fine_energy"] + out["recovery_energy"] + out["shock_energy"]
        out["recovery_share"] = out["upside_recovery_energy"] / out["total_energy"].replace(0.0, np.nan)
        out["recovery_threshold"] = out["recovery_share"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.recovery_percentile).shift(1)
        out["rolling_peak"] = data["close"].rolling(
            params.drawdown_window, min_periods=params.drawdown_window
        ).max()
        out["drawdown"] = data["close"] / out["rolling_peak"].replace(0.0, np.nan) - 1.0
        out["drawdown_threshold"] = out["drawdown"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.drawdown_percentile).shift(1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        required = [
            "recovery_wavelet", "shock_wavelet", "recovery_share",
            "recovery_threshold", "drawdown", "drawdown_threshold",
        ]
        valid = indicators[required].notna().all(axis=1)
        entry_condition = (
            valid
            & (indicators["drawdown"] < 0.0)
            & (indicators["drawdown"] <= indicators["drawdown_threshold"])
            & (indicators["shock_wavelet"] < 0.0)
            & (indicators["recovery_wavelet"] > 0.0)
            & (indicators["recovery_share"] > indicators["recovery_threshold"])
        )
        df = pd.DataFrame(index=data.index)
        # Reversal exit: remain long only while the complete entry condition holds.
        df["signal"] = entry_condition.astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
