from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDeficitParams:
    # Rolling window over which overnight gaps are accumulated into a deficit.
    gap_window: int = 5
    # Lookback for z-scoring the accumulated gap deficit.
    z_window: int = 60
    # Lower Schmitt threshold: deficit z below this turns the 'stretched-down' state ON.
    enter_z: float = -1.5
    # Upper Schmitt threshold: deficit z above this turns the state OFF (must exceed enter_z).
    exit_z: float = 0.3
    # Constant position size (positive float).
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[GapDeficitParams]):
    strategy_id = "gen_a1_1778913369"

    @classmethod
    def params_type(cls) -> type[GapDeficitParams]:
        return GapDeficitParams

    @staticmethod
    def warmup_bars(params: GapDeficitParams) -> int:
        # gap needs 1 prior bar; cumgap needs gap_window; z needs z_window of cumgap.
        return int(params.gap_window) + int(params.z_window) + 5

    def indicators(self, data: pd.DataFrame, params: GapDeficitParams) -> pd.DataFrame:
        gw = max(int(params.gap_window), 1)
        zw = max(int(params.z_window), 2)

        prior_close = data["close"].shift(1)
        # Overnight gap as a return; first bar is NaN by construction.
        gap = (data["open"] - prior_close) / prior_close.replace(0.0, np.nan)

        # Accumulated overnight deficit over the rolling window.
        cumgap = gap.rolling(gw, min_periods=gw).sum()

        roll_mean = cumgap.rolling(zw, min_periods=zw).mean()
        roll_std = cumgap.rolling(zw, min_periods=zw).std()
        roll_std = roll_std.replace(0.0, np.nan)
        cumgap_z = (cumgap - roll_mean) / roll_std

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["cumgap"] = cumgap
        out["cumgap_z"] = cumgap_z
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapDeficitParams,
    ) -> SignalFrame:
        z = indicators["cumgap_z"].to_numpy(dtype=float)
        n = len(z)

        enter_z = float(params.enter_z)
        # Guard hysteresis ordering: exit threshold must sit above entry threshold.
        exit_z = float(params.exit_z)
        if exit_z <= enter_z:
            exit_z = enter_z + 0.5

        # Schmitt-trigger state machine for the 'stretched-down' entry condition.
        stretched = np.zeros(n, dtype=bool)
        is_stretched = False
        for i in range(n):
            zi = z[i]
            if not np.isfinite(zi):
                is_stretched = False
                stretched[i] = False
                continue
            if not is_stretched and zi < enter_z:
                is_stretched = True
            elif is_stretched and zi > exit_z:
                is_stretched = False
            stretched[i] = is_stretched

        # Position machine: two-bar confirmation to enter, signal-reversal to exit.
        signal = np.zeros(n, dtype=int)
        in_pos = False
        for i in range(n):
            if i == 0:
                continue
            if not in_pos:
                # Two-bar confirmation: condition must hold this bar AND the prior bar.
                if stretched[i] and stretched[i - 1]:
                    in_pos = True
            else:
                # Signal-reversal exit: leave only when the entry condition flips off.
                if not stretched[i]:
                    in_pos = False
            signal[i] = 1 if in_pos else 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.base_size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
