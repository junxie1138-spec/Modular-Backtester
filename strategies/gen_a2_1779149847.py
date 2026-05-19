from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringReleaseParams:
    roc_period: int = 3
    load_window: int = 5
    load_count: int = 3
    profit_target: float = 0.01
    time_stop: int = 2
    size_gain: float = 0.5
    require_up_bar: bool = True


class GeneratedStrategy(BaseStrategy[SpringReleaseParams]):
    strategy_id = "gen_a2_1779149847"

    @classmethod
    def params_type(cls) -> type[SpringReleaseParams]:
        return SpringReleaseParams

    @staticmethod
    def warmup_bars(params: SpringReleaseParams) -> int:
        # roc uses pct_change(roc_period); accel = roc.diff() adds 1;
        # loaded_count/tension roll over load_window bars.
        return int(params.roc_period + params.load_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SpringReleaseParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        roc_period = max(1, int(params.roc_period))
        load_window = max(2, int(params.load_window))

        roc = close.pct_change(roc_period)
        accel = roc.diff()

        neg = (accel < 0).astype(float)
        loaded_count = neg.rolling(load_window, min_periods=1).sum()

        accel_std = accel.rolling(load_window, min_periods=2).std()
        deepest = (-accel).rolling(load_window, min_periods=1).max()
        tension = (deepest / accel_std).replace([np.inf, -np.inf], np.nan)

        out = pd.DataFrame(index=data.index)
        out["roc"] = roc
        out["accel"] = accel
        out["loaded_count"] = loaded_count.fillna(0.0)
        out["tension"] = tension.fillna(0.0).clip(lower=0.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SpringReleaseParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        n = len(close)

        accel = indicators["accel"].to_numpy(dtype=float)
        loaded = indicators["loaded_count"].to_numpy(dtype=float)
        tension = indicators["tension"].to_numpy(dtype=float)

        prev_accel = np.empty(n, dtype=float)
        prev_accel[0] = np.nan
        if n > 1:
            prev_accel[1:] = accel[:-1]

        cross_up = (
            np.isfinite(accel)
            & np.isfinite(prev_accel)
            & (accel > 0.0)
            & (prev_accel <= 0.0)
        )
        is_loaded = loaded >= float(params.load_count)

        up_bar = np.ones(n, dtype=bool)
        if params.require_up_bar and n > 1:
            up_bar[0] = False
            up_bar[1:] = close[1:] > close[:-1]
        elif params.require_up_bar:
            up_bar[0] = False

        entry = cross_up & is_loaded & up_bar

        warm = int(params.roc_period + params.load_window + 1)
        pt = float(params.profit_target)
        ts = max(1, int(params.time_stop))

        signal = np.zeros(n, dtype=int)
        i = max(warm, 0)
        while i < n:
            if entry[i] and (i + 1) < n:
                entry_ref = close[i]
                exit_j = n - 1
                for j in range(i + 1, n):
                    held = j - i
                    gain = (close[j] / entry_ref) - 1.0 if entry_ref > 0.0 else 0.0
                    if gain >= pt or held >= ts:
                        exit_j = j
                        break
                # hold (decision bars) from entry i up to bar before exit_j
                signal[i:exit_j] = 1
                i = exit_j
            else:
                i += 1

        size = np.clip(1.0 + float(params.size_gain) * tension, 0.25, 2.0)
        size = np.where(np.isfinite(size), size, 1.0)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size.astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
