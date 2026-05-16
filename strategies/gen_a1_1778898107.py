from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    atr_period: int = 14
    tr_median_window: int = 50
    min_quiet: int = 4
    roc_period: int = 5
    accel_vol_window: int = 20
    accel_thresh: float = 0.25
    ma_regime: int = 200
    trail_k: float = 2.5
    max_hold: int = 12
    allow_short: bool = True


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1778898107"

    @classmethod
    def params_type(cls) -> type[GenA1Params]:
        return GenA1Params

    def warmup_bars(self, params: GenA1Params) -> int:
        lookbacks = [
            int(params.atr_period),
            int(params.tr_median_window),
            int(params.ma_regime),
            int(params.roc_period) + int(params.accel_vol_window) + 1,
        ]
        return max(lookbacks) + 2

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(int(params.atr_period), min_periods=int(params.atr_period)).mean()
        tr_median = tr.rolling(
            int(params.tr_median_window), min_periods=int(params.tr_median_window)
        ).median()
        ma_regime = close.rolling(
            int(params.ma_regime), min_periods=int(params.ma_regime)
        ).mean()

        roc = close.pct_change(int(params.roc_period))
        accel = roc.diff()
        accel_std = accel.rolling(
            int(params.accel_vol_window), min_periods=int(params.accel_vol_window)
        ).std()
        accel_z = accel / accel_std.where(accel_std > 0)

        out = pd.DataFrame(index=data.index)
        out["tr"] = tr
        out["atr"] = atr
        out["tr_median"] = tr_median
        out["ma_regime"] = ma_regime
        out["accel_z"] = accel_z
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        tr = indicators["tr"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        tr_median = indicators["tr_median"].to_numpy(dtype=float)
        ma = indicators["ma_regime"].to_numpy(dtype=float)
        accel_z = indicators["accel_z"].to_numpy(dtype=float)
        n = len(close)

        state = np.zeros(n, dtype=int)
        pos = 0
        entry_idx = -1
        water = 0.0
        quiet_streak = 0

        min_quiet = int(params.min_quiet)
        max_hold = int(params.max_hold)
        trail_k = float(params.trail_k)
        thresh = float(params.accel_thresh)
        allow_short = bool(params.allow_short)

        for i in range(n):
            med_i = tr_median[i]
            is_quiet = (not np.isnan(med_i)) and (tr[i] < med_i)
            prev_streak = quiet_streak
            if is_quiet:
                quiet_streak += 1
            else:
                quiet_streak = 0

            ready = (
                not np.isnan(atr[i])
                and not np.isnan(ma[i])
                and not np.isnan(accel_z[i])
                and atr[i] > 0.0
            )

            if pos == 0:
                if ready and (not is_quiet) and prev_streak >= min_quiet:
                    az = accel_z[i]
                    if az > thresh and close[i] > ma[i]:
                        pos = 1
                        entry_idx = i
                        water = close[i]
                    elif allow_short and az < -thresh and close[i] < ma[i]:
                        pos = -1
                        entry_idx = i
                        water = close[i]
            elif pos == 1:
                water = max(water, close[i])
                a = atr[i] if (not np.isnan(atr[i]) and atr[i] > 0.0) else 0.0
                stop = water - trail_k * a
                if close[i] < stop or (i - entry_idx) >= max_hold:
                    pos = 0
            elif pos == -1:
                water = min(water, close[i])
                a = atr[i] if (not np.isnan(atr[i]) and atr[i] > 0.0) else 0.0
                stop = water + trail_k * a
                if close[i] > stop or (i - entry_idx) >= max_hold:
                    pos = 0

            state[i] = pos

        signal = pd.Series(state, index=data.index)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal.shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
