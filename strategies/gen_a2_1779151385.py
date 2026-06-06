from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    channel_window: int = 20
    smooth_window: int = 3
    confirm_bars: int = 2
    accel_threshold: float = 0.0
    profit_target: float = 0.04
    time_stop: int = 5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779151385"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.channel_window) + int(params.smooth_window) + int(params.confirm_bars) + 3

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        cw = max(2, int(params.channel_window))
        sw = max(1, int(params.smooth_window))

        high = data["high"]
        low = data["low"]
        close = data["close"]

        roll_high = high.rolling(cw, min_periods=cw).max()
        roll_low = low.rolling(cw, min_periods=cw).min()
        rng = (roll_high - roll_low).replace(0.0, np.nan)

        # Relative position of close within its Donchian range, in [0, 1].
        pos = ((close - roll_low) / rng).clip(lower=0.0, upper=1.0)
        pos_smooth = pos.rolling(sw, min_periods=sw).mean()

        # Velocity and acceleration of the relative-position state variable.
        vel = pos_smooth.diff()
        accel = vel.diff()

        out = pd.DataFrame(index=data.index)
        out["pos"] = pos
        out["pos_smooth"] = pos_smooth
        out["vel"] = vel
        out["accel"] = accel
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        confirm = max(1, int(params.confirm_bars))
        thr = float(params.accel_threshold)
        pt = float(params.profit_target)
        tstop = max(1, int(params.time_stop))

        vel = indicators["vel"]
        accel = indicators["accel"]

        # Relative position rising AND accelerating -> long candidate; mirror -> short.
        long_raw = (vel > 0.0) & (accel > thr)
        short_raw = (vel < 0.0) & (accel < -thr)
        long_raw = long_raw.fillna(False)
        short_raw = short_raw.fillna(False)

        # Two-bar (confirm_bars) confirmation: condition held every bar in the window.
        long_confirm = (
            long_raw.rolling(confirm, min_periods=confirm).sum() == confirm
        ).fillna(False).to_numpy()
        short_confirm = (
            short_raw.rolling(confirm, min_periods=confirm).sum() == confirm
        ).fillna(False).to_numpy()

        close = data["close"].to_numpy(dtype=float)
        n = len(close)
        raw = np.zeros(n, dtype=int)

        pos_state = 0
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if pos_state != 0:
                bars_held += 1
                cur = close[i]
                if entry_price > 0.0 and np.isfinite(cur):
                    if pos_state == 1:
                        ret = cur / entry_price - 1.0
                    else:
                        ret = entry_price / cur - 1.0
                else:
                    ret = 0.0
                # profit-target OR time-stop, whichever fires first.
                if ret >= pt or bars_held >= tstop:
                    pos_state = 0
                    entry_price = 0.0
                    bars_held = 0
                    raw[i] = 0
                else:
                    raw[i] = pos_state
                continue

            if long_confirm[i]:
                pos_state = 1
                entry_price = close[i]
                bars_held = 0
                raw[i] = 1
            elif short_confirm[i]:
                pos_state = -1
                entry_price = close[i]
                bars_held = 0
                raw[i] = -1
            else:
                raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
