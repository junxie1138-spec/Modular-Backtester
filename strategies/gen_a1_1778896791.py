from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    roc_period: int = 5
    accel_norm_window: int = 20
    atr_period: int = 14
    vol_window: int = 20
    target_vol: float = 0.15
    entry_z: float = 0.5
    accel_clip: float = 3.0
    strength_floor: float = 0.35
    be_pct: float = 0.02
    init_stop_atr: float = 2.0
    trail_atr: float = 2.5
    max_hold: int = 5
    min_size: float = 0.25
    max_size: float = 2.0


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1778896791"

    @classmethod
    def params_type(cls) -> type[GenA1Params]:
        return GenA1Params

    @staticmethod
    def warmup_bars(params: GenA1Params) -> int:
        accel_chain = params.roc_period + 1 + params.accel_norm_window
        return int(max(accel_chain, params.atr_period + 1, params.vol_window + 1)) + 5

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ind = pd.DataFrame(index=data.index)

        # Rate-of-change and its acceleration (second difference of momentum).
        roc = close.pct_change(params.roc_period)
        accel = roc.diff()
        accel_std = accel.rolling(params.accel_norm_window, min_periods=2).std()
        accel_std = accel_std.replace(0.0, np.nan)
        accel_z = accel / accel_std

        # Average True Range for stop placement.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=2).mean()

        # Annualized realized volatility -> volatility-targeting multiplier.
        daily_ret = close.pct_change()
        realized_vol = daily_ret.rolling(params.vol_window, min_periods=3).std() * np.sqrt(252.0)
        realized_vol = realized_vol.replace(0.0, np.nan)
        vol_mult = (params.target_vol / realized_vol).clip(params.min_size, params.max_size)

        # Signal-scaled component: stronger acceleration -> larger size.
        strength = accel_z.clip(lower=0.0, upper=params.accel_clip) / params.accel_clip
        signal_mult = params.strength_floor + (1.0 - params.strength_floor) * strength

        size_mult = (vol_mult * signal_mult).clip(params.min_size, params.max_size)

        ind["roc"] = roc
        ind["accel"] = accel
        ind["accel_z"] = accel_z
        ind["atr"] = atr
        ind["size_mult"] = size_mult
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)

        accel = indicators["accel"].to_numpy(dtype=float)
        accel_z = indicators["accel_z"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        size_mult = indicators["size_mult"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)

        accel_prev = np.concatenate(([np.nan], accel[:-1]))
        cross_up = (accel_prev <= 0.0) & (accel > 0.0)
        valid = ~np.isnan(accel_z) & ~np.isnan(atr) & ~np.isnan(size_mult)
        entry_cond = cross_up & valid & (accel_z >= params.entry_z)

        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=np.float64)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        bars_held = 0
        breakeven = False
        entry_size = 1.0

        for i in range(n):
            if not in_pos:
                if entry_cond[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                    in_pos = True
                    entry_price = close[i]
                    stop = entry_price - params.init_stop_atr * atr[i]
                    bars_held = 0
                    breakeven = False
                    entry_size = float(size_mult[i])
                    if not np.isfinite(entry_size) or entry_size <= 0.0:
                        entry_size = 1.0
                    signal[i] = 1
                    size[i] = entry_size
            else:
                bars_held += 1
                a = atr[i]
                if not np.isfinite(a):
                    a = 0.0
                # Breakeven: once price reaches +be_pct, lift the stop to entry.
                if (not breakeven) and high[i] >= entry_price * (1.0 + params.be_pct):
                    breakeven = True
                    if entry_price > stop:
                        stop = entry_price
                # After breakeven, trail by k*ATR; stop only ratchets up.
                if breakeven and a > 0.0:
                    trail = close[i] - params.trail_atr * a
                    if trail > stop:
                        stop = trail
                exit_now = (low[i] <= stop) or (bars_held >= params.max_hold)
                if exit_now:
                    in_pos = False
                    signal[i] = 0
                    size[i] = 1.0
                else:
                    signal[i] = 1
                    size[i] = entry_size

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].where(df["size"] > 0.0, 1.0)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
