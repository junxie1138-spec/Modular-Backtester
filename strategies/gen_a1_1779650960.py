from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlasticDeformationParams:
    ma_window: int = 60
    std_window: int = 60
    yield_z: float = 1.5
    strain_rate_window: int = 5
    strain_rate_min: float = 0.25
    atr_window: int = 14
    trail_k_atr: float = 3.0
    max_hold_bars: int = 30
    vol_target_annual: float = 0.15
    vol_lookback: int = 20
    size_cap: float = 1.5
    size_floor: float = 0.1
    allow_short: bool = True


class GeneratedStrategy(BaseStrategy[PlasticDeformationParams]):
    strategy_id = "gen_a1_1779650960"

    @classmethod
    def params_type(cls):
        return PlasticDeformationParams

    @classmethod
    def warmup_bars(cls, params: PlasticDeformationParams) -> int:
        base = max(params.ma_window, params.std_window, params.atr_window, params.vol_lookback)
        return int(base + params.strain_rate_window + 2)

    def indicators(self, data: pd.DataFrame, params: PlasticDeformationParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        std = close.rolling(params.std_window, min_periods=params.std_window).std(ddof=0)
        std_safe = std.replace(0.0, np.nan)
        z = (close - ma) / std_safe
        z_prev = z.shift(1)
        srw = max(int(params.strain_rate_window), 1)
        strain_rate = (z - z.shift(srw)) / float(srw)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        rets = close.pct_change()
        realized_vol = rets.rolling(params.vol_lookback, min_periods=params.vol_lookback).std(ddof=0) * np.sqrt(252.0)

        return pd.DataFrame(
            {
                "z": z,
                "z_prev": z_prev,
                "strain_rate": strain_rate,
                "atr": atr,
                "realized_vol": realized_vol,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PlasticDeformationParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        z_prev = indicators["z_prev"].to_numpy(dtype=float)
        strain_rate = indicators["strain_rate"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        realized_vol = indicators["realized_vol"].to_numpy(dtype=float)

        raw_signal = np.zeros(n, dtype=np.int64)
        raw_size = np.ones(n, dtype=np.float64)

        eps = 1e-12
        position = 0
        entry_atr = 0.0
        high_water = -np.inf
        low_water = np.inf
        bars_held = 0
        current_size = 1.0

        for i in range(n):
            c = close[i]
            zi = z[i]
            zp = z_prev[i]
            sr = strain_rate[i]
            a = atr[i]
            rv = realized_vol[i]

            inputs_ok = not (
                np.isnan(zi)
                or np.isnan(zp)
                or np.isnan(sr)
                or np.isnan(a)
                or np.isnan(rv)
                or np.isnan(c)
            )

            if position == 0:
                if inputs_ok:
                    long_yield = (zp <= params.yield_z) and (zi > params.yield_z) and (sr >= params.strain_rate_min)
                    short_yield = (
                        params.allow_short
                        and (zp >= -params.yield_z)
                        and (zi < -params.yield_z)
                        and (sr <= -params.strain_rate_min)
                    )
                    if long_yield:
                        position = 1
                        entry_atr = float(a)
                        high_water = float(c)
                        bars_held = 0
                        strain = max(abs(zi) - params.yield_z, 0.0)
                        vol_scale = params.vol_target_annual / max(rv, eps)
                        current_size = float(
                            min(max(vol_scale * (1.0 + strain), params.size_floor), params.size_cap)
                        )
                        raw_signal[i] = 1
                        raw_size[i] = current_size
                    elif short_yield:
                        position = -1
                        entry_atr = float(a)
                        low_water = float(c)
                        bars_held = 0
                        strain = max(abs(zi) - params.yield_z, 0.0)
                        vol_scale = params.vol_target_annual / max(rv, eps)
                        current_size = float(
                            min(max(vol_scale * (1.0 + strain), params.size_floor), params.size_cap)
                        )
                        raw_signal[i] = -1
                        raw_size[i] = current_size
            else:
                bars_held += 1
                if position == 1:
                    if not np.isnan(c) and c > high_water:
                        high_water = float(c)
                    stop = high_water - params.trail_k_atr * entry_atr
                    exit_now = (not np.isnan(c) and c <= stop) or (bars_held >= params.max_hold_bars)
                    if exit_now:
                        raw_signal[i] = 0
                        raw_size[i] = 1.0
                        position = 0
                        current_size = 1.0
                        high_water = -np.inf
                        bars_held = 0
                    else:
                        raw_signal[i] = 1
                        raw_size[i] = current_size
                else:
                    if not np.isnan(c) and c < low_water:
                        low_water = float(c)
                    stop = low_water + params.trail_k_atr * entry_atr
                    exit_now = (not np.isnan(c) and c >= stop) or (bars_held >= params.max_hold_bars)
                    if exit_now:
                        raw_signal[i] = 0
                        raw_size[i] = 1.0
                        position = 0
                        current_size = 1.0
                        low_water = np.inf
                        bars_held = 0
                    else:
                        raw_signal[i] = -1
                        raw_size[i] = current_size

        df = pd.DataFrame(
            {
                "signal": raw_signal,
                "size": raw_size,
            },
            index=data.index,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df.loc[df["size"] <= 0.0, "size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
