from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StealthAccumulationParams:
    volume_window: int = 20
    volume_z_threshold: float = 1.5
    vol_window: int = 20
    vol_pct_window: int = 100
    vol_pct_ceiling: float = 0.40
    hold_bars: int = 7
    target_annual_vol: float = 0.15
    max_size: float = 1.0
    min_size: float = 0.10


class GeneratedStrategy(BaseStrategy[StealthAccumulationParams]):
    strategy_id = "gen_a1_1779648357"

    @classmethod
    def params_type(cls):
        return StealthAccumulationParams

    @classmethod
    def warmup_bars(cls, params: StealthAccumulationParams) -> int:
        longest = max(
            int(params.vol_pct_window),
            int(params.volume_window),
            int(params.vol_window),
        )
        return int(longest) + 2

    def indicators(self, data: pd.DataFrame, params: StealthAccumulationParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        ret = close.pct_change()

        vol_mean = volume.rolling(
            int(params.volume_window), min_periods=int(params.volume_window)
        ).mean()
        vol_std = volume.rolling(
            int(params.volume_window), min_periods=int(params.volume_window)
        ).std()
        vol_std_safe = vol_std.replace(0.0, np.nan)
        volume_z = (volume - vol_mean) / vol_std_safe

        realized_vol = ret.rolling(
            int(params.vol_window), min_periods=int(params.vol_window)
        ).std()
        vol_pct_rank = realized_vol.rolling(
            int(params.vol_pct_window), min_periods=int(params.vol_pct_window)
        ).rank(pct=True)
        annualized_vol = realized_vol * np.sqrt(252.0)

        out = pd.DataFrame(
            {
                "ret": ret,
                "volume_z": volume_z,
                "realized_vol": realized_vol,
                "vol_pct_rank": vol_pct_rank,
                "annualized_vol": annualized_vol,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StealthAccumulationParams,
    ) -> SignalFrame:
        n = len(data)
        ret_arr = indicators["ret"].to_numpy(dtype=float, copy=False)
        volume_z = indicators["volume_z"].to_numpy(dtype=float, copy=False)
        vol_pct_rank = indicators["vol_pct_rank"].to_numpy(dtype=float, copy=False)
        annualized_vol = indicators["annualized_vol"].to_numpy(dtype=float, copy=False)

        vz_clean = np.where(np.isnan(volume_z), -np.inf, volume_z)
        vpr_clean = np.where(np.isnan(vol_pct_rank), np.inf, vol_pct_rank)
        ret_clean = np.where(np.isnan(ret_arr), 0.0, ret_arr)

        prim_volume_up = (vz_clean >= float(params.volume_z_threshold)) & (ret_clean > 0.0)
        prim_quiet_vol = vpr_clean <= float(params.vol_pct_ceiling)
        entry_mask = prim_volume_up & prim_quiet_vol

        raw_signal = np.zeros(n, dtype=np.int64)
        raw_size = np.zeros(n, dtype=np.float64)

        hold_bars = max(int(params.hold_bars), 1)
        max_size = float(params.max_size)
        min_size = float(params.min_size)
        target_av = float(params.target_annual_vol)

        hold_remaining = 0
        current_size = 0.0
        for i in range(n):
            if hold_remaining > 0:
                raw_signal[i] = 1
                raw_size[i] = current_size
                hold_remaining -= 1
                continue
            if entry_mask[i]:
                av = annualized_vol[i]
                if np.isnan(av) or av <= 1e-8:
                    s = min_size
                else:
                    s = target_av / av
                s = float(np.clip(s, min_size, max_size))
                current_size = s
                raw_signal[i] = 1
                raw_size[i] = s
                hold_remaining = hold_bars - 1
            else:
                raw_signal[i] = 0
                raw_size[i] = 0.0
                current_size = 0.0

        signal_series = (
            pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        size_shifted = pd.Series(raw_size, index=data.index).shift(1)

        size_series = size_shifted.where(signal_series != 0, other=max_size)
        size_series = size_series.fillna(max_size)
        size_series = size_series.where(size_series > 0.0, other=max_size)
        size_series = size_series.astype(float)

        out = pd.DataFrame(index=data.index)
        out["signal"] = signal_series
        out["size"] = size_series

        return SignalFrame(data=out, signal_column="signal", size_column="size")
