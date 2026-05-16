from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapInfectionParams:
    gap_lookback: int = 20
    gap_z_threshold: float = 1.0
    vol_lookback: int = 20
    vol_surge_mult: float = 1.5
    susceptibility_ma: int = 50
    max_extension: float = 0.05
    profit_target: float = 0.04
    time_stop: int = 10
    base_size: float = 0.40
    size_gain: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[GapInfectionParams]):
    """Volume-confirmed up-gap as an infection event in a susceptible market."""

    strategy_id = "gen_a1_1778893282"

    @classmethod
    def params_type(cls) -> type[GapInfectionParams]:
        return GapInfectionParams

    @staticmethod
    def warmup_bars(params: GapInfectionParams) -> int:
        return int(max(params.gap_lookback, params.vol_lookback,
                       params.susceptibility_ma)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapInfectionParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        volume = data["volume"]

        prev_close = close.shift(1)
        gap = open_ / prev_close - 1.0

        gap_mean = gap.rolling(params.gap_lookback).mean()
        gap_std = gap.rolling(params.gap_lookback).std()
        gap_z = (gap - gap_mean) / gap_std.replace(0.0, np.nan)

        vol_avg = volume.rolling(params.vol_lookback).mean()
        vol_ratio = volume / vol_avg.replace(0.0, np.nan)

        sma = close.rolling(params.susceptibility_ma).mean()
        extension = close / sma.replace(0.0, np.nan) - 1.0

        susceptibility = ((params.max_extension - extension)
                          / params.max_extension).clip(lower=0.0, upper=1.0)

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["gap_z"] = gap_z
        out["vol_ratio"] = vol_ratio
        out["extension"] = extension
        out["susceptibility"] = susceptibility
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        idx = data.index
        n = len(data)
        close = data["close"].to_numpy(dtype=np.float64)

        gap = indicators["gap"].to_numpy(dtype=np.float64)
        gap_z = indicators["gap_z"].to_numpy(dtype=np.float64)
        vol_ratio = indicators["vol_ratio"].to_numpy(dtype=np.float64)
        extension = indicators["extension"].to_numpy(dtype=np.float64)
        susc = indicators["susceptibility"].to_numpy(dtype=np.float64)

        # Infection event: a positive overnight gap that is statistically large
        # AND confirmed by a volume surge, landing in an un-extended
        # (susceptible) market with fuel left for the move to spread.
        entry = (
            (gap > 0.0)
            & (gap_z > params.gap_z_threshold)
            & (vol_ratio > params.vol_surge_mult)
            & (extension < params.max_extension)
        )
        bad = np.isnan(gap_z) | np.isnan(vol_ratio) | np.isnan(extension)
        entry = np.where(bad, False, entry)

        # Signal-scaled size: stronger gap z-score + heavier volume surge =>
        # larger position; depleted susceptibility damps it toward half size.
        strength = (np.maximum(np.nan_to_num(gap_z, nan=0.0)
                               - params.gap_z_threshold, 0.0)
                    + np.maximum(np.nan_to_num(vol_ratio, nan=0.0)
                                 - params.vol_surge_mult, 0.0))
        susc_safe = np.nan_to_num(susc, nan=0.0)
        size_score = ((params.base_size + params.size_gain * strength)
                      * (0.5 + 0.5 * susc_safe))
        size_score = np.clip(size_score, 0.10, params.max_size)

        pos = np.zeros(n, dtype=np.int64)
        size_arr = np.ones(n, dtype=np.float64)

        in_pos = False
        entry_px = 0.0
        held = 0
        cur_size = 1.0
        for i in range(n):
            if not in_pos:
                if bool(entry[i]):
                    in_pos = True
                    entry_px = close[i]
                    held = 0
                    cur_size = float(size_score[i])
                    pos[i] = 1
                    size_arr[i] = cur_size
            else:
                held += 1
                ret = (close[i] / entry_px - 1.0) if entry_px > 0.0 else 0.0
                if ret >= params.profit_target or held >= params.time_stop:
                    in_pos = False
                    pos[i] = 0
                else:
                    pos[i] = 1
                    size_arr[i] = cur_size

        df = pd.DataFrame(index=idx)
        df["signal"] = pd.Series(pos, index=idx)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(size_arr, index=idx).shift(1).fillna(1.0).clip(lower=0.10)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
