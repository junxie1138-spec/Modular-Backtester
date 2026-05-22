from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    compress_window: int = 8
    baseline_window: int = 50
    vol_ma_window: int = 20
    compress_thresh: float = 0.70
    vol_mult: float = 1.3
    refractory_bars: int = 3
    size_floor: float = 0.40
    size_cap: float = 1.00


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778906501"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.baseline_window, params.vol_ma_window, params.compress_window)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        openp = data["open"]
        volume = data["volume"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        short_tr = tr.rolling(params.compress_window, min_periods=params.compress_window).mean()
        long_tr = tr.rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        compression_ratio = short_tr / long_tr.replace(0.0, np.nan)

        vol_ma = volume.rolling(params.vol_ma_window, min_periods=params.vol_ma_window).mean()
        vol_ratio = volume / vol_ma.replace(0.0, np.nan)

        up_bar = (close > openp).astype(float)
        down_bar = (close < openp).astype(float)
        vol_confirm = (vol_ratio > params.vol_mult).astype(float)

        thresh = params.compress_thresh if params.compress_thresh > 0.0 else 1e-9
        compression_strength = ((thresh - compression_ratio) / thresh).clip(lower=0.0, upper=1.0)
        vmult = params.vol_mult if params.vol_mult > 0.0 else 1e-9
        volume_strength = ((vol_ratio - params.vol_mult) / vmult).clip(lower=0.0, upper=1.0)
        strength = 0.5 * (compression_strength + volume_strength)

        size_factor = params.size_floor + (params.size_cap - params.size_floor) * strength

        ind = pd.DataFrame(index=data.index)
        ind["compression_ratio"] = compression_ratio
        ind["vol_ratio"] = vol_ratio.fillna(0.0)
        ind["up_bar"] = up_bar.fillna(0.0)
        ind["down_bar"] = down_bar.fillna(0.0)
        ind["vol_confirm"] = vol_confirm.fillna(0.0)
        ind["strength"] = strength.fillna(0.0)
        ind["size_factor"] = size_factor.fillna(params.size_floor)
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        compression = indicators["compression_ratio"].to_numpy(dtype=float)
        up_bar = indicators["up_bar"].to_numpy(dtype=float)
        down_bar = indicators["down_bar"].to_numpy(dtype=float)
        vol_confirm = indicators["vol_confirm"].to_numpy(dtype=float)
        size_factor = indicators["size_factor"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=np.int64)
        size_arr = np.ones(n, dtype=float)

        position = 0
        refractory = 0
        held_size = 1.0

        for i in range(n):
            comp = compression[i]
            entry = (
                np.isfinite(comp)
                and comp < params.compress_thresh
                and up_bar[i] == 1.0
                and vol_confirm[i] == 1.0
            )
            exit_flip = down_bar[i] == 1.0 and vol_confirm[i] == 1.0

            if position == 1:
                if exit_flip:
                    position = 0
                    held_size = 1.0
            else:
                if entry and refractory == 0:
                    position = 1
                    sf = size_factor[i]
                    if not np.isfinite(sf):
                        sf = params.size_floor
                    held_size = float(min(max(sf, params.size_floor), params.size_cap))
                    refractory = params.refractory_bars

            signal[i] = position
            size_arr[i] = held_size if position == 1 else 1.0

            if refractory > 0:
                refractory -= 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size_arr
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=params.size_floor, upper=params.size_cap)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
