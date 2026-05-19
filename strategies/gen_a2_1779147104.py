from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_z_window: int = 60
    entry_z_lo: float = 0.25
    entry_z_hi: float = 1.75
    spike_z: float = 2.5
    refractory_bars: int = 5
    atr_window: int = 14
    atr_k: float = 2.5
    max_hold: int = 10
    ma_window: int = 200
    vol_window: int = 20
    target_vol: float = 0.15
    max_leverage: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779147104"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(
            params.ma_window,
            params.gap_z_window + 1,
            params.atr_window + 1,
            params.vol_window + 1,
        )) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]

        prior_close = close.shift(1)
        gap = open_ / prior_close - 1.0

        gap_mean = gap.rolling(params.gap_z_window).mean()
        gap_std = gap.rolling(params.gap_z_window).std()
        gap_z = (gap - gap_mean) / gap_std.replace(0.0, np.nan)

        ma = close.rolling(params.ma_window).mean()

        tr = pd.concat([
            high - low,
            (high - prior_close).abs(),
            (low - prior_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        ret = close.pct_change()
        rvol = ret.rolling(params.vol_window).std() * np.sqrt(252.0)
        vol_size = params.target_vol / rvol.replace(0.0, np.nan)
        vol_size = vol_size.clip(lower=0.05, upper=params.max_leverage)

        out = pd.DataFrame(index=data.index)
        out["gap_z"] = gap_z
        out["ma"] = ma
        out["atr"] = atr
        out["vol_size"] = vol_size
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        gap_z = indicators["gap_z"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        raw = np.zeros(n, dtype=int)

        in_pos = False
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0
        refractory = 0

        for i in range(n):
            gz = gap_z[i]

            if np.isfinite(gz) and abs(gz) > params.spike_z:
                refractory = params.refractory_bars

            if in_pos:
                bars_held += 1
                stop_level = entry_price - params.atr_k * entry_atr
                exit_now = (close[i] < stop_level) or (bars_held >= params.max_hold)
                if exit_now:
                    raw[i] = 0
                    in_pos = False
                else:
                    raw[i] = 1
            else:
                can_enter = (
                    np.isfinite(gz)
                    and np.isfinite(ma[i])
                    and np.isfinite(atr[i])
                    and refractory == 0
                    and close[i] > ma[i]
                    and params.entry_z_lo <= gz <= params.entry_z_hi
                )
                if can_enter:
                    in_pos = True
                    entry_price = close[i]
                    entry_atr = atr[i]
                    bars_held = 0
                    raw[i] = 1
                else:
                    raw[i] = 0

            if refractory > 0:
                refractory -= 1

        signal = pd.Series(raw, index=data.index)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal.shift(1).fillna(0).astype(int)

        size = indicators["vol_size"].ffill().fillna(0.05)
        size = size.clip(lower=0.05, upper=params.max_leverage)
        df["size"] = size.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
