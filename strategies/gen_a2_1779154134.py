from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    vwap_window: int = 20
    vol_window: int = 20
    atr_window: int = 14
    confirm_window: int = 5
    entry_disp: float = 0.04
    vol_spike: float = 1.5
    disp_cap: float = 0.10
    base_size: float = 0.40
    size_scale: float = 0.60
    profit_trigger: float = 0.04
    trail_atr_mult: float = 2.5
    init_atr_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779154134"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(
            params.vwap_window,
            params.atr_window,
            params.vol_window + params.confirm_window,
        )) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]
        volume = data["volume"]

        typical = (high + low + close) / 3.0
        pv = typical * volume
        vol_sum = volume.rolling(params.vwap_window).sum()
        vwap = pv.rolling(params.vwap_window).sum() / vol_sum.replace(0.0, np.nan)
        displacement = close / vwap - 1.0

        vol_sma = volume.rolling(params.vol_window).mean()
        vol_ratio = volume / vol_sma.replace(0.0, np.nan)
        vol_spike_recent = vol_ratio.rolling(params.confirm_window).max()

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        green = (close > open_).astype(float)

        out = pd.DataFrame(index=data.index)
        out["vwap"] = vwap
        out["displacement"] = displacement
        out["vol_spike_recent"] = vol_spike_recent
        out["atr"] = atr
        out["green"] = green
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        idx = data.index
        n = len(idx)

        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)

        displacement = indicators["displacement"].to_numpy(dtype=float)
        vol_spike_recent = indicators["vol_spike_recent"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        green = indicators["green"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        high_water = 0.0
        breakeven = False
        pos_size = 1.0

        disp_cap = params.disp_cap if params.disp_cap > 0 else 1e-9

        for i in range(n):
            d = displacement[i]
            vs = vol_spike_recent[i]
            a = atr[i]
            g = green[i]

            valid = np.isfinite(d) and np.isfinite(vs) and np.isfinite(a)

            if not in_pos:
                arm = (
                    valid
                    and (d <= -params.entry_disp)
                    and (vs >= params.vol_spike)
                )
                if arm and g > 0.5:
                    in_pos = True
                    entry_price = close[i]
                    high_water = high[i]
                    breakeven = False
                    stop = entry_price - params.init_atr_mult * a
                    frac = min(abs(d) / disp_cap, 1.0)
                    sz = params.base_size + params.size_scale * frac
                    sz = min(max(sz, 0.05), 1.0)
                    pos_size = sz
                    signal[i] = 1
                    size[i] = sz
            else:
                if high[i] > high_water:
                    high_water = high[i]
                if (not breakeven) and high[i] >= entry_price * (1.0 + params.profit_trigger):
                    breakeven = True
                    if entry_price > stop:
                        stop = entry_price
                if breakeven and np.isfinite(a):
                    trail = high_water - params.trail_atr_mult * a
                    if trail > stop:
                        stop = trail
                if low[i] <= stop:
                    in_pos = False
                    signal[i] = 0
                    size[i] = pos_size
                else:
                    signal[i] = 1
                    size[i] = pos_size

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].fillna(1.0).clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
