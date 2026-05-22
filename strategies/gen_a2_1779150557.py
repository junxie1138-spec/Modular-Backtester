from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ThrustParams:
    vol_lookback: int = 20
    vol_mult: float = 1.5
    atr_lookback: int = 14
    range_mult: float = 1.3
    close_frac: float = 0.66
    target_vol: float = 0.15
    rv_lookback: int = 20
    size_min: float = 0.3
    size_max: float = 2.0
    be_trigger: float = 0.03
    init_stop_atr: float = 1.5
    trail_atr: float = 2.5
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[ThrustParams]):
    strategy_id = "gen_a2_1779150557"

    @classmethod
    def params_type(cls):
        return ThrustParams

    @staticmethod
    def warmup_bars(params: ThrustParams) -> int:
        return int(max(params.vol_lookback, params.atr_lookback + 1, params.rv_lookback + 1)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: ThrustParams) -> pd.DataFrame:
        p = params
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        volume = data["volume"].astype(float)

        ret = close.pct_change()

        vol_sma = volume.rolling(p.vol_lookback, min_periods=p.vol_lookback).mean()
        vol_ratio = volume / vol_sma.replace(0.0, np.nan)

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(p.atr_lookback, min_periods=p.atr_lookback).mean()

        rng = high - low
        close_pos = ((close - low) / rng.where(rng > 0.0, np.nan)).where(rng > 0.0, 0.5)

        rv = ret.rolling(p.rv_lookback, min_periods=p.rv_lookback).std() * np.sqrt(252.0)

        prim_volume = (vol_ratio >= p.vol_mult) & (ret > 0.0)
        prim_thrust = (tr >= p.range_mult * atr.shift(1)) & (close_pos >= p.close_frac)
        entry = (prim_volume & prim_thrust).fillna(False)

        size = (p.target_vol / rv.where(rv > 0.0, np.nan))
        size = size.clip(lower=p.size_min, upper=p.size_max).fillna(p.size_min)

        ind = pd.DataFrame(index=data.index)
        ind["ret"] = ret
        ind["vol_ratio"] = vol_ratio
        ind["tr"] = tr
        ind["atr"] = atr
        ind["close_pos"] = close_pos
        ind["rv"] = rv
        ind["entry"] = entry.astype(bool)
        ind["size"] = size
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ThrustParams,
    ) -> SignalFrame:
        p = params
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].to_numpy(dtype=bool)

        raw = np.zeros(n, dtype=int)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        highest = 0.0
        bars_held = 0
        be_armed = False

        for i in range(n):
            if not in_pos:
                if entry[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                    in_pos = True
                    entry_price = close[i]
                    highest = close[i]
                    stop = entry_price - p.init_stop_atr * atr[i]
                    be_armed = False
                    bars_held = 0
                    raw[i] = 1
                else:
                    raw[i] = 0
            else:
                bars_held += 1
                if close[i] > highest:
                    highest = close[i]
                if high[i] >= entry_price * (1.0 + p.be_trigger):
                    be_armed = True
                    if entry_price > stop:
                        stop = entry_price
                if be_armed and np.isfinite(atr[i]) and atr[i] > 0.0:
                    trail = highest - p.trail_atr * atr[i]
                    if trail > stop:
                        stop = trail
                if close[i] <= stop or bars_held >= p.max_hold:
                    in_pos = False
                    raw[i] = 0
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size = indicators["size"].to_numpy(dtype=float)
        size_series = pd.Series(size, index=data.index).shift(1).fillna(p.size_min)
        size_series = size_series.clip(lower=p.size_min, upper=p.size_max)
        df["size"] = size_series.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
