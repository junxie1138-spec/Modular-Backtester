from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    lookback_high: int = 60
    atr_period: int = 14
    vol_period: int = 20
    entry_threshold: float = 0.04
    release_threshold: float = 0.01
    max_depth: float = 0.15
    stop_atr_mult: float = 2.5
    target_vol: float = 0.15
    max_size_mult: float = 3.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1779877607"

    @classmethod
    def params_type(cls):
        return GenA1Params

    @classmethod
    def warmup_bars(cls, params: GenA1Params) -> int:
        return int(max(params.lookback_high, params.atr_period, params.vol_period)) + 2

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        rolling_max = close.rolling(params.lookback_high, min_periods=params.lookback_high).max()
        drawdown_pct = close / rolling_max - 1.0

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        ret = close.pct_change()
        realised_vol_daily = ret.rolling(params.vol_period, min_periods=params.vol_period).std()
        realised_vol_annual = realised_vol_daily * np.sqrt(252.0)

        out = pd.DataFrame(
            {
                "drawdown_pct": drawdown_pct,
                "atr": atr,
                "realised_vol": realised_vol_annual,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy()
        dd = indicators["drawdown_pct"].to_numpy()
        atr = indicators["atr"].to_numpy()
        rv = indicators["realised_vol"].to_numpy()

        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=np.float64)

        in_position = False
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0
        re_entry_armed = True
        current_size = 1.0

        entry_thr = float(params.entry_threshold)
        release_thr = float(params.release_threshold)
        max_depth = float(params.max_depth)
        stop_mult = float(params.stop_atr_mult)
        target_vol = float(params.target_vol)
        max_size_mult = float(params.max_size_mult)
        max_hold = int(params.max_hold_bars)
        span = max(max_depth - entry_thr, 1e-9)

        for i in range(n):
            c = close[i]
            d = dd[i]
            a = atr[i]
            v = rv[i]

            if not in_position:
                if not np.isnan(d) and d > -release_thr:
                    re_entry_armed = True

            if in_position:
                bars_held += 1
                stop_level = entry_price - stop_mult * entry_atr
                if (not np.isnan(c) and c <= stop_level) or bars_held >= max_hold:
                    signal[i] = 0
                    in_position = False
                    re_entry_armed = False
                    bars_held = 0
                    current_size = 1.0
                else:
                    signal[i] = 1
            else:
                if (
                    re_entry_armed
                    and not np.isnan(d)
                    and not np.isnan(a)
                    and not np.isnan(v)
                    and d <= -entry_thr
                    and v > 1e-6
                    and a > 1e-9
                ):
                    depth = -d
                    depth_clipped = min(max(depth, entry_thr), max_depth)
                    depth_factor = 1.0 + (depth_clipped - entry_thr) / span * (max_size_mult - 1.0)
                    vol_scale = target_vol / max(v, 1e-6)
                    raw_size = vol_scale * depth_factor
                    if raw_size > max_size_mult:
                        raw_size = max_size_mult
                    if raw_size > 1e-6:
                        signal[i] = 1
                        in_position = True
                        entry_price = c
                        entry_atr = a
                        bars_held = 0
                        current_size = raw_size
                        re_entry_armed = False

            size[i] = current_size if current_size > 1e-6 else 1.0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(size, index=data.index).shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
