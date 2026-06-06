from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapShockwaveParams:
    gap_vol_window: int = 20
    shock_threshold: float = 1.6
    drift_window: int = 5
    drift_gate: float = 0.55
    ma_window: int = 200
    atr_window: int = 14
    vol_window: int = 20
    target_vol: float = 0.15
    breakeven_pct: float = 0.015
    trail_atr_mult: float = 2.5
    max_hold: int = 2
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[GapShockwaveParams]):
    strategy_id = "gen_a1_1778903700"

    @classmethod
    def params_type(cls) -> type[GapShockwaveParams]:
        return GapShockwaveParams

    @staticmethod
    def warmup_bars(params: GapShockwaveParams) -> int:
        gap_chain = params.gap_vol_window + params.drift_window + 1
        return int(max(params.ma_window, gap_chain,
                       params.atr_window + 1, params.vol_window + 1)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapShockwaveParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]
        prev_close = close.shift(1)

        ind = pd.DataFrame(index=data.index)

        # overnight gap as a fraction of prior close
        gap = open_ / prev_close - 1.0
        gap_std = gap.rolling(params.gap_vol_window).std()
        gap_z = gap / gap_std.replace(0.0, np.nan)
        ind["gap_z"] = gap_z
        # propagation front: net drift of recent normalized gaps
        ind["gap_drift"] = gap_z.rolling(params.drift_window).mean()

        # 200-day regime filter
        ind["ma"] = close.rolling(params.ma_window).mean()

        # average true range
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window).mean()

        # volatility-target position size
        ret = close.pct_change()
        rv_ann = ret.rolling(params.vol_window).std() * np.sqrt(252.0)
        size = params.target_vol / rv_ann.replace(0.0, np.nan)
        ind["vt_size"] = size.clip(lower=0.2, upper=params.max_size)

        return ind

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: GapShockwaveParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)

        gap_z = indicators["gap_z"].to_numpy(dtype=float)
        gap_drift = indicators["gap_drift"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        vt_size = indicators["vt_size"].to_numpy(dtype=float)

        n = len(close)
        position = np.zeros(n, dtype=int)
        size_arr = np.ones(n, dtype=float)

        in_pos = 0
        entry_price = 0.0
        stop = 0.0
        bars_held = 0
        be_armed = False
        held_size = 1.0

        st = params.shock_threshold
        dg = params.drift_gate
        be = params.breakeven_pct
        k = params.trail_atr_mult
        max_hold = params.max_hold

        for i in range(n):
            valid = (np.isfinite(gap_z[i]) and np.isfinite(gap_drift[i])
                     and np.isfinite(ma[i]) and np.isfinite(atr[i])
                     and np.isfinite(close[i]))

            if in_pos == 0:
                if valid:
                    regime_up = close[i] > ma[i]
                    regime_dn = close[i] < ma[i]
                    # fade an isolated counter-trend shock;
                    # skip a propagating same-direction shockwave
                    long_entry = (gap_z[i] < -st and gap_drift[i] > -dg
                                  and regime_up)
                    short_entry = (gap_z[i] > st and gap_drift[i] < dg
                                   and regime_dn)
                    sz = vt_size[i] if np.isfinite(vt_size[i]) else 1.0
                    if long_entry:
                        in_pos = 1
                        entry_price = close[i]
                        stop = entry_price - k * atr[i]
                        be_armed = False
                        bars_held = 0
                        held_size = sz
                        position[i] = 1
                        size_arr[i] = sz
                    elif short_entry:
                        in_pos = -1
                        entry_price = close[i]
                        stop = entry_price + k * atr[i]
                        be_armed = False
                        bars_held = 0
                        held_size = sz
                        position[i] = -1
                        size_arr[i] = sz
                continue

            # managing an open position
            bars_held += 1
            exit_now = False
            a = atr[i] if np.isfinite(atr[i]) else 0.0

            if in_pos == 1:
                # breakeven: lock stop at entry once +be reached
                if not be_armed and high[i] >= entry_price * (1.0 + be):
                    be_armed = True
                    stop = max(stop, entry_price)
                # trail by k*ATR, stop only moves up
                if be_armed:
                    stop = max(stop, close[i] - k * a)
                if low[i] <= stop or bars_held >= max_hold:
                    exit_now = True
            else:
                if not be_armed and low[i] <= entry_price * (1.0 - be):
                    be_armed = True
                    stop = min(stop, entry_price)
                if be_armed:
                    stop = min(stop, close[i] + k * a)
                if high[i] >= stop or bars_held >= max_hold:
                    exit_now = True

            if exit_now:
                position[i] = 0
                size_arr[i] = 1.0
                in_pos = 0
            else:
                position[i] = in_pos
                size_arr[i] = held_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = position
        df["size"] = size_arr
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
