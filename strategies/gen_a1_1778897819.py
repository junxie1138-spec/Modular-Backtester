from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendInfectionParams:
    vol_win: int = 20
    infect_c: float = 0.6
    window_w: int = 15
    slope_win: int = 3
    entry_thr: float = 0.20
    sat_cap: float = 0.85
    ma_win: int = 200
    atr_win: int = 14
    init_stop_mult: float = 2.0
    breakeven_pct: float = 0.02
    trail_k: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[TrendInfectionParams]):
    strategy_id = "gen_a1_1778897819"

    @classmethod
    def params_type(cls):
        return TrendInfectionParams

    @staticmethod
    def warmup_bars(params: TrendInfectionParams) -> int:
        return int(max(params.ma_win,
                       params.vol_win + params.window_w + params.slope_win) + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: TrendInfectionParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()
        # volatility primitive: rolling std of returns
        sigma = ret.rolling(params.vol_win).std()

        # a bar is "infected" when its return is large relative to volatility
        up_inf = (ret > params.infect_c * sigma).astype(float)
        down_inf = (ret < -params.infect_c * sigma).astype(float)

        # infection prevalence: fraction of recent bars infected in each direction
        iu = up_inf.rolling(params.window_w).mean()
        idn = down_inf.rolling(params.window_w).mean()
        net = iu - idn
        net_slope = net.diff(params.slope_win)

        # 200-day regime filter (hard twist)
        ma = close.rolling(params.ma_win).mean()

        # ATR for breakeven-then-trail exit
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_win).mean()

        # entry: net prevalence above threshold, still rising, not saturated, regime-aligned
        raw_long = (
            (close > ma)
            & (net > params.entry_thr)
            & (net_slope > 0.0)
            & (iu < params.sat_cap)
        )
        raw_short = (
            (close < ma)
            & (net < -params.entry_thr)
            & (net_slope < 0.0)
            & (idn < params.sat_cap)
        )

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["ma"] = ma
        out["net"] = net
        out["iu"] = iu
        out["idn"] = idn
        out["raw_long"] = raw_long.astype(float)
        out["raw_short"] = raw_short.astype(float)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: TrendInfectionParams) -> SignalFrame:
        n = len(data)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw_long = indicators["raw_long"].to_numpy(dtype=float)
        raw_short = indicators["raw_short"].to_numpy(dtype=float)

        pos = np.zeros(n, dtype=int)

        position = 0
        entry_price = 0.0
        stop = 0.0
        armed = False
        hold = 0

        be = params.breakeven_pct
        k = params.trail_k
        ism = params.init_stop_mult
        max_hold = params.max_hold

        for i in range(n):
            a = atr[i]

            if position == 0:
                # only enter when ATR is valid (handles warmup NaN)
                if np.isfinite(a) and a > 0.0:
                    if raw_long[i] > 0.5:
                        position = 1
                        entry_price = close[i]
                        stop = entry_price - ism * a
                        armed = False
                        hold = 0
                    elif raw_short[i] > 0.5:
                        position = -1
                        entry_price = close[i]
                        stop = entry_price + ism * a
                        armed = False
                        hold = 0
                pos[i] = position
                continue

            hold += 1
            exit_now = False

            if position == 1:
                # breakeven: once +be reached, lift stop to entry (never down)
                if not armed and high[i] >= entry_price * (1.0 + be):
                    if entry_price > stop:
                        stop = entry_price
                    armed = True
                # trail by k*ATR after breakeven; stop only ratchets up
                if armed and np.isfinite(a):
                    trail = close[i] - k * a
                    if trail > stop:
                        stop = trail
                if low[i] <= stop:
                    exit_now = True
            else:
                # short mirror: stop only ratchets down
                if not armed and low[i] <= entry_price * (1.0 - be):
                    if entry_price < stop:
                        stop = entry_price
                    armed = True
                if armed and np.isfinite(a):
                    trail = close[i] + k * a
                    if trail < stop:
                        stop = trail
                if high[i] >= stop:
                    exit_now = True

            if hold >= max_hold:
                exit_now = True

            if exit_now:
                position = 0
            pos[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["size"] = 1.0
        # MANDATORY one-bar shift: decide on bar N close, fill on N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
