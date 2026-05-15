from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_period: int = 50
    price_streak_min: int = 5
    slope_streak_min: int = 3
    atr_period: int = 14
    breakeven_pct: float = 0.03
    init_stop_atr_mult: float = 2.0
    trail_atr_mult: float = 3.0


def _streak(cond: pd.Series) -> pd.Series:
    c = cond.fillna(False).astype(bool)
    grp = (~c).cumsum()
    return c.astype(int).groupby(grp).cumsum()


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_a1_1778885477'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.ma_period + 1, params.atr_period + 1) + 5)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data['close']
        high = data['high']
        low = data['low']

        ma = close.rolling(params.ma_period, min_periods=params.ma_period).mean()
        ma_diff = ma.diff()

        above = close > ma
        below = close < ma
        rising = ma_diff > 0
        falling = ma_diff < 0

        price_up_streak = _streak(above)
        price_dn_streak = _streak(below)
        slope_up_streak = _streak(rising)
        slope_dn_streak = _streak(falling)

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        out = pd.DataFrame(index=data.index)
        out['ma'] = ma
        out['price_up_streak'] = price_up_streak.astype(float)
        out['price_dn_streak'] = price_dn_streak.astype(float)
        out['slope_up_streak'] = slope_up_streak.astype(float)
        out['slope_dn_streak'] = slope_dn_streak.astype(float)
        out['atr'] = atr
        return out

    def generate_signals(self, data, indicators, ctx, params):
        idx = data.index
        n = len(idx)

        high = data['high'].to_numpy(dtype=float)
        low = data['low'].to_numpy(dtype=float)
        close = data['close'].to_numpy(dtype=float)

        pu = indicators['price_up_streak'].to_numpy(dtype=float)
        pdn = indicators['price_dn_streak'].to_numpy(dtype=float)
        su = indicators['slope_up_streak'].to_numpy(dtype=float)
        sdn = indicators['slope_dn_streak'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)

        sig = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=float)

        pmin = params.price_streak_min
        smin = params.slope_streak_min
        be = params.breakeven_pct
        init_mult = params.init_stop_atr_mult
        trail_mult = params.trail_atr_mult

        position = 0
        entry_price = 0.0
        stop = 0.0
        armed = False
        size_val = 1.0

        for i in range(n):
            a = atr[i]
            valid = np.isfinite(a) and a > 0.0

            if position == 0:
                if not valid:
                    continue
                long_ok = (pu[i] >= pmin) and (su[i] >= smin)
                short_ok = (pdn[i] >= pmin) and (sdn[i] >= smin)
                if long_ok:
                    position = 1
                    entry_price = close[i]
                    stop = entry_price - init_mult * a
                    armed = False
                    excess = (pu[i] + su[i]) - (pmin + smin)
                    size_val = float(min(1.4, max(0.5, 0.7 + 0.08 * excess)))
                    sig[i] = 1
                    size[i] = size_val
                elif short_ok:
                    position = -1
                    entry_price = close[i]
                    stop = entry_price + init_mult * a
                    armed = False
                    excess = (pdn[i] + sdn[i]) - (pmin + smin)
                    size_val = float(min(1.4, max(0.5, 0.7 + 0.08 * excess)))
                    sig[i] = -1
                    size[i] = size_val
                continue

            if position == 1:
                if not armed and high[i] >= entry_price * (1.0 + be):
                    armed = True
                    if entry_price > stop:
                        stop = entry_price
                if armed and valid:
                    trail = close[i] - trail_mult * a
                    if trail > stop:
                        stop = trail
                if low[i] <= stop:
                    position = 0
                    sig[i] = 0
                    size[i] = 1.0
                else:
                    sig[i] = 1
                    size[i] = size_val
            else:
                if not armed and low[i] <= entry_price * (1.0 - be):
                    armed = True
                    if entry_price < stop:
                        stop = entry_price
                if armed and valid:
                    trail = close[i] + trail_mult * a
                    if trail < stop:
                        stop = trail
                if high[i] >= stop:
                    position = 0
                    sig[i] = 0
                    size[i] = 1.0
                else:
                    sig[i] = -1
                    size[i] = size_val

        df = pd.DataFrame(index=idx)
        df['signal'] = sig
        df['size'] = size
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
