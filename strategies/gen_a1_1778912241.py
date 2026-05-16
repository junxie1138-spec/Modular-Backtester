from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RankSpreadParams:
    short_win: int = 21
    long_win: int = 126
    ma_win: int = 200
    atr_win: int = 14
    vol_win: int = 20
    entry_gap: float = 0.35
    short_floor: float = 0.50
    be_trigger: float = 0.04
    init_stop_k: float = 2.0
    trail_k: float = 3.0
    max_hold: int = 25
    spike_k: float = 2.5
    refractory_bars: int = 5


class GeneratedStrategy(BaseStrategy[RankSpreadParams]):
    strategy_id = "gen_a1_1778912241"

    @classmethod
    def params_type(cls):
        return RankSpreadParams

    def warmup_bars(self, params):
        return int(max(params.ma_win, params.long_win, params.atr_win,
                       params.vol_win, params.short_win)) + 2

    @staticmethod
    def _rolling_rank_pct(arr, win):
        n = arr.shape[0]
        out = np.full(n, np.nan, dtype=float)
        win = int(win)
        if win >= 2 and n >= win:
            w = np.lib.stride_tricks.sliding_window_view(arr, win)
            last = arr[win - 1:]
            cnt = (w <= last[:, None]).sum(axis=1)
            out[win - 1:] = cnt.astype(float) / float(win)
        return out

    def indicators(self, data, params):
        p = params
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        c = close.to_numpy(dtype=float)

        short_rank = self._rolling_rank_pct(c, p.short_win)
        long_rank = self._rolling_rank_pct(c, p.long_win)
        rank_gap = short_rank - long_rank

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(max(int(p.atr_win), 2)).mean()

        ma = close.rolling(max(int(p.ma_win), 2)).mean()

        ret = close.pct_change()
        vol = ret.rolling(max(int(p.vol_win), 2)).std()
        spike = (ret.abs() > p.spike_k * vol) & vol.notna()

        out = pd.DataFrame(index=data.index)
        out["short_rank"] = short_rank
        out["long_rank"] = long_rank
        out["rank_gap"] = rank_gap
        out["atr"] = atr
        out["ma"] = ma
        out["spike"] = spike.astype(float)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        p = params
        n = len(data)
        close = data["close"].to_numpy(dtype=float)

        rank_gap = indicators["rank_gap"].to_numpy(dtype=float)
        short_rank = indicators["short_rank"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        spike = indicators["spike"].to_numpy(dtype=float)

        sig = np.zeros(n, dtype=float)
        warm = self.warmup_bars(p)

        position = 0
        entry_price = 0.0
        stop = 0.0
        peak = 0.0
        trough = 0.0
        breakeven = False
        bars_held = 0
        refractory_until = -1

        for i in range(n):
            if not np.isnan(spike[i]) and spike[i] > 0.5:
                refractory_until = i + int(p.refractory_bars)

            a = atr[i]
            px = close[i]
            just_exited = False

            if position == 1:
                bars_held += 1
                if px > peak:
                    peak = px
                if (not breakeven) and px >= entry_price * (1.0 + p.be_trigger):
                    breakeven = True
                    if entry_price > stop:
                        stop = entry_price
                if breakeven and not np.isnan(a):
                    trail = peak - p.trail_k * a
                    if trail > stop:
                        stop = trail
                if px <= stop or bars_held >= int(p.max_hold):
                    position = 0
                    breakeven = False
                    bars_held = 0
                    just_exited = True
            elif position == -1:
                bars_held += 1
                if px < trough:
                    trough = px
                if (not breakeven) and px <= entry_price * (1.0 - p.be_trigger):
                    breakeven = True
                    if entry_price < stop:
                        stop = entry_price
                if breakeven and not np.isnan(a):
                    trail = trough + p.trail_k * a
                    if trail < stop:
                        stop = trail
                if px >= stop or bars_held >= int(p.max_hold):
                    position = 0
                    breakeven = False
                    bars_held = 0
                    just_exited = True

            valid = (
                i >= warm
                and not np.isnan(rank_gap[i])
                and not np.isnan(ma[i])
                and not np.isnan(a)
                and a > 0.0
            )

            if position == 0 and valid and not just_exited and i > refractory_until:
                if (px > ma[i]
                        and rank_gap[i] >= p.entry_gap
                        and short_rank[i] >= p.short_floor):
                    position = 1
                    entry_price = px
                    peak = px
                    stop = px - p.init_stop_k * a
                    breakeven = False
                    bars_held = 0
                elif (px < ma[i]
                        and rank_gap[i] <= -p.entry_gap
                        and short_rank[i] <= (1.0 - p.short_floor)):
                    position = -1
                    entry_price = px
                    trough = px
                    stop = px + p.init_stop_k * a
                    breakeven = False
                    bars_held = 0

            sig[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig.astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
