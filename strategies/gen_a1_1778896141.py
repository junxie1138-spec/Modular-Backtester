from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


def _consecutive_streak(mask: pd.Series) -> pd.Series:
    """Length of the run of True values ending at each bar (0 where False)."""
    b = mask.fillna(False).astype(int)
    grp = (b != b.shift()).cumsum()
    cnt = b.groupby(grp).cumcount() + 1
    return cnt.where(b.astype(bool), 0).astype(int)


@dataclass(slots=True)
class GapBodyDivergenceParams:
    streak_gap: int = 3
    streak_body: int = 2
    atr_len: int = 14
    k_init: float = 2.0
    k_trail: float = 1.5
    breakeven_pct: float = 0.01
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[GapBodyDivergenceParams]):
    strategy_id = "gen_a1_1778896141"

    @classmethod
    def params_type(cls):
        return GapBodyDivergenceParams

    @staticmethod
    def warmup_bars(params: GapBodyDivergenceParams) -> int:
        return int(params.atr_len) + 5

    def indicators(self, data: pd.DataFrame, params: GapBodyDivergenceParams) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        gap = open_ - prev_close

        up_gap = gap > 0.0
        down_gap = gap < 0.0
        green_body = close > open_
        red_body = close < open_

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(int(params.atr_len), min_periods=int(params.atr_len)).mean()

        out = pd.DataFrame(index=data.index)
        out["streak_up_gap"] = _consecutive_streak(up_gap)
        out["streak_down_gap"] = _consecutive_streak(down_gap)
        out["streak_green"] = _consecutive_streak(green_body)
        out["streak_red"] = _consecutive_streak(red_body)
        out["atr"] = atr
        return out

    def generate_signals(self, data, indicators, ctx, params):
        idx = data.index
        n = len(data)

        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)

        su_gap = indicators["streak_up_gap"].to_numpy()
        sd_gap = indicators["streak_down_gap"].to_numpy()
        sg = indicators["streak_green"].to_numpy()
        sr = indicators["streak_red"].to_numpy()
        atr = indicators["atr"].to_numpy(dtype=float)

        ng = max(1, int(params.streak_gap))
        nb = max(1, int(params.streak_body))
        k_init = float(params.k_init)
        k_trail = float(params.k_trail)
        be = float(params.breakeven_pct)
        max_hold = max(1, int(params.max_hold))

        pos = np.zeros(n, dtype=int)
        state = 0
        entry_price = 0.0
        entry_idx = 0
        stop = 0.0
        breakeven = False
        extreme = 0.0

        for i in range(n):
            a = atr[i]
            if state == 0:
                if not np.isfinite(a):
                    continue
                long_sig = (sd_gap[i] >= ng) and (sg[i] >= nb)
                short_sig = (su_gap[i] >= ng) and (sr[i] >= nb)
                if long_sig and not short_sig:
                    state = 1
                    entry_price = close[i]
                    entry_idx = i
                    stop = entry_price - k_init * a
                    extreme = high[i]
                    breakeven = False
                    pos[i] = 1
                elif short_sig and not long_sig:
                    state = -1
                    entry_price = close[i]
                    entry_idx = i
                    stop = entry_price + k_init * a
                    extreme = low[i]
                    breakeven = False
                    pos[i] = -1
            elif state == 1:
                extreme = max(extreme, high[i])
                if not breakeven and high[i] >= entry_price * (1.0 + be):
                    breakeven = True
                    stop = max(stop, entry_price)
                if breakeven and np.isfinite(a):
                    stop = max(stop, extreme - k_trail * a)
                if (low[i] <= stop) or ((i - entry_idx) >= max_hold):
                    state = 0
                    pos[i] = 0
                else:
                    pos[i] = 1
            else:
                extreme = min(extreme, low[i])
                if not breakeven and low[i] <= entry_price * (1.0 - be):
                    breakeven = True
                    stop = min(stop, entry_price)
                if breakeven and np.isfinite(a):
                    stop = min(stop, extreme + k_trail * a)
                if (high[i] >= stop) or ((i - entry_idx) >= max_hold):
                    state = 0
                    pos[i] = 0
                else:
                    pos[i] = -1

        df = pd.DataFrame(index=idx)
        df["signal"] = pos
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
