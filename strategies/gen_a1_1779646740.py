from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    short_horizon: int = 5
    mid_horizon: int = 10
    long_horizon: int = 20
    percentile_lookback: int = 252
    percentile_threshold: float = 0.85
    atr_window: int = 14
    breakeven_pct: float = 0.03
    trail_k: float = 2.5
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy):
    strategy_id = "gen_a1_1779646740"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(max(params.long_horizon, params.percentile_lookback, params.atr_window)) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        # Close-to-close cumulative returns at three horizons
        cum_s = close.pct_change(periods=int(params.short_horizon))
        cum_m = close.pct_change(periods=int(params.mid_horizon))
        cum_l = close.pct_change(periods=int(params.long_horizon))

        # Rolling percentile rank of each cumulative-return series within its own long lookback
        win = int(params.percentile_lookback)
        min_p = max(int(win // 2), 2)
        rank_s = cum_s.rolling(window=win, min_periods=min_p).rank(pct=True)
        rank_m = cum_m.rolling(window=win, min_periods=min_p).rank(pct=True)
        rank_l = cum_l.rolling(window=win, min_periods=min_p).rank(pct=True)

        # ATR for trailing-stop distance
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=int(params.atr_window), min_periods=int(params.atr_window)).mean()

        out = pd.DataFrame({
            "cum_s": cum_s,
            "cum_m": cum_m,
            "cum_l": cum_l,
            "rank_s": rank_s,
            "rank_m": rank_m,
            "rank_l": rank_l,
            "atr": atr,
        }, index=data.index)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        rank_s = indicators["rank_s"].to_numpy(dtype=float)
        rank_m = indicators["rank_m"].to_numpy(dtype=float)
        rank_l = indicators["rank_l"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        thr = float(params.percentile_threshold)
        be_pct = float(params.breakeven_pct)
        k = float(params.trail_k)
        max_hold = int(params.max_hold_bars)

        valid = (
            ~np.isnan(rank_s)
            & ~np.isnan(rank_m)
            & ~np.isnan(rank_l)
            & ~np.isnan(atr)
        )
        align = valid & (rank_s >= thr) & (rank_m >= thr) & (rank_l >= thr)

        raw_signal = np.zeros(n, dtype=np.int64)
        in_pos = False
        entry_price = 0.0
        stop = -np.inf
        bars_held = 0
        breakeven_armed = False

        for i in range(n):
            price = close[i]
            if np.isnan(price):
                if in_pos:
                    raw_signal[i] = 1
                continue

            if not in_pos:
                if align[i]:
                    in_pos = True
                    entry_price = price
                    stop = -np.inf
                    bars_held = 0
                    breakeven_armed = False
                    raw_signal[i] = 1
                else:
                    raw_signal[i] = 0
            else:
                bars_held += 1

                if (not breakeven_armed) and price >= entry_price * (1.0 + be_pct):
                    breakeven_armed = True
                    stop = entry_price

                if breakeven_armed:
                    a = atr[i]
                    if not np.isnan(a):
                        candidate = price - k * a
                        if candidate > stop:
                            stop = candidate

                hit_stop = breakeven_armed and price <= stop
                time_exit = bars_held >= max_hold

                if hit_stop or time_exit:
                    in_pos = False
                    entry_price = 0.0
                    stop = -np.inf
                    breakeven_armed = False
                    bars_held = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1

        df = pd.DataFrame(index=data.index)
        sig = pd.Series(raw_signal, index=data.index)
        df["signal"] = sig.shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
