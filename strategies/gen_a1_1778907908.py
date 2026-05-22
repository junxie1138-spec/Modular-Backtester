from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalVolParams:
    vol_window: int = 21
    ma_window: int = 200
    hold_bars: int = 7
    season_band: float = 0.05
    vol_target: float = 0.012
    size_min: float = 0.5
    size_max: float = 2.0


class GeneratedStrategy(BaseStrategy[SeasonalVolParams]):
    strategy_id = "gen_a1_1778907908"

    @classmethod
    def params_type(cls):
        return SeasonalVolParams

    def warmup_bars(self, params):
        return int(max(params.ma_window, params.vol_window)) + 2

    def indicators(self, data, params):
        p = params
        close = data["close"]
        returns = close.pct_change()
        rv = returns.rolling(p.vol_window).std()
        ma = close.rolling(p.ma_window).mean()
        month = pd.Series(data.index.month, index=data.index)

        # Per-calendar-month expanding mean of realized vol, excluding the
        # current bar (shift(1)) so the seasonal estimate has no lookahead.
        season = pd.DataFrame(index=data.index)
        for m in range(1, 13):
            s = rv.where(month == m)
            season[m] = s.expanding().mean().shift(1).ffill()

        season_curr = pd.Series(np.nan, index=data.index)
        season_prev = pd.Series(np.nan, index=data.index)
        for m in range(1, 13):
            mask = (month == m).to_numpy()
            pm = 12 if m == 1 else m - 1
            season_curr.loc[mask] = season[m].loc[mask]
            season_prev.loc[mask] = season[pm].loc[mask]

        prev_safe = season_prev.replace(0.0, np.nan)
        ratio = (season_curr / prev_safe).replace([np.inf, -np.inf], np.nan)

        out = pd.DataFrame(index=data.index)
        out["rv"] = rv
        out["ma"] = ma
        out["season_curr"] = season_curr
        out["season_prev"] = season_prev
        out["ratio"] = ratio
        out["month_change"] = (month != month.shift(1)).astype(int)
        out["above_ma"] = (close > ma).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        p = params
        n = len(data)

        rv = indicators["rv"].to_numpy(dtype=float)
        ratio = indicators["ratio"].to_numpy(dtype=float)
        mc = indicators["month_change"].to_numpy(dtype=float)
        above = indicators["above_ma"].to_numpy(dtype=float)

        # Seasonal vol gradient: stepping up -> short, stepping down -> long.
        # NaN ratios (warmup / unseen prior month) compare False -> no trade.
        raw_dir = np.zeros(n, dtype=int)
        step_up = ratio > (1.0 + p.season_band)
        step_dn = ratio < (1.0 - p.season_band)
        raw_dir[step_up] = -1
        raw_dir[step_dn] = 1

        # 200-day MA regime filter: longs only above the MA, shorts only below.
        gated = np.zeros(n, dtype=int)
        gated[(raw_dir == 1) & (above == 1.0)] = 1
        gated[(raw_dir == -1) & (above == 0.0)] = -1

        # Inverse-volatility position sizing (NaN-safe).
        rv_safe = np.where(np.isnan(rv) | (rv <= 0.0), np.nan, rv)
        size_raw = np.clip(p.vol_target / rv_safe, p.size_min, p.size_max)
        size_raw = np.where(np.isnan(size_raw), 1.0, size_raw)

        # Fixed-bar exit: enter at a month boundary, hold exactly hold_bars,
        # then flat. No signal-based exit; refractory until the hold completes.
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)
        hold = max(1, int(p.hold_bars))
        i = 0
        while i < n:
            if mc[i] == 1.0 and gated[i] != 0:
                d = int(gated[i])
                sz = float(size_raw[i])
                end = min(i + hold, n)
                for j in range(i, end):
                    signal[j] = d
                    size[j] = sz
                i = end
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
