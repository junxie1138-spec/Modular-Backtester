from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalWeekdayParams:
    season_window: int = 20
    min_obs: int = 8
    t_enter: float = 1.0
    t_ref: float = 2.0
    profit_target: float = 0.02
    time_stop: int = 4
    size_min: float = 0.3
    size_max: float = 1.0


class GeneratedStrategy(BaseStrategy[SeasonalWeekdayParams]):
    strategy_id = "gen_a1_1778887853"

    @classmethod
    def params_type(cls):
        return SeasonalWeekdayParams

    def warmup_bars(self, params):
        # season_window counts weekday-occurrences (~5 calendar bars each);
        # +6 covers pct_change and the within-group shift(1).
        return int(params.season_window * 5 + 6)

    def indicators(self, data, params):
        p = params
        close = data["close"]
        ret = close.pct_change()

        wd = pd.Series(np.asarray(data.index.dayofweek), index=data.index)
        tmp = pd.DataFrame({"ret": ret, "wd": wd})
        grp = tmp.groupby("wd")["ret"]

        win = int(p.season_window)
        mp = int(min(p.min_obs, win))

        # Rolling stats over each weekday's OWN past occurrences only.
        mean = grp.transform(
            lambda s: s.shift(1).rolling(win, min_periods=mp).mean()
        )
        std = grp.transform(
            lambda s: s.shift(1).rolling(win, min_periods=mp).std()
        )

        std_safe = std.replace(0.0, np.nan)
        tstat = mean / std_safe * np.sqrt(float(win))

        size_raw = (tstat.abs() / float(p.t_ref)).clip(
            lower=float(p.size_min), upper=float(p.size_max)
        )
        size_raw = size_raw.fillna(float(p.size_min))

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["seasonal_mean"] = mean
        out["seasonal_std"] = std
        out["tstat"] = tstat
        out["size_raw"] = size_raw
        return out

    def generate_signals(self, data, indicators, ctx, params):
        p = params
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        tstat = indicators["tstat"].to_numpy(dtype=float)
        size_raw = indicators["size_raw"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        t_enter = float(p.t_enter)
        profit_target = float(p.profit_target)
        time_stop = int(p.time_stop)

        pos = 0
        entry_price = 0.0
        entry_size = 1.0
        bars_held = 0

        for i in range(n):
            t = tstat[i]
            if pos == 0:
                # Hysteresis-armed one-shot entry: only fire on a clear edge.
                if np.isfinite(t) and abs(t) >= t_enter:
                    pos = 1 if t > 0.0 else -1
                    entry_price = close[i]
                    s = size_raw[i]
                    if not np.isfinite(s) or s <= 0.0:
                        s = float(p.size_min)
                    entry_size = s
                    bars_held = 0
                    signal[i] = pos
                    size[i] = entry_size
                else:
                    signal[i] = 0
            else:
                bars_held += 1
                pnl = (close[i] / entry_price - 1.0) * pos
                if pnl >= profit_target or bars_held >= time_stop:
                    pos = 0
                    signal[i] = 0
                else:
                    signal[i] = pos
                    size[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=float(p.size_min))
        return SignalFrame(data=df, signal_column="signal", size_column="size")
