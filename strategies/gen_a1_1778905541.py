from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    season_lookback: int = 26
    z_thresh: float = 1.0
    profit_target: float = 0.02
    time_stop: int = 2
    target_range: float = 0.012


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1778905541"

    @classmethod
    def params_type(cls):
        return GenA1Params

    def warmup_bars(self, params):
        return int(params.season_lookback) * 5 + 5

    def indicators(self, data, params):
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        safe_close = close.replace(0.0, np.nan)
        nrange = tr / safe_close
        ret = close.pct_change()

        wd = pd.Series(data.index.dayofweek, index=data.index)
        L = max(int(params.season_lookback), 2)

        # Per-weekday rolling volatility baseline: each weekday is normalized
        # against its own seasonal history of normalized true range.
        season_mean = nrange.groupby(wd).transform(
            lambda s: s.rolling(L, min_periods=L).mean()
        )
        season_std = nrange.groupby(wd).transform(
            lambda s: s.rolling(L, min_periods=L).std()
        )
        season_std = season_std.replace(0.0, np.nan)

        # Seasonal volatility z-score (elastic band vs plastic yield).
        zvol = (nrange - season_mean) / season_std
        zvol = zvol.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        yld = zvol > float(params.z_thresh)
        ret_up = ret > 0.0
        ret_dn = ret < 0.0

        prev_yld = yld.shift(1, fill_value=False)
        prev_up = ret_up.shift(1, fill_value=False)
        prev_dn = ret_dn.shift(1, fill_value=False)

        # Two-bar confirmation: both bars breach their weekday seasonal
        # volatility norm AND agree in close-to-close direction.
        long_conf = yld & prev_yld & ret_up & prev_up
        short_conf = yld & prev_yld & ret_dn & prev_dn

        entry = pd.Series(0, index=data.index, dtype=int)
        entry = entry.mask(long_conf, 1).mask(short_conf, -1)

        out = pd.DataFrame(index=data.index)
        out["nrange"] = nrange.fillna(0.0)
        out["ret"] = ret.fillna(0.0)
        out["zvol"] = zvol
        out["entry"] = entry.astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        entry = indicators["entry"].to_numpy()
        nrange = indicators["nrange"].to_numpy(dtype=float)
        n = len(close)

        pt = float(params.profit_target)
        ts = max(int(params.time_stop), 1)

        pos = np.zeros(n, dtype=int)
        cur = 0
        entry_price = 0.0
        bars_held = 0

        # Path-dependent exit: profit-target OR time-stop, whichever first.
        for i in range(n):
            if cur != 0:
                bars_held += 1
                ep = entry_price if entry_price > 0.0 else close[i]
                pnl = (close[i] - ep) / ep * cur
                if pnl >= pt or bars_held >= ts:
                    cur = 0
                    entry_price = 0.0
                    bars_held = 0
            if cur == 0 and entry[i] != 0:
                cur = int(entry[i])
                entry_price = close[i]
                bars_held = 0
            pos[i] = cur

        target = float(params.target_range)
        size_arr = np.where(nrange > 1e-9, target / nrange, 1.0)
        size_arr = np.clip(size_arr, 0.4, 1.25)

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = size_arr.astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
