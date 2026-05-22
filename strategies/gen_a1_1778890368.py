from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    closure_min_days: int = 3
    gap_threshold: float = 0.003
    profit_target: float = 0.015
    time_stop_bars: int = 2
    vol_window: int = 20


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778890368"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(params.vol_window) + 2

    def indicators(self, data, params):
        close = data["close"]
        open_ = data["open"]
        prior_close = close.shift(1)

        gap = open_ / prior_close - 1.0
        ret_co = close / open_ - 1.0

        idx = data.index.to_series()
        closure_days = idx.diff().dt.days.astype("float64")
        closure_days = closure_days.fillna(1.0)
        is_long_closure = (closure_days >= float(params.closure_min_days)).astype(float)

        win = int(params.vol_window)
        vol = close.pct_change().rolling(win, min_periods=win).std()

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["ret_co"] = ret_co
        out["closure_days"] = closure_days
        out["is_long_closure"] = is_long_closure
        out["vol"] = vol
        return out

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        gap = indicators["gap"].to_numpy(dtype=float)
        ret_co = indicators["ret_co"].to_numpy(dtype=float)
        is_long_closure = indicators["is_long_closure"].to_numpy(dtype=float)
        vol = indicators["vol"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        gap_thr = float(params.gap_threshold)
        pt = float(params.profit_target)
        ts = max(1, int(params.time_stop_bars))

        in_pos = 0
        entry_price = 0.0
        bars_held = 0
        cur_size = 1.0

        for i in range(n):
            if in_pos != 0:
                bars_held += 1
                pnl = in_pos * (close[i] / entry_price - 1.0)
                if pnl >= pt or bars_held >= ts:
                    signal[i] = 0
                    in_pos = 0
                    entry_price = 0.0
                    bars_held = 0
                    cur_size = 1.0
                else:
                    signal[i] = in_pos
                    size[i] = cur_size
                continue

            j = i - 1
            if j < 1:
                continue
            if is_long_closure[j] < 0.5:
                continue

            g = gap[j]
            v = vol[j]
            fade_j = ret_co[j]
            if not np.isfinite(g) or not np.isfinite(v) or v <= 0.0:
                continue
            if not np.isfinite(fade_j):
                continue
            if abs(g) < gap_thr:
                continue
            if close[j] <= 0.0:
                continue

            rev_dir = -1 if g > 0.0 else 1
            cont_i = close[i] / close[j] - 1.0

            if rev_dir < 0:
                confirmed = (fade_j < 0.0) and (cont_i < 0.0)
            else:
                confirmed = (fade_j > 0.0) and (cont_i > 0.0)

            if confirmed:
                conviction = abs(g) / (2.0 * gap_thr)
                cur_size = float(min(max(conviction, 0.5), 2.0))
                signal[i] = rev_dir
                size[i] = cur_size
                in_pos = rev_dir
                entry_price = close[i]
                bars_held = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
