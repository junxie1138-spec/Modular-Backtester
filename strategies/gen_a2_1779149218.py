from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    comp_window: int = 60
    comp_low: float = 0.20
    comp_high: float = 0.50
    pos_window: int = 20
    drift_lag: int = 2
    drift_thresh: float = 0.10
    hold_bars: int = 2
    base_size: float = 1.0


def _rolling_rank(arr: np.ndarray, w: int) -> np.ndarray:
    """Fraction of the trailing window of length w that is <= the last value."""
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if n >= w and w > 0:
        windows = np.lib.stride_tricks.sliding_window_view(arr, w)
        last = windows[:, -1:]
        out[w - 1:] = (windows <= last).mean(axis=1)
    return out


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779149218"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.comp_window,
                       params.pos_window + params.drift_lag + 1)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        tr = tr.fillna(high - low)

        cw = max(int(params.comp_window), 2)
        pw = max(int(params.pos_window), 2)
        dlag = max(int(params.drift_lag), 1)

        comp_rank = pd.Series(_rolling_rank(tr.to_numpy(dtype=float), cw),
                              index=data.index)
        pos_rank = pd.Series(_rolling_rank(close.to_numpy(dtype=float), pw),
                             index=data.index)
        drift = pos_rank - pos_rank.shift(dlag)

        cr = comp_rank.to_numpy()
        n = len(cr)
        armed = np.zeros(n, dtype=bool)
        state = False
        lo = float(params.comp_low)
        hi = float(params.comp_high)
        for i in range(n):
            v = cr[i]
            if np.isnan(v):
                state = False
            elif not state and v < lo:
                state = True
            elif state and v > hi:
                state = False
            armed[i] = state

        out = pd.DataFrame(index=data.index)
        out["tr"] = tr
        out["comp_rank"] = comp_rank
        out["pos_rank"] = pos_rank
        out["drift"] = drift
        out["armed"] = armed
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        idx = data.index
        n = len(idx)

        drift = indicators["drift"]
        armed = indicators["armed"].to_numpy()
        comp_rank = indicators["comp_rank"]

        thr = float(params.drift_thresh)
        long_now = (drift > thr).to_numpy()
        long_prev = (drift.shift(1) > thr).to_numpy()
        short_now = (drift < -thr).to_numpy()
        short_prev = (drift.shift(1) < -thr).to_numpy()

        confirm_long = long_now & long_prev
        confirm_short = short_now & short_prev

        entry = np.zeros(n, dtype=int)
        entry[armed & confirm_long] = 1
        entry[armed & confirm_short] = -1

        hold = max(int(params.hold_bars), 1)
        sig = np.zeros(n, dtype=int)
        pos = 0
        bars_held = 0
        for i in range(n):
            if pos != 0:
                bars_held += 1
                if bars_held >= hold:
                    pos = 0
                    bars_held = 0
                sig[i] = pos
            else:
                if entry[i] != 0:
                    pos = int(entry[i])
                    bars_held = 0
                    sig[i] = pos
                else:
                    sig[i] = 0

        cr = comp_rank.fillna(1.0).to_numpy()
        lo = max(float(params.comp_low), 1e-6)
        depth = np.clip((lo - cr) / lo, 0.0, 1.0)
        size = float(params.base_size) * (0.75 + 0.5 * depth)
        size = np.clip(size, 0.25, 2.0)

        df = pd.DataFrame(index=idx)
        df["signal"] = pd.Series(sig, index=idx).shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(size, index=idx).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
