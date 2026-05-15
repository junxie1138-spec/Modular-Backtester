from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeEchoParams:
    """Two tunable params only (hard twist: <=2).

    window         : lookback for range autocorrelation AND the regime gate.
    profit_target  : fractional gain that triggers the profit-target exit.
    """

    window: int = 60
    profit_target: float = 0.03


class GeneratedStrategy(BaseStrategy[RangeEchoParams]):
    """Range-echo autocorrelation regime gate.

    Mechanism
    ---------
    1. Compute the daily log high-low range, log(high / low).
    2. Compute its rolling lag-1 autocorrelation over `window` bars. This is a
       coherence / signal-to-noise proxy: high positive autocorrelation means
       volatility clustering is organized and directional drift is low-noise.
    3. When that autocorrelation exceeds a fixed threshold, take a position in
       the direction of the trailing 3-bar return (long/short). Otherwise stay
       flat - the gate is the signal-to-noise filter.
    4. Exit on profit-target (+profit_target) OR a fixed 5-bar time-stop,
       whichever fires first.

    All non-tunable knobs (autocorrelation threshold, time-stop length,
    direction lookback) are hard-coded literals to honour the <=2 param twist.
    """

    strategy_id = "gen_a1_1778888631"

    # --- fixed, non-tunable constants -------------------------------------
    _AC_THRESHOLD = 0.15   # coherence gate on range autocorrelation
    _TIME_STOP = 5         # bars; matches 3-5 day holding horizon
    _DIR_LOOKBACK = 3      # bars used to read trailing return direction

    @classmethod
    def params_type(cls):
        return RangeEchoParams

    def warmup_bars(self, params):
        # rolling(window).corr over a shift(1) series -> needs window + 1;
        # pad a little extra for the pct_change direction lookback.
        return int(params.window) + self._DIR_LOOKBACK + 2

    def indicators(self, data, params):
        window = max(int(params.window), 2)

        high = data["high"].astype(float)
        low = data["low"].astype(float).clip(lower=1e-9)
        close = data["close"].astype(float)

        # Daily log high-low range. high >= low >= 0 for OHLC data, so the
        # ratio is >= 1 and the log is finite and non-negative.
        log_range = np.log((high / low).clip(lower=1.0 + 1e-12))

        # Rolling lag-1 autocorrelation of the range series. NaN during warmup
        # and NaN if the range is constant across the window (corr undefined).
        ac = log_range.rolling(window).corr(log_range.shift(1))

        # Trailing directional read.
        ret_dir = close.pct_change(self._DIR_LOOKBACK)

        out = pd.DataFrame(index=data.index)
        out["log_range"] = log_range
        out["ac"] = ac
        out["ret_dir"] = ret_dir
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        ac = indicators["ac"].to_numpy(dtype=float)
        ret_dir = indicators["ret_dir"].to_numpy(dtype=float)
        n = len(close)

        profit_target = float(params.profit_target)
        threshold = self._AC_THRESHOLD
        time_stop = self._TIME_STOP

        # --- raw entry intent: direction only when the gate is open --------
        raw = np.zeros(n, dtype=int)
        for i in range(n):
            a = ac[i]
            r = ret_dir[i]
            if np.isnan(a) or np.isnan(r):
                continue
            if a > threshold and r != 0.0:
                raw[i] = 1 if r > 0.0 else -1

        # --- path-dependent exit: profit-target + time-stop ----------------
        pos = np.zeros(n, dtype=int)
        cur = 0
        entry_price = 0.0
        bars_held = 0
        for i in range(n):
            if cur == 0:
                if raw[i] != 0:
                    cur = raw[i]
                    entry_price = close[i]
                    bars_held = 0
            else:
                bars_held += 1
                if entry_price > 0.0:
                    gain = cur * (close[i] / entry_price - 1.0)
                else:
                    gain = 0.0
                if gain >= profit_target or bars_held >= time_stop:
                    cur = 0
            pos[i] = cur

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
