from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_window: int = 15
    breakeven_pct: float = 0.03


class GeneratedStrategy(BaseStrategy[Params]):
    """Drawdown-recovery via rate-of-change acceleration inflection.

    Treats a drawdown as a filling queue. Entry fires only when the
    curvature of momentum (the acceleration of rate-of-change) crosses
    from non-positive to positive while price is still in a drawdown -
    i.e. the moment the queue stops overflowing. Exit is breakeven-then-trail.
    """

    strategy_id = "gen_a1_1778883838"

    _DD_WINDOW = 252
    _DD_THRESH = -0.05
    _ATR_WINDOW = 14
    _SMOOTH_SPAN = 5
    _K_TRAIL = 3.0
    _MAX_HOLD = 20

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(self._DD_WINDOW, int(params.roc_window) + self._SMOOTH_SPAN + 1) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        roll_max = close.rolling(self._DD_WINDOW, min_periods=1).max()
        dd = close / roll_max - 1.0

        roc = close.pct_change(int(params.roc_window))
        roc_smooth = roc.ewm(span=self._SMOOTH_SPAN, adjust=False).mean()
        roc_accel = roc_smooth.diff()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self._ATR_WINDOW, min_periods=1).mean()

        out = pd.DataFrame(index=data.index)
        out["dd"] = dd
        out["roc_accel"] = roc_accel
        out["atr"] = atr
        return out

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        dd = indicators["dd"].to_numpy(dtype=float)
        accel = indicators["roc_accel"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        warmup = self.warmup_bars(params)
        bp = float(params.breakeven_pct)

        signal = np.zeros(n, dtype=int)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        breakeven_done = False
        bars_held = 0

        for i in range(n):
            if not in_pos:
                if i < warmup or i < 1:
                    continue
                ok = (
                    not np.isnan(dd[i])
                    and not np.isnan(accel[i])
                    and not np.isnan(accel[i - 1])
                    and dd[i] < self._DD_THRESH
                    and accel[i] > 0.0
                    and accel[i - 1] <= 0.0
                )
                if ok:
                    in_pos = True
                    entry_price = close[i]
                    e_atr = atr[i] if not np.isnan(atr[i]) else 0.0
                    stop = entry_price - self._K_TRAIL * e_atr
                    breakeven_done = False
                    bars_held = 0
                    signal[i] = 1
            else:
                bars_held += 1
                if not breakeven_done and high[i] >= entry_price * (1.0 + bp):
                    stop = max(stop, entry_price)
                    breakeven_done = True
                if breakeven_done and not np.isnan(atr[i]):
                    stop = max(stop, high[i] - self._K_TRAIL * atr[i])
                if low[i] <= stop or bars_held >= self._MAX_HOLD:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
