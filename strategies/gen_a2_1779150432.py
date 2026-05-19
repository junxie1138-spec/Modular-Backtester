from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendAccelParams:
    # Lookback (bars) for the rate-of-change and its acceleration smoothing.
    roc_period: int = 20
    # Fixed volatility-stop distance in multiples of entry-bar ATR.
    atr_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[TrendAccelParams]):
    """Long/short while ROC direction and smoothed ROC-acceleration direction agree.

    The trend is traded only when it is actively strengthening: both the
    rate-of-change and the change-in-rate-of-change must share a sign. A fixed
    (non-trailing) ATR stop set at entry caps each trade; a refractory window
    follows every stop-out to avoid immediate re-entry into chop.
    """

    strategy_id = "gen_a2_1779150432"

    # Non-tunable structural constants (kept off the param space by design).
    _ATR_WINDOW = 14
    _REFRACTORY = 5
    _MAX_HOLD = 20

    @classmethod
    def params_type(cls) -> type[TrendAccelParams]:
        return TrendAccelParams

    @staticmethod
    def warmup_bars(params: TrendAccelParams) -> int:
        p = max(int(params.roc_period), 2)
        # pct_change(p) + diff() + ewm(span=p) + ATR window, with headroom.
        return 2 * p + GeneratedStrategy._ATR_WINDOW + 5

    def indicators(self, data: pd.DataFrame, params: TrendAccelParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        p = max(int(params.roc_period), 2)

        roc = close.pct_change(p)
        accel = roc.diff()
        accel_smooth = accel.ewm(span=p, adjust=False, min_periods=1).mean()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self._ATR_WINDOW, min_periods=self._ATR_WINDOW).mean()

        up = (roc > 0) & (accel_smooth > 0)
        dn = (roc < 0) & (accel_smooth < 0)
        direction = pd.Series(0.0, index=data.index, dtype=float)
        direction[up.fillna(False)] = 1.0
        direction[dn.fillna(False)] = -1.0

        out = pd.DataFrame(index=data.index)
        out["roc"] = roc
        out["accel_smooth"] = accel_smooth
        out["atr"] = atr
        out["dir"] = direction.fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TrendAccelParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        direction = indicators["dir"].to_numpy(dtype=float)
        n = len(close)
        raw = np.zeros(n, dtype=int)

        k = float(params.atr_mult)
        refractory = self._REFRACTORY
        max_hold = self._MAX_HOLD

        position = 0
        entry_price = 0.0
        stop_dist = 0.0
        bars_held = 0
        refractory_until = -1

        for i in range(n):
            if position != 0:
                bars_held += 1
                exit_now = False
                spike = False
                if position == 1 and close[i] <= entry_price - stop_dist:
                    exit_now = True
                    spike = True
                elif position == -1 and close[i] >= entry_price + stop_dist:
                    exit_now = True
                    spike = True
                elif bars_held >= max_hold:
                    exit_now = True
                elif direction[i] == -position:
                    exit_now = True
                if exit_now:
                    if spike:
                        refractory_until = i + refractory
                    elif bars_held >= max_hold:
                        refractory_until = i
                    position = 0
                    bars_held = 0

            if position == 0 and i > refractory_until:
                d = direction[i]
                a = atr[i]
                if d != 0.0 and np.isfinite(a) and a > 0.0:
                    position = int(d)
                    entry_price = close[i]
                    stop_dist = k * a
                    bars_held = 0

            raw[i] = position

        signal = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
