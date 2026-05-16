from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


_ATR_WINDOW = 14


@dataclass(slots=True)
class ShockwaveParams:
    window: int = 20
    atr_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    """Range-band shockwave: trade the sign of net midpoint displacement
    normalized by total range churned; exit on a fixed ATR volatility stop."""

    strategy_id = "gen_a1_1778910941"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        win = max(int(params.window), 2)
        return int(max(win + 1, _ATR_WINDOW + 1))

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # --- range-band propagation: traffic-shockwave front speed ---
        # net displacement of the [low, high] band's midpoint divided by the
        # total range "churned" over the window. ~+1 => clean up-front,
        # ~-1 => clean down-front, ~0 => congested chop.
        midpoint = (high + low) / 2.0
        step = midpoint.diff()
        rng = (high - low).abs()

        win = max(int(params.window), 2)
        net_step = step.rolling(win).sum()
        churn = rng.rolling(win).sum()
        churn = churn.where(churn > 0.0)  # avoid divide-by-zero -> NaN
        propagation = net_step / churn

        raw_dir = np.sign(propagation.to_numpy())
        raw_dir = np.where(np.isnan(raw_dir), 0.0, raw_dir)

        # --- ATR for the fixed volatility stop ---
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(_ATR_WINDOW).mean()

        out = pd.DataFrame(index=data.index)
        out["propagation"] = propagation
        out["raw_dir"] = raw_dir
        out["atr"] = atr
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        raw = indicators["raw_dir"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        k = float(params.atr_mult)
        signal = np.zeros(n, dtype=int)

        position = 0      # -1 / 0 / 1
        entry_price = 0.0
        entry_atr = 0.0   # ATR frozen at entry -> fixed (non-trailing) stop
        stopped_dir = 0   # direction last stopped out of; blocks re-entry

        for i in range(n):
            c = close[i]
            r = raw[i]
            d = int(r) if not np.isnan(r) else 0
            a = atr[i]
            atr_ok = (not np.isnan(a)) and a > 0.0

            # 1) fixed ATR volatility stop on the open position
            if position == 1:
                if c < entry_price - k * entry_atr:
                    position = 0
                    stopped_dir = 1
            elif position == -1:
                if c > entry_price + k * entry_atr:
                    position = 0
                    stopped_dir = -1

            # release the refractory block once the front flips away
            if stopped_dir != 0 and d != stopped_dir:
                stopped_dir = 0

            # 2) entries and reversals (need a valid ATR to set the stop)
            if d != 0 and atr_ok:
                if position == 0:
                    if d != stopped_dir:
                        position = d
                        entry_price = c
                        entry_atr = a
                elif d != position:
                    position = d
                    entry_price = c
                    entry_atr = a

            signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
