from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


# Hardcoded mechanics (kept off the tunable surface to satisfy the <=2 param twist).
_ATR_PERIOD = 14
_TREND_PERIOD = 100
_BREAKEVEN_PCT = 0.01   # +1% before the stop is lifted to entry
_MAX_HOLD = 8           # generous time backstop; the trail drives most exits


@dataclass(slots=True)
class AutocorrVolParams:
    ac_window: int = 20
    trail_atr_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[AutocorrVolParams]):
    strategy_id = "gen_a1_1778905708"

    @classmethod
    def params_type(cls):
        return AutocorrVolParams

    def warmup_bars(self, params: AutocorrVolParams) -> int:
        w = int(params.ac_window)
        # ac needs w bars of returns (+1 for pct_change); trend MA needs _TREND_PERIOD.
        return int(max(w + 2, _TREND_PERIOD + 5, _ATR_PERIOD + 2))

    def indicators(self, data: pd.DataFrame, params: AutocorrVolParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        w = int(params.ac_window)

        ret = close.pct_change()
        # Rolling lag-1 autocorrelation coefficient of daily returns.
        ac = ret.rolling(w).corr(ret.shift(1))
        vol_ma = volume.rolling(w).mean()
        trend_ma = close.rolling(_TREND_PERIOD).mean()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(_ATR_PERIOD).mean()

        # Entry candidate: positive-autocorrelation regime AND a volume-confirmed
        # up-bar AND price above its long trend. Any NaN component -> False.
        entry = (
            (ac > 0.0)
            & (close > prev_close)
            & (volume > vol_ma)
            & (close > trend_ma)
        )

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["ac"] = ac
        out["entry"] = entry.astype(float)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AutocorrVolParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].to_numpy(dtype=float)

        k = float(params.trail_atr_mult)

        raw = np.zeros(n, dtype=int)
        position = 0
        entry_price = 0.0
        stop = 0.0
        armed = False
        bars_held = 0

        for i in range(n):
            if position == 0:
                # Open a long on a volume-confirmed up-bar inside the
                # positive-autocorrelation regime; needs a valid ATR for the stop.
                if entry[i] >= 1.0 and np.isfinite(atr[i]) and atr[i] > 0.0:
                    position = 1
                    entry_price = close[i]
                    stop = close[i] - k * atr[i]   # initial protective stop
                    armed = False
                    bars_held = 0
                    raw[i] = 1
            else:
                bars_held += 1

                # Breakeven: once +X% is touched, lift the stop to entry (up only).
                if not armed and high[i] >= entry_price * (1.0 + _BREAKEVEN_PCT):
                    armed = True
                    if entry_price > stop:
                        stop = entry_price

                # Trail: after breakeven, ratchet the stop up by k*ATR; never down.
                if armed and np.isfinite(atr[i]) and atr[i] > 0.0:
                    new_stop = close[i] - k * atr[i]
                    if new_stop > stop:
                        stop = new_stop

                # Exit checks.
                if low[i] <= stop:
                    position = 0
                    raw[i] = 0
                elif bars_held >= _MAX_HOLD:
                    position = 0
                    raw[i] = 0
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
