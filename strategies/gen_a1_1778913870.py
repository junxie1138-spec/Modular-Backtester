from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


# --- fixed mechanism constants (kept off the tunable surface) ---
_VOL_WINDOW = 20          # bars for the realized-volatility estimate
_ACC_SMOOTH_SPAN = 3      # EMA span applied to ROC acceleration
_ATR_WINDOW = 14          # bars for ATR
_INIT_ATR_MULT = 2.5      # initial protective stop distance (ATR multiples)
_TRAIL_ATR_MULT = 3.0     # post-breakeven trailing distance (ATR multiples)
_BREAKEVEN_PCT = 0.03     # favorable move that arms the breakeven stop
_SIZE_FLOOR = 0.10
_SIZE_CAP = 1.00
_ANNUALISER = float(np.sqrt(252.0))


@dataclass(slots=True)
class PredatorPreyParams:
    roc_window: int = 10
    vol_target: float = 0.15


class GeneratedStrategy(BaseStrategy[PredatorPreyParams]):
    strategy_id = "gen_a1_1778913870"

    @classmethod
    def params_type(cls) -> type[PredatorPreyParams]:
        return PredatorPreyParams

    @staticmethod
    def warmup_bars(params: PredatorPreyParams) -> int:
        w = max(int(params.roc_window), 2)
        return int(max(w + _ACC_SMOOTH_SPAN + 2,
                       _VOL_WINDOW + 2,
                       _ATR_WINDOW + 2,
                       30))

    @staticmethod
    def indicators(data: pd.DataFrame, params: PredatorPreyParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        w = max(int(params.roc_window), 2)

        # prey level: momentum (rate of change over the lookback)
        roc = close.pct_change(w)
        # rate-of-change acceleration: change in momentum, then EMA-smoothed
        acc = roc.diff()
        acc_s = acc.ewm(span=_ACC_SMOOTH_SPAN, adjust=False).mean()

        # cycle-phase direction: sign of acceleration = prey-population slope
        direction = np.sign(acc_s)
        direction = direction.fillna(0.0)

        # realized volatility for the volatility-targeting core
        ret = close.pct_change()
        vol = ret.rolling(_VOL_WINDOW).std() * _ANNUALISER

        # ATR for the breakeven-then-trail exit geometry
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(_ATR_WINDOW).mean()

        out = pd.DataFrame(index=data.index)
        out["roc"] = roc
        out["acc_s"] = acc_s
        out["direction"] = direction
        out["vol"] = vol
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: PredatorPreyParams) -> SignalFrame:
        idx = data.index
        n = len(idx)

        close = data["close"].to_numpy(dtype=float)
        direction = indicators["direction"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        vol = indicators["vol"].to_numpy(dtype=float)

        warmup = GeneratedStrategy.warmup_bars(params)
        raw_signal = np.zeros(n, dtype=np.int64)

        pos = 0
        entry = 0.0
        stop = 0.0
        armed = False

        for t in range(n):
            a = atr[t]
            price = close[t]

            # no valid exit geometry yet -> stay flat
            if t < warmup or not np.isfinite(a) or a <= 0.0 \
                    or not np.isfinite(price) or price <= 0.0:
                pos = 0
                armed = False
                raw_signal[t] = 0
                continue

            d = direction[t]
            desired = 1 if d > 0.0 else (-1 if d < 0.0 else 0)
            exited = False

            if pos != 0:
                # --- breakeven-then-trail stop update (stop only moves favorably) ---
                if pos == 1:
                    gain = (price - entry) / entry
                    if not armed and gain >= _BREAKEVEN_PCT:
                        stop = max(stop, entry)
                        armed = True
                    if armed:
                        stop = max(stop, price - _TRAIL_ATR_MULT * a)
                    hit = price <= stop
                else:
                    gain = (entry - price) / entry
                    if not armed and gain >= _BREAKEVEN_PCT:
                        stop = min(stop, entry)
                        armed = True
                    if armed:
                        stop = min(stop, price + _TRAIL_ATR_MULT * a)
                    hit = price >= stop

                if hit:
                    # exit fires -> force flat this bar
                    pos = 0
                    armed = False
                    exited = True
                elif desired != 0 and desired != pos:
                    # predator-prey cycle flipped phase -> reverse
                    pos = desired
                    entry = price
                    armed = False
                    stop = entry - pos * _INIT_ATR_MULT * a

            if pos == 0 and desired != 0 and not exited:
                pos = desired
                entry = price
                armed = False
                stop = entry - pos * _INIT_ATR_MULT * a

            raw_signal[t] = pos

        df = pd.DataFrame(index=idx)
        sig = pd.Series(raw_signal, index=idx)
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1
        df["signal"] = sig.shift(1).fillna(0).astype(int)

        # volatility-targeted position size
        size = pd.Series(params.vol_target, index=idx) / pd.Series(vol, index=idx)
        size = size.replace([np.inf, -np.inf], np.nan)
        size = size.clip(lower=_SIZE_FLOOR, upper=_SIZE_CAP)
        size = size.fillna(_SIZE_FLOOR)
        df["size"] = size.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
