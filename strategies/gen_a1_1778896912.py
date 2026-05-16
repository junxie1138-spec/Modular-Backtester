from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    capacity: float = 0.06
    service_rate: float = 1.0
    drain_frac: float = 0.4
    k_atr: float = 2.5
    saturation_lookback: int = 10
    atr_period: int = 14
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778896912"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        # 200-day regime MA dominates; pad for ATR window, saturation lookback,
        # and the single NaN introduced by pct_change.
        return 200 + int(params.atr_period) + int(params.saturation_lookback) + 1

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Close-to-close returns: the primary signal primitive.
        ret = close.pct_change()

        # Arrivals = down-return magnitude; service = up-return magnitude.
        arrival = (-ret).clip(lower=0.0).fillna(0.0).to_numpy()
        service = (ret.clip(lower=0.0).fillna(0.0) * float(params.service_rate)).to_numpy()

        # Finite-capacity queue: recursive clamp is path-dependent -> explicit loop.
        cap = float(params.capacity)
        n = len(close)
        q = np.zeros(n, dtype=float)
        overflow = np.zeros(n, dtype=bool)
        prev = 0.0
        for i in range(n):
            val = prev + arrival[i] - service[i]
            if val > cap:
                val = cap
                overflow[i] = True
            elif val < 0.0:
                val = 0.0
            q[i] = val
            prev = val

        # ATR for the fixed volatility stop.
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(int(params.atr_period)).mean()

        # 200-day regime filter (the mandated twist).
        ma200 = close.rolling(200).mean()

        # Was the buffer saturated (overflowed) recently?
        overflow_s = pd.Series(overflow, index=data.index)
        recent_overflow = (
            overflow_s.rolling(int(params.saturation_lookback)).max().fillna(0.0) > 0.0
        )

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret.fillna(0.0)
        out["q"] = q
        out["recent_overflow"] = recent_overflow.astype(float)
        out["atr"] = atr
        out["ma200"] = ma200
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        ret = indicators["ret"].to_numpy(dtype=float)
        q = indicators["q"].to_numpy(dtype=float)
        recent_overflow = indicators["recent_overflow"].to_numpy(dtype=float) > 0.0
        atr = indicators["atr"].to_numpy(dtype=float)
        ma200 = indicators["ma200"].to_numpy(dtype=float)

        cap = float(params.capacity)
        drain_level = float(params.drain_frac) * cap
        k = float(params.k_atr)
        max_hold = int(params.max_hold)

        n = len(close)
        sig = np.zeros(n, dtype=int)

        in_pos = False
        stop_level = 0.0
        bars_held = 0

        for i in range(n):
            regime_ok = (not np.isnan(ma200[i])) and close[i] > ma200[i]
            entry_cond = (
                (not in_pos)
                and recent_overflow[i]
                and q[i] < drain_level
                and ret[i] > 0.0
                and regime_ok
            )

            if in_pos:
                bars_held += 1
                # Fixed volatility stop (entry-anchored, not trailing) + horizon cap.
                if close[i] < stop_level or bars_held >= max_hold:
                    in_pos = False
                    bars_held = 0
                    sig[i] = 0
                else:
                    sig[i] = 1
            elif entry_cond:
                atr_i = atr[i]
                if np.isnan(atr_i) or atr_i <= 0.0:
                    sig[i] = 0
                else:
                    in_pos = True
                    bars_held = 0
                    stop_level = close[i] - k * atr_i
                    sig[i] = 1
            else:
                sig[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
