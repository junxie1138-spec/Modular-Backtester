from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_len: int = 20
    pct_len: int = 252
    entry_pct: float = 0.10
    atr_len: int = 14
    trail_k: float = 3.0
    max_hold: int = 20
    refractory: int = 5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778908870"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.ma_len + params.pct_len + 1, params.atr_len + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        ma_len = max(int(params.ma_len), 2)
        pct_len = max(int(params.pct_len), 2)
        atr_len = max(int(params.atr_len), 2)

        ma = close.rolling(ma_len, min_periods=ma_len).mean()
        sd = close.rolling(ma_len, min_periods=ma_len).std()
        sd = sd.replace(0.0, np.nan)
        z = (close - ma) / sd

        # Hard twist: percentile rank of the z-score within its own recent
        # history -- the entry threshold self-calibrates instead of a fixed level.
        z_pct = z.rolling(pct_len, min_periods=pct_len).rank(pct=True)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_len, min_periods=atr_len).mean()

        out = pd.DataFrame(index=data.index)
        out["z"] = z
        out["z_pct"] = z_pct
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        z_pct = indicators["z_pct"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        entry_pct = float(params.entry_pct)
        trail_k = float(params.trail_k)
        max_hold = max(int(params.max_hold), 1)
        refractory = max(int(params.refractory), 0)

        sig = np.zeros(n, dtype=np.int64)
        position = 0
        bars_held = 0
        entry_high = 0.0
        stop_level = 0.0
        lockout = 0

        for i in range(n):
            ready = np.isfinite(z_pct[i]) and np.isfinite(atr[i]) and np.isfinite(close[i])
            if position == 0:
                if lockout > 0:
                    lockout -= 1
                if ready and lockout == 0 and z_pct[i] <= entry_pct:
                    position = 1
                    bars_held = 0
                    entry_high = close[i]
                    stop_level = close[i] - trail_k * atr[i]
            else:
                bars_held += 1
                if np.isfinite(close[i]) and close[i] > entry_high:
                    entry_high = close[i]
                if np.isfinite(atr[i]):
                    new_stop = entry_high - trail_k * atr[i]
                    # Ratchet only upward -- the trailing stop never loosens.
                    if new_stop > stop_level:
                        stop_level = new_stop
                hit_stop = np.isfinite(close[i]) and close[i] <= stop_level
                if hit_stop or bars_held >= max_hold:
                    position = 0
                    lockout = refractory
            sig[i] = position

        df = pd.DataFrame(index=data.index)
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = pd.Series(sig, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
