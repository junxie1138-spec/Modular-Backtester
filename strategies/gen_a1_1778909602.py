from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMemoryParams:
    ac_window: int = 40
    pos_smooth: int = 5
    ac_threshold: float = 0.15
    pos_high: float = 0.60
    pos_low: float = 0.40
    ma_regime: int = 200
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[RangeMemoryParams]):
    strategy_id = "gen_a1_1778909602"

    @classmethod
    def params_type(cls):
        return RangeMemoryParams

    @staticmethod
    def warmup_bars(params: RangeMemoryParams) -> int:
        return int(max(params.ma_regime, params.ac_window + params.pos_smooth + 1)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeMemoryParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Close location within the daily high-low range, in [0, 1].
        rng = (high - low)
        rng = rng.where(rng > 0.0, np.nan)
        close_pos = ((close - low) / rng).clip(0.0, 1.0).fillna(0.5)

        # Lag-1 autocorrelation of the close-within-range series (regime gauge).
        ac = close_pos.rolling(params.ac_window).corr(close_pos.shift(1))
        ac = ac.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Smoothed recent close-within-range strength.
        pos_smooth = close_pos.rolling(params.pos_smooth).mean().fillna(0.5)

        # 200-day regime moving average.
        ma = close.rolling(params.ma_regime).mean()

        out = pd.DataFrame(index=data.index)
        out["close_pos"] = close_pos
        out["ac"] = ac
        out["pos_smooth"] = pos_smooth
        out["ma"] = ma
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeMemoryParams) -> SignalFrame:
        idx = data.index
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        ac = indicators["ac"].to_numpy(dtype=float)
        sp = indicators["pos_smooth"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)

        # Raw desired direction per bar.
        raw = np.zeros(n, dtype=int)
        thr = float(params.ac_threshold)
        for i in range(n):
            m = ma[i]
            if not np.isfinite(m):
                continue
            a = ac[i]
            p = sp[i]
            if not (np.isfinite(a) and np.isfinite(p)):
                continue
            direction = 0
            if a > thr:
                # Persistence regime: ride current intraday strength.
                if p > params.pos_high:
                    direction = 1
                elif p < params.pos_low:
                    direction = -1
            elif a < -thr:
                # Alternation regime: fade current intraday strength.
                if p < params.pos_low:
                    direction = 1
                elif p > params.pos_high:
                    direction = -1
            # |ac| inside the noise band -> no trade (signal-to-noise filter).
            if direction == 1 and close[i] > m:
                raw[i] = 1
            elif direction == -1 and close[i] < m:
                raw[i] = -1

        # Fixed-bar exit: hold exactly hold_bars bars, no signal-based exit.
        hold = max(1, int(params.hold_bars))
        pos = np.zeros(n, dtype=int)
        current = 0
        held = 0
        for i in range(n):
            if current != 0:
                pos[i] = current
                held += 1
                if held >= hold:
                    current = 0
            elif raw[i] != 0:
                current = raw[i]
                held = 1
                pos[i] = current

        # Conviction-scaled size from autocorrelation magnitude; always positive.
        size = np.clip(0.5 + np.abs(np.nan_to_num(ac, nan=0.0)), 0.5, 1.5)

        df = pd.DataFrame(index=idx)
        df["signal"] = pos
        df["size"] = size.astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
